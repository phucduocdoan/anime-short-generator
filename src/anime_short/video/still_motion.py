"""CPU fallback video generator (SPEC §7.2): animates a single keyframe with a smooth,
sub-pixel camera move plus gentle secondary motion. This is the generator actually used for most
shots right now, since GPU VRAM is shared with other services.

Jitter-free motion: ffmpeg's `zoompan` steps the crop position/size at integer pixel/frame
granularity, which is what causes the classic "juddering" Ken Burns look. Instead we render an
oversampled canvas once (Lanczos upscale of the keyframe) and, per output frame, resample a
*float* crop box straight out of that canvas with `PIL.Image.resize(..., box=...)` — Pillow's box
argument accepts sub-pixel floats, so the crop position and size move continuously frame to frame.
"""

from __future__ import annotations

import math
import os
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from anime_short.schemas import CameraMove, VideoResult
from anime_short.video import postprocess

# Motion magnitudes. Not exposed in config: SPEC only calls out oversample/grain/vignette as
# tunables; these constants just define what "push_in" etc. look like.
ZOOM_RANGE = 0.12  # push_in/pull_out zoom sweep
STATIC_ZOOM_RANGE = 0.035  # "static" still gets a very slow push-in
PAN_ZOOM = 0.08  # fixed zoom held during pan/tilt moves, so there is crop margin to sweep across
BREATH_AMPLITUDE = 0.012  # secondary "breathing" scale oscillation (fraction of zoom)
DRIFT_NORM = 0.06  # secondary drift, as a fraction of the available crop margin

# Per-frame rendering (Lanczos resize + grain) is independent frame-to-frame and dominated by C
# code that releases the GIL (Pillow's resampler), so a thread pool gets real wall-clock
# parallelism without the complexity of multiprocessing. Capped well below all cores since this
# machine also runs other services.
MAX_RENDER_WORKERS = 16


def _ease_in_out(t: float) -> float:
    """Smoothstep. Zero velocity at both ends, strictly monotonic in between -> no jitter."""
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def _cover_resize(img: Image.Image, canvas_w: int, canvas_h: int) -> Image.Image:
    """Lanczos-resize `img` to cover `canvas_w`x`canvas_h` (like CSS background-size: cover),
    then center-crop to that exact size. This is a one-off setup step, done once per clip."""
    src_w, src_h = img.size
    scale = max(canvas_w / src_w, canvas_h / src_h)
    new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - canvas_w) // 2
    top = (new_h - canvas_h) // 2
    return img.crop((left, top, left + canvas_w, top + canvas_h))


def _vignette_mask(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = (width - 1) / 2, (height - 1) / 2
    r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    r = np.clip(r / 1.4, 0.0, 1.0)
    return (1.0 - strength * r**2).astype(np.float32)


def _path(t: float, camera: CameraMove) -> tuple[float, float, float]:
    """Return (zoom, px, py) for progress `t` in [0, 1]. px/py are normalised pan offsets in
    [-1, 1] (0 = centred) applied against the margin available at that zoom level."""
    eased = _ease_in_out(t)
    if camera == CameraMove.push_in:
        return 1.0 + ZOOM_RANGE * eased, 0.0, 0.0
    if camera == CameraMove.pull_out:
        return (1.0 + ZOOM_RANGE) - ZOOM_RANGE * eased, 0.0, 0.0
    if camera == CameraMove.pan_left:
        return 1.0 + PAN_ZOOM, 1.0 - 2.0 * eased, 0.0
    if camera == CameraMove.pan_right:
        return 1.0 + PAN_ZOOM, -1.0 + 2.0 * eased, 0.0
    if camera == CameraMove.tilt_up:
        return 1.0 + PAN_ZOOM, 0.0, 1.0 - 2.0 * eased
    if camera == CameraMove.tilt_down:
        return 1.0 + PAN_ZOOM, 0.0, -1.0 + 2.0 * eased
    # static: nothing should be fully frozen -> very slow push-in
    return 1.0 + STATIC_ZOOM_RANGE * eased, 0.0, 0.0


def _secondary(t: float, camera: CameraMove) -> tuple[float, float]:
    """Gentle breathing/drift, on top of the primary path. A single smooth hump
    (sin(pi*t)^2: zero value AND zero velocity at both ends, like `_ease_in_out`) so it never
    fights the primary move's ease-in/out with a competing velocity at the boundary."""
    hump = math.sin(math.pi * t) ** 2
    if camera in (CameraMove.pan_left, CameraMove.pan_right):
        return 0.0, DRIFT_NORM * hump  # drift on the axis the pan isn't using
    if camera in (CameraMove.tilt_up, CameraMove.tilt_down):
        return DRIFT_NORM * hump, 0.0
    return DRIFT_NORM * hump, DRIFT_NORM * 0.6 * math.cos(math.pi * t)


def camera_boxes(
    canvas_w: float, canvas_h: float, camera: CameraMove, n_frames: int
) -> list[tuple[float, float, float, float]]:
    """Precompute the (left, top, right, bottom) source crop box for every frame of the shot's
    camera move. Cheap and sequential, so per-frame rendering can be parallelised afterwards.
    Also used by animegen, which applies the camera move to the generated frames in post."""
    canvas_cx, canvas_cy = canvas_w / 2, canvas_h / 2
    boxes = []
    for i in range(n_frames):
        t = i / (n_frames - 1) if n_frames > 1 else 0.0
        zoom, px, py = _path(t, camera)
        zoom *= 1.0 + BREATH_AMPLITUDE * math.sin(math.pi * t) ** 2
        drift_x, drift_y = _secondary(t, camera)
        px = max(-1.0, min(1.0, px + drift_x))
        py = max(-1.0, min(1.0, py + drift_y))

        # zoom 1 = the whole canvas (whole keyframe); oversample only adds resampling headroom
        box_w, box_h = canvas_w / zoom, canvas_h / zoom
        margin_x = max(0.0, (canvas_w - box_w) / 2)
        margin_y = max(0.0, (canvas_h - box_h) / 2)
        cx = canvas_cx + px * margin_x
        cy = canvas_cy + py * margin_y
        boxes.append((cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2))
    return boxes


class StillMotionGenerator:
    """Fallback `VideoGenerator` (SPEC §7.2). CPU only."""

    name = "still_motion"

    def __init__(
        self,
        still_motion_config: dict[str, Any] | None = None,
        width: int = 1080,
        height: int = 1920,
        fps: int = 24,
    ) -> None:
        config = still_motion_config or {}
        self.oversample = float(config.get("oversample", 1.4))
        self.grain = float(config.get("grain", 0.03))
        self.vignette_strength = float(config.get("vignette", 0.15))
        self.width = width
        self.height = height
        self.fps = fps

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def _frames(
        self, canvas: Image.Image, camera: CameraMove, n_frames: int, seed: int
    ) -> list[np.ndarray]:
        canvas_w, canvas_h = canvas.size
        boxes = camera_boxes(canvas_w, canvas_h, camera, n_frames)
        vignette = _vignette_mask(self.width, self.height, self.vignette_strength)
        grain_sigma = self.grain * 255.0
        # One independent RNG stream per frame (not one shared generator) so the thread pool
        # below can draw grain concurrently without a data race.
        rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n_frames)]

        def render(i: int) -> np.ndarray:
            frame = canvas.resize((self.width, self.height), Image.LANCZOS, box=boxes[i])
            arr = np.asarray(frame, dtype=np.float32)
            if grain_sigma > 0:
                arr += rngs[i].standard_normal(arr.shape, dtype=np.float32) * grain_sigma
            arr *= vignette[..., None]
            return np.clip(arr, 0, 255).astype(np.uint8)

        workers = min(MAX_RENDER_WORKERS, os.cpu_count() or 1, n_frames)
        if workers <= 1:
            return [render(i) for i in range(n_frames)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(render, range(n_frames)))

    def generate(
        self,
        image: Path,
        prompt: str,
        camera: CameraMove,
        duration: float,
        output_path: Path,
    ) -> VideoResult:
        n_frames = postprocess.target_frame_count(duration, self.fps)
        canvas_w = round(self.width * self.oversample)
        canvas_h = round(self.height * self.oversample)
        source = Image.open(image).convert("RGB")
        canvas = _cover_resize(source, canvas_w, canvas_h)
        seed = zlib.crc32(str(output_path).encode()) & 0xFFFFFFFF
        frames = self._frames(canvas, camera, max(1, n_frames), seed)
        return postprocess.encode_raw_frames(frames, output_path, self.width, self.height, self.fps)
