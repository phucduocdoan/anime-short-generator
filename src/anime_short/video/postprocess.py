"""Shared clip normalisation (SPEC §7).

Every `VideoGenerator` returns a clip that has already gone through here: project width x height,
project fps, exact `duration` (frame-accurate — `round(duration * fps)` frames), H.264 yuv420p,
no audio. Two entry points, because the two generators produce raw output differently:

- `normalize_clip`: for a generator whose raw output is an existing video file at a different
  resolution/fps (e.g. animegen's 16 fps 480x832 clip) — fps-convert, lanczos scale+crop to
  project size, trim to the exact frame count, strip audio, re-encode.
- `encode_raw_frames`: for a generator that produces frames itself, already at the exact target
  size and count (e.g. still_motion, which computes sub-pixel crops in Python) — pipe them
  straight into the same H.264 encode settings.

A source shorter than `duration` (animegen caps at `max_frames`, ~5.06s) is slowed down with
`setpts` to fill it; a longer one is trimmed.
"""

from __future__ import annotations

import contextlib
import subprocess
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from anime_short.schemas import VideoResult
from anime_short.utils.media import FFmpegError, check_ffmpeg, probe_duration, run_ffmpeg

DEFAULT_CRF = 14


def target_frame_count(duration: float, fps: int) -> int:
    return round(duration * fps)


def _x264_args(crf: int) -> list[str]:
    return [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
    ]


def normalize_clip(
    input_path: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    duration: float,
    crf: int = DEFAULT_CRF,
) -> VideoResult:
    """Fps-convert, lanczos scale/crop to `width`x`height`, stretch or trim to the exact frame
    count for `duration`, strip audio, and re-encode `input_path` to H.264."""
    frames = target_frame_count(duration, fps)
    stretch = max(1.0, duration / probe_duration(input_path))
    vf = (
        f"setpts={stretch:.6f}*PTS,"
        f"fps={fps},"
        f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{height},"
        "setsar=1"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-i",
            str(input_path),
            "-vf",
            vf,
            "-frames:v",
            str(frames),
            *_x264_args(crf),
            str(output_path),
        ]
    )
    return VideoResult(path=output_path, duration=frames / fps, fps=fps, width=width, height=height)


def encode_raw_frames(
    frames: Iterable[np.ndarray],
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    crf: int = DEFAULT_CRF,
) -> VideoResult:
    """Pipe RGB24 frames (already exactly `width`x`height`, already the exact count needed) to
    ffmpeg and encode with the same H.264 settings as `normalize_clip`."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        check_ffmpeg(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        *_x264_args(crf),
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    count = 0
    try:
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
            count += 1
    except BrokenPipeError:
        pass  # ffmpeg exited early; the real reason is in stderr below
    finally:
        with contextlib.suppress(BrokenPipeError):
            proc.stdin.close()
    stderr = proc.stderr.read() if proc.stderr else b""
    proc.wait()
    if proc.returncode != 0:
        raise FFmpegError(f"{' '.join(cmd)}\n{stderr.decode(errors='replace').strip()}")
    return VideoResult(path=output_path, duration=count / fps, fps=fps, width=width, height=height)
