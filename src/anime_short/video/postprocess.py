"""Shared clip encoding (SPEC §7).

Every `VideoGenerator` returns a clip encoded here: project width x height, project fps, exact
`duration` (frame-accurate — `round(duration * fps)` frames), H.264 yuv420p, no audio. Generators
produce the frames themselves, already at the exact target size and count (still_motion and
animegen both resample sub-pixel crop boxes in Python), and `encode_raw_frames` pipes them
straight into ffmpeg.
"""

from __future__ import annotations

import contextlib
import subprocess
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from anime_short.schemas import VideoResult
from anime_short.utils.media import FFmpegError, check_ffmpeg

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


def encode_raw_frames(
    frames: Iterable[np.ndarray],
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    crf: int = DEFAULT_CRF,
) -> VideoResult:
    """Pipe RGB24 frames (already exactly `width`x`height`, already the exact count needed) to
    ffmpeg and encode them to H.264."""
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
