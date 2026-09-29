"""Thin wrappers over ffmpeg/ffprobe shared by all stages."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def run_ffmpeg(args: list[str]) -> None:
    """Run ffmpeg quietly with `args`; raise FFmpegError carrying stderr on failure."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"{' '.join(cmd)}\n{proc.stderr.strip()}")


def ffprobe(path: Path) -> dict:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe {path}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def probe_duration(path: Path) -> float:
    return float(ffprobe(path)["format"]["duration"])


@dataclass(frozen=True)
class VideoProbe:
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool


def probe_video(path: Path) -> VideoProbe:
    info = ffprobe(path)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = video["avg_frame_rate"].split("/")
    return VideoProbe(
        width=int(video["width"]),
        height=int(video["height"]),
        fps=float(num) / float(den) if float(den) else 0.0,
        duration=float(info["format"]["duration"]),
        has_audio=any(s["codec_type"] == "audio" for s in info["streams"]),
    )
