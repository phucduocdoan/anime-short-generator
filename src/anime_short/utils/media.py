"""Thin wrappers over ffmpeg/ffprobe shared by all stages."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from dotenv import load_dotenv

from anime_short.config import REPO_ROOT

# The ffmpeg first on PATH may be a --disable-gpl build without libx264/libass (e.g. miniconda's).
# Point ANIME_SHORT_FFMPEG / ANIME_SHORT_FFPROBE (env or .env) at a full build.
REQUIRED_ENCODERS = ("libx264",)
REQUIRED_FILTERS = ("ass", "subtitles")


class FFmpegError(RuntimeError):
    pass


def ffmpeg_bin() -> str:
    load_dotenv(REPO_ROOT / ".env")
    return os.environ.get("ANIME_SHORT_FFMPEG", "ffmpeg")


def ffprobe_bin() -> str:
    load_dotenv(REPO_ROOT / ".env")
    return os.environ.get("ANIME_SHORT_FFPROBE", "ffprobe")


@cache
def check_ffmpeg() -> str:
    """Fail fast with a clear message if the configured ffmpeg lacks libx264 or libass."""
    binary = ffmpeg_bin()

    def listing(kind: str) -> str:
        return subprocess.run([binary, "-hide_banner", kind], capture_output=True, text=True).stdout

    encoders, filters = listing("-encoders"), listing("-filters")
    missing = [e for e in REQUIRED_ENCODERS if f" {e} " not in encoders]
    missing += [f for f in REQUIRED_FILTERS if f" {f} " not in filters]
    if missing:
        raise FFmpegError(
            f"{binary} lacks {', '.join(missing)}; set ANIME_SHORT_FFMPEG/ANIME_SHORT_FFPROBE "
            "to a full (GPL) ffmpeg build"
        )
    return binary


def run_ffmpeg(args: list[str]) -> None:
    """Run ffmpeg quietly with `args`; raise FFmpegError carrying stderr on failure."""
    cmd = [check_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"{' '.join(cmd)}\n{proc.stderr.strip()}")


def ffprobe(path: Path) -> dict:
    proc = subprocess.run(
        [
            ffprobe_bin(),
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
