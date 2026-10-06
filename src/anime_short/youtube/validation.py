from __future__ import annotations

from pathlib import Path

from anime_short.utils.media import VideoProbe, probe_video

MAX_SHORT_SECONDS = 180.0


def validate_short_video(path: Path) -> VideoProbe:
    """Validate that a rendered file is eligible to be classified as a YouTube Short."""
    if not path.exists():
        raise FileNotFoundError(path)
    probe = probe_video(path)
    if probe.width > probe.height:
        raise ValueError(
            f"YouTube Shorts need square or vertical video; got {probe.width}x{probe.height}"
        )
    if probe.duration > MAX_SHORT_SECONDS:
        raise ValueError(
            f"YouTube Shorts must be at most 180 seconds; got {probe.duration:.2f}s"
        )
    return probe
