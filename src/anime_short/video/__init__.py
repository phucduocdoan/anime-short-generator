"""Video generators (SPEC §7)."""

from __future__ import annotations

from anime_short.config import AppConfig
from anime_short.video.base import VideoGenerator


def get_video_generator(config: AppConfig, name: str) -> VideoGenerator:
    if name == "still_motion":
        from anime_short.video.still_motion import StillMotionGenerator

        return StillMotionGenerator(config.video.still_motion)
    if name == "animegen":
        from anime_short.video.animegen import AnimeGenGenerator

        return AnimeGenGenerator(config.video.animegen)
    raise ValueError(f"unknown video provider: {name!r}")
