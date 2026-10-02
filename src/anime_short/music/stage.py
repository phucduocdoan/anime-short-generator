"""Music stage: generate one background track per project, long enough for the whole video
(`music.provider: acestep`). With no provider, the render stage uses the fixed `music.path`."""

from __future__ import annotations

import logging
from pathlib import Path

from anime_short.config import AppConfig
from anime_short.image.stage import deterministic_seed
from anime_short.music.acestep import AceStepClient
from anime_short.pipeline.state import save_project
from anime_short.schemas import ProjectSpec

log = logging.getLogger(__name__)

MUSIC_FILE = "music/music.mp3"
# Generated tracks end abruptly; the render trims and fades out well before the track's end.
EXTRA_SECONDS = 8.0


def total_duration(project: ProjectSpec) -> float:
    shots = [shot for scene in project.scenes for shot in scene.shots]
    if not shots or any(shot.start is None or shot.duration is None for shot in shots):
        raise ValueError("run the timeline stage before music")
    return max(shot.start + shot.duration for shot in shots)


def run_music(
    project: ProjectSpec, project_dir: Path, config: AppConfig, force: bool = False
) -> None:
    music = config.music
    if not (music.enabled and music.provider):
        return
    if project.music and (project_dir / project.music).exists() and not force:
        return
    if music.provider != "acestep":
        raise ValueError(f"unknown music provider {music.provider!r}")
    duration = round(total_duration(project) + EXTRA_SECONDS)
    client = AceStepClient(music.acestep)
    try:
        client.generate(duration, deterministic_seed(project.id, "music"), project_dir / MUSIC_FILE)
    finally:
        client.close()
    project.music = MUSIC_FILE
    save_project(project, project_dir)
