"""TTS stage (SPEC §8, §13): synthesize each scene's narration into `audio/<scene_id>.wav`."""

from __future__ import annotations

import logging
from pathlib import Path

from anime_short.config import AppConfig
from anime_short.pipeline.state import save_project
from anime_short.schemas import AudioInfo, ProjectSpec
from anime_short.tts import get_tts
from anime_short.utils.gpu import gpu_session

log = logging.getLogger(__name__)


def run_tts(
    project: ProjectSpec, project_dir: Path, config: AppConfig, force: bool = False
) -> None:
    def audio_path(scene_id: str) -> Path:
        return project_dir / "audio" / f"{scene_id}.wav"

    # Skip only what project.json records as done: a wav without state (crash before save, or a
    # leftover from an earlier script) must be regenerated, not silently kept.
    pending = [
        s
        for s in project.scenes
        if force or s.audio is None or not (project_dir / s.audio.path).exists()
    ]
    if not pending:
        log.info("tts: nothing to do (%d scene(s) already synthesized)", len(project.scenes))
        return

    provider = get_tts(config, project.voice)
    with gpu_session(provider, min_free_gb=provider.min_free_vram_gb):
        for scene in pending:
            output_path = audio_path(scene.id)
            result = provider.synthesize(scene.narration, output_path)
            scene.audio = AudioInfo(
                path=str(output_path.relative_to(project_dir)),
                duration=result.duration,
                words=result.words,
                provider=provider.name,
            )
            save_project(project, project_dir)
            log.info(
                "tts: %s -> %.2fs (%s, %d word timing(s))",
                scene.id,
                result.duration,
                provider.name,
                len(result.words),
            )
