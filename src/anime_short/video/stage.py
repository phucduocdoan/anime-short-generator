"""`videos` pipeline stage (SPEC §7, §12): one clip per shot, batched behind a single GPU session
(or none, for the CPU fallback) so models are never interleaved on the shared GPU."""

from __future__ import annotations

import logging
from pathlib import Path

from anime_short.config import AppConfig
from anime_short.pipeline.state import save_project
from anime_short.schemas import ProjectSpec, Shot, VideoInfo, VideoMode
from anime_short.utils.gpu import InsufficientVRAM, gpu_session
from anime_short.video import get_video_generator
from anime_short.video.base import VideoGenerator

log = logging.getLogger(__name__)


def _is_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    return "out of memory" in str(exc).lower()


def _needs_generation(shot: Shot) -> bool:
    return shot.image is not None and shot.duration is not None


def _run_shots(
    generator: VideoGenerator,
    mode: VideoMode,
    provider: str,
    project: ProjectSpec,
    project_dir: Path,
    force: bool,
) -> None:
    clips_dir = project_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    for scene in project.scenes:
        for shot in scene.shots:
            if not _needs_generation(shot):
                continue
            out_path = clips_dir / f"{shot.id}.mp4"
            # State-backed skip: images/timeline clear `shot.video` when the clip goes stale.
            done = shot.video is not None and (project_dir / shot.video.path).exists()
            if done and not force:
                continue

            log.info("videos: %s (%s, %s, %.2fs)", shot.id, provider, shot.camera, shot.duration)
            assert shot.image is not None and shot.duration is not None
            generator.generate(
                image=project_dir / shot.image.path,
                prompt=shot.motion_prompt,
                camera=shot.camera,
                duration=shot.duration,
                output_path=out_path,
            )
            shot.video = VideoInfo(
                path=str(out_path.relative_to(project_dir)), mode=mode, provider=provider
            )
            save_project(project, project_dir)


def run_videos(
    project: ProjectSpec, project_dir: Path, config: AppConfig, force: bool = False
) -> None:
    provider = config.video.provider

    if provider == "animegen":
        generator = get_video_generator(config, "animegen")
        generator.width, generator.height, generator.fps = (
            project.width,
            project.height,
            project.fps,
        )
        min_free_gb = config.video.animegen.get("min_free_vram_gb", 24)
        try:
            with gpu_session(generator, min_free_gb=min_free_gb):
                _run_shots(generator, VideoMode.i2v, "animegen", project, project_dir, force)
            return
        except InsufficientVRAM as exc:
            log.warning("videos: animegen unavailable, falling back to still_motion: %s", exc)
        except Exception as exc:  # noqa: BLE001 - narrowed to OOM below, re-raised otherwise
            if not _is_oom(exc):
                raise
            log.warning("videos: animegen ran out of VRAM loading the model, falling back: %s", exc)

    fallback = get_video_generator(config, "still_motion")
    fallback.width, fallback.height, fallback.fps = project.width, project.height, project.fps
    _run_shots(fallback, VideoMode.still_motion, "still_motion", project, project_dir, force)
