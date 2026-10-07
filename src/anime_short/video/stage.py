"""`videos` pipeline stage (SPEC §7, §12): one clip per shot, batched behind a single GPU session
(or none, for the CPU fallback) so models are never interleaved on the shared GPU."""

from __future__ import annotations

import logging
from collections.abc import Callable
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


def _needs_generation(shot: Shot, project_dir: Path, force: bool) -> bool:
    if shot.image is None or shot.duration is None:
        return False
    # State-backed skip: images/timeline clear `shot.video` when the clip goes stale.
    done = shot.video is not None and (project_dir / shot.video.path).exists()
    return force or not done


def _keeps_keyframe(shot: Shot) -> bool:
    """Shots with on-screen text, or a screen in a close-up or in the foreground, are animated
    with still_motion: i2v rewrites screens and writing into gibberish even when the prompt says
    they stay unchanged (checked on real clips). A screen held in the background of a wider shot
    is too small to matter."""
    v = shot.visual
    described = " ".join((v.subject, v.action, v.setting, v.extra)).lower()
    large_screen = "close-up" in v.shot_type.lower() or "foreground" in described
    return bool(v.text) or ("screen" in described and large_screen)


def _run_shots(
    generator: VideoGenerator,
    mode: VideoMode,
    provider: str,
    project: ProjectSpec,
    project_dir: Path,
    force: bool,
    select: Callable[[Shot], bool],
) -> None:
    clips_dir = project_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    for scene in project.scenes:
        for shot in scene.shots:
            if not _needs_generation(shot, project_dir, force) or not select(shot):
                continue
            out_path = clips_dir / f"{shot.id}.mp4"

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
    i2v_ok = False
    allow_fallback = bool(config.video.animegen.get("allow_still_motion_fallback", True))

    if provider == "animegen" and not any(
        _needs_generation(s, project_dir, force) and not _keeps_keyframe(s)
        for sc in project.scenes
        for s in sc.shots
    ):
        i2v_ok = True  # nothing left to animate with i2v; don't load the model
    elif provider == "animegen":
        service_mode = config.video.animegen.get("service_mode", "on_demand")
        try:
            if service_mode == "resident":
                from anime_short.video.service import AnimeGenServiceGenerator

                generator = AnimeGenServiceGenerator(
                    service_url=config.video.animegen.get(
                        "service_url", "http://127.0.0.1:28180"
                    ),
                    timeout_seconds=float(
                        config.video.animegen.get("service_timeout_seconds", 3600)
                    ),
                    width=project.width,
                    height=project.height,
                    fps=project.fps,
                )
                _run_shots(
                    generator,
                    VideoMode.i2v,
                    "animegen",
                    project,
                    project_dir,
                    force,
                    lambda shot: not _keeps_keyframe(shot),
                )
                i2v_ok = True
            elif service_mode == "on_demand":
                generator = get_video_generator(config, "animegen")
                generator.width, generator.height, generator.fps = (
                    project.width,
                    project.height,
                    project.fps,
                )
                min_free_gb = config.video.animegen.get("min_free_vram_gb", 24)
                with gpu_session(generator, min_free_gb=min_free_gb):
                    _run_shots(
                        generator,
                        VideoMode.i2v,
                        "animegen",
                        project,
                        project_dir,
                        force,
                        lambda shot: not _keeps_keyframe(shot),
                    )
                i2v_ok = True
            else:
                raise ValueError(f"unknown animegen service_mode: {service_mode}")
        except InsufficientVRAM as exc:
            if not allow_fallback:
                raise RuntimeError(f"animegen unavailable and fallback is disabled: {exc}") from exc
            log.warning("videos: animegen unavailable, falling back to still_motion: %s", exc)
        except Exception as exc:  # noqa: BLE001 - narrowed to OOM below, re-raised otherwise
            if not _is_oom(exc):
                if not allow_fallback:
                    raise RuntimeError(
                        f"animegen unavailable and fallback is disabled: {exc}"
                    ) from exc
                log.warning("videos: animegen unavailable, falling back to still_motion: %s", exc)
            else:
                if not allow_fallback:
                    raise RuntimeError(
                        f"animegen ran out of VRAM and fallback is disabled: {exc}"
                    ) from exc
                log.warning(
                    "videos: animegen ran out of VRAM loading the model, falling back: %s", exc
                )

    fallback = get_video_generator(config, "still_motion")
    fallback.width, fallback.height, fallback.fps = project.width, project.height, project.fps
    select = _keeps_keyframe if i2v_ok else (lambda shot: True)
    _run_shots(
        fallback, VideoMode.still_motion, "still_motion", project, project_dir, force, select
    )
