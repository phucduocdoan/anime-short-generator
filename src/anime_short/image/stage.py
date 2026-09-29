"""Images stage (SPEC §6): keyframe per shot via the configured `ImageGenerator`."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from anime_short.config import AppConfig, load_style
from anime_short.image import get_image_generator
from anime_short.image.prompt_builder import build as build_prompt
from anime_short.pipeline.state import save_project
from anime_short.schemas import ImageInfo, ProjectSpec
from anime_short.utils.gpu import gpu_session

log = logging.getLogger(__name__)


def deterministic_seed(project_id: str, shot_id: str) -> int:
    """Stable seed derived from project id + shot id (stable across runs, unlike `hash()`)."""
    digest = hashlib.sha256(f"{project_id}:{shot_id}".encode()).hexdigest()
    return int(digest[:8], 16)


def run_images(
    project: ProjectSpec, project_dir: Path, config: AppConfig, force: bool = False
) -> None:
    style = load_style(project.style)
    generator = get_image_generator(config)
    images_dir = project_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    with gpu_session(generator):
        for scene in project.scenes:
            for shot in scene.shots:
                output_path = images_dir / f"{shot.id}.png"
                already_done = shot.image is not None and (project_dir / shot.image.path).exists()
                if already_done and not force:
                    log.info("skip %s (exists)", shot.id)
                    continue

                prompt, negative = build_prompt(
                    shot, project.characters, style, generator.prompt_style
                )
                seed = (
                    shot.seed if shot.seed is not None else deterministic_seed(project.id, shot.id)
                )

                result = generator.generate(
                    prompt=prompt,
                    negative_prompt=negative,
                    width=config.image.width,
                    height=config.image.height,
                    seed=seed,
                    output_path=output_path,
                )

                shot.image = ImageInfo(
                    path=str(output_path.relative_to(project_dir)),
                    seed=result.seed,
                    provider=generator.name,
                    prompt=prompt,
                    negative_prompt=negative,
                )
                shot.video = None  # the clip was animated from the old keyframe
                save_project(project, project_dir)
