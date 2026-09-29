"""Entry point for the render stage (SPEC §11)."""

from __future__ import annotations

from pathlib import Path

from anime_short.config import AppConfig
from anime_short.pipeline.state import save_project
from anime_short.render.composer import compose
from anime_short.schemas import ProjectSpec, StageStatus


def run_render(
    project: ProjectSpec, project_dir: Path, config: AppConfig, force: bool = False
) -> None:
    if project.stages.get("render") == StageStatus.done and not force:
        return

    final_path = compose(project, project_dir, config)
    project.output = str(final_path.relative_to(project_dir))
    project.stages["render"] = StageStatus.done
    save_project(project, project_dir)
