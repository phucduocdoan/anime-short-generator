"""Entry point for the subtitles stage (SPEC §10)."""

from __future__ import annotations

from pathlib import Path

from anime_short.config import AppConfig
from anime_short.pipeline.state import save_project
from anime_short.schemas import ProjectSpec, StageStatus
from anime_short.subtitle.ass import build_ass_document


def run_subtitles(
    project: ProjectSpec, project_dir: Path, config: AppConfig, force: bool = False
) -> None:
    if project.stages.get("subtitles") == StageStatus.done and not force:
        return
    if not config.subtitle.enabled:
        project.stages["subtitles"] = StageStatus.done
        save_project(project, project_dir)
        return

    document = build_ass_document(project, config)
    out_path = project_dir / "subtitles" / "final.ass"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(document, encoding="utf-8")

    project.stages["subtitles"] = StageStatus.done
    save_project(project, project_dir)
