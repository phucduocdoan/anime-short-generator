"""Load/save project.json atomically (SPEC §3)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from anime_short.schemas import ProjectSpec

PROJECT_FILE = "project.json"
SUBDIRS = ("audio", "images", "clips", "subtitles", "output")


def init_project_dir(project_dir: Path) -> None:
    for sub in SUBDIRS:
        (project_dir / sub).mkdir(parents=True, exist_ok=True)


def load_project(project_dir: Path) -> ProjectSpec:
    return ProjectSpec.model_validate_json((project_dir / PROJECT_FILE).read_text(encoding="utf-8"))


def save_project(project: ProjectSpec, project_dir: Path) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    data = project.model_dump_json(indent=2)
    fd, tmp = tempfile.mkstemp(dir=project_dir, prefix=".project.", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, project_dir / PROJECT_FILE)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
