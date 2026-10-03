"""Deterministic stage runner (SPEC §2, §13). Orchestration is dumb; stage modules are smart."""

from __future__ import annotations

import importlib
import logging
import re
import unicodedata
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml
from filelock import FileLock, Timeout

from anime_short.config import AppConfig, load_config, load_yaml
from anime_short.pipeline.state import init_project_dir, load_project, save_project
from anime_short.schemas import (
    STAGES,
    Character,
    ContentSpec,
    ProjectSpec,
    StageStatus,
    VoiceSpec,
)

log = logging.getLogger(__name__)

SPEC_FILE = "spec.yaml"
LOCK_FILE = ".lock"


class ProjectLocked(RuntimeError):
    """Raised when another anime-short process already holds this project's lock."""


# stage -> "module:function"; imported lazily so the CLI does not pull in torch for every command.
STAGE_FUNCTIONS: dict[str, str] = {
    "script": "anime_short.script.stage:run_script",
    "plan": "anime_short.script.stage:run_plan",
    "tts": "anime_short.tts.stage:run_tts",
    "timeline": "anime_short.pipeline.timeline:run_timeline",
    "images": "anime_short.image.stage:run_images",
    "videos": "anime_short.video.stage:run_videos",
    "subtitles": "anime_short.subtitle.stage:run_subtitles",
    "music": "anime_short.music.stage:run_music",
    "render": "anime_short.render.stage:run_render",
}

StageFn = Callable[[ProjectSpec, Path, AppConfig, bool], None]


def stage_function(stage: str) -> StageFn:
    module_name, func_name = STAGE_FUNCTIONS[stage].split(":")
    return getattr(importlib.import_module(module_name), func_name)


def slugify(text: str, max_len: int = 40) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:max_len].rstrip("_") or "project"


# --- spec.yaml -> (ProjectSpec, config overrides) --------------------------------------------


def spec_to_project(spec: dict[str, Any]) -> ProjectSpec:
    project = spec.get("project", {})
    content = dict(spec["content"])
    language = content.pop("language", "vi")
    title = project.get("title") or content["topic"].strip()
    output = spec.get("output", {})
    return ProjectSpec(
        id=project.get("id") or slugify(title),
        title=title,
        language=language,
        width=output.get("width", 1080),
        height=output.get("height", 1920),
        fps=spec.get("video", {}).get("fps", 24),
        style=spec.get("visual", {}).get("style", "cinematic_anime"),
        content=ContentSpec.model_validate(content),
        voice=VoiceSpec.model_validate(spec.get("voice", {})),
        characters={
            cid: Character.model_validate(c) for cid, c in spec.get("characters", {}).items()
        },
    )


def spec_to_overrides(spec: dict[str, Any]) -> dict[str, Any]:
    """Map the user-facing spec.yaml (v0 §17 format) onto configs/default.yaml keys."""
    overrides: dict[str, Any] = {}
    if provider := spec.get("visual", {}).get("provider"):
        overrides["image"] = {"provider": provider}
    if provider := spec.get("video", {}).get("provider"):
        overrides["video"] = {"provider": provider}
    if provider := spec.get("voice", {}).get("provider"):
        overrides["tts"] = {"provider": provider}
    script = spec.get("script", {})
    for key in ("hook_seconds", "words_per_second"):
        if key in script:
            overrides.setdefault("script", {})[key] = script[key]
    if "max_scene_seconds" in script:
        overrides["timeline"] = {"max_shot_seconds": script["max_scene_seconds"]}
    if "timeline" in spec:
        overrides["timeline"] = {**overrides.get("timeline", {}), **spec["timeline"]}
    for section in ("llm", "subtitle", "music", "render"):
        if section in spec:
            overrides[section] = spec[section]
    return overrides


def project_config(project_dir: Path) -> AppConfig:
    """Config for an existing project: defaults + the spec.yaml stored in its directory."""
    spec_path = project_dir / SPEC_FILE
    spec = load_yaml(spec_path) if spec_path.exists() else {}
    return load_config(spec_to_overrides(spec))


def create_project(
    spec: dict[str, Any], config: AppConfig | None = None
) -> tuple[ProjectSpec, Path]:
    config = config or load_config(spec_to_overrides(spec))
    project = spec_to_project(spec)
    project_dir = config.project_dir(project.id)
    if (project_dir / "project.json").exists():
        raise FileExistsError(
            f"project {project.id!r} already exists at {project_dir}; use `resume {project.id}`"
        )
    init_project_dir(project_dir)
    (project_dir / SPEC_FILE).write_text(
        yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    save_project(project, project_dir)
    return project, project_dir


def create_project_from_file(spec_path: Path) -> tuple[ProjectSpec, Path]:
    return create_project(load_yaml(spec_path))


# --- running ---------------------------------------------------------------------------------


@contextmanager
def _project_lock(project_dir: Path, stage: str | None = None) -> Iterator[None]:
    """Exclusive, non-blocking lock for one project directory: two `anime-short` processes must
    never run stages against the same project at once, or the later one's whole-document save
    silently clobbers the other's changes (the `project.music` revert bug). Fails fast instead of
    waiting, since a stuck lock from a killed process should be visible, not block forever.
    `is_singleton=True` makes acquiring the "same" lock from several call sites in one process
    (`run_pending` -> `run_stage`) reentrant, instead of the OS-level flock conflicting with itself.
    """
    lock = FileLock(str(project_dir / LOCK_FILE), is_singleton=True)
    try:
        with lock.acquire(timeout=0):
            yield
    except Timeout:
        detail = f" (stage {stage!r})" if stage else ""
        raise ProjectLocked(
            f"project {project_dir.name!r} is in use by another anime-short process{detail}"
        ) from None


def run_stage(
    project: ProjectSpec, project_dir: Path, config: AppConfig, stage: str, force: bool = False
) -> None:
    with _project_lock(project_dir, stage):
        log.info("stage %s: start", stage)
        # Any run may regenerate items (a missing or invalidated artifact, or all of them with
        # --force), so every later stage must be re-checked; their own skips keep that cheap.
        for later in STAGES[STAGES.index(stage) + 1 :]:
            project.stages[later] = StageStatus.pending
        try:
            stage_function(stage)(project, project_dir, config, force)
        except BaseException:
            project.stages[stage] = StageStatus.failed
            save_project(project, project_dir)
            log.exception("stage %s: failed", stage)
            raise
        project.stages[stage] = StageStatus.done
        save_project(project, project_dir)
        log.info("stage %s: done", stage)


def run_pending(project_dir: Path, force: bool = False) -> ProjectSpec:
    """Run every stage not marked done, in order (used by `generate` and `resume`). Locked for the
    whole sequence, not just per-stage, so another process can't slip in between two stages."""
    with _project_lock(project_dir):
        config = project_config(project_dir)
        project = load_project(project_dir)
        for stage in STAGES:
            if force or project.stages.get(stage) != StageStatus.done:
                run_stage(project, project_dir, config, stage, force)
        return project
