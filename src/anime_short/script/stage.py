"""Public entry points the pipeline runner calls for the `script` and `plan` stages (SPEC §2, §5).

Resumability: each function mutates `project` in place and saves after each item, so a killed run
picks up where it left off. `run_script`/`run_plan` own `project_dir` and call `save_project`
themselves; `replan_scene` does not (it has no `project_dir`) — its caller (the `timeline` stage)
saves after it returns.
"""

from __future__ import annotations

from pathlib import Path

from anime_short.config import AppConfig
from anime_short.pipeline.state import save_project
from anime_short.schemas import CameraMove, ProjectSpec, Scene
from anime_short.script.generator import characters_from_draft, generate_script, scenes_from_draft
from anime_short.script.llm import LLMClient
from anime_short.script.scene_planner import plan_scene, shot_count_for_duration


def run_script(
    project: ProjectSpec, project_dir: Path, config: AppConfig, force: bool = False
) -> None:
    if project.scenes and not force:
        return
    llm = LLMClient(config.llm)
    need_characters = not project.characters and project.content.format == "story"
    draft = generate_script(project.content, project.language, config, llm, need_characters)
    if need_characters:
        project.characters.update(characters_from_draft(draft))
    project.scenes = scenes_from_draft(draft)
    project.hook = project.scenes[0].narration if project.scenes else None
    project.ending = draft.ending.strip() or None
    save_project(project, project_dir)


# Quotes videos (ContentSpec.format) are moody frames with the spoken line overlaid mid-frame.
QUOTES_PLAN_CONSTRAINTS = (
    "This is a quotes video over moody retro anime frames. Vary the framing: mostly wide or "
    "medium-wide shots where people are clearly visible in the lower half of the frame (often a "
    "young couple together, sometimes one person), seen from behind, in profile or three-quarter "
    "view; now and then a medium shot of one person in profile looking away, or a place with "
    "nobody. Keep the middle of the frame calm (sky, a wall, a slope) but never mention text, "
    "typography or captions in any field. Light is dim and muted: dusk, night, overcast, a single "
    "warm lamp or low sun; never write pastel, gradient, bright or minimalist. Name one dominant "
    "color in `lighting` and change it from shot to shot (navy blue with rust orange, dark teal, "
    "forest green, olive and ochre, deep red-brown). `text` stays empty."
)


def _format_constraints(project: ProjectSpec) -> str:
    return QUOTES_PLAN_CONSTRAINTS if project.content.format == "quotes" else ""


def run_plan(
    project: ProjectSpec, project_dir: Path, config: AppConfig, force: bool = False
) -> None:
    llm = LLMClient(config.llm)
    for index, scene in enumerate(project.scenes):
        if scene.shots and not force:
            continue
        scene.shots = plan_scene(
            scene,
            project.characters,
            config,
            llm,
            language=project.language,
            previous_camera=_previous_camera(project, index),
            extra_constraints=_format_constraints(project),
        )
        save_project(project, project_dir)


def _previous_camera(project: ProjectSpec, index: int) -> CameraMove | None:
    previous = project.scenes[index - 1].shots if index > 0 else []
    return previous[-1].camera if previous else None


def replan_scene(project: ProjectSpec, scene: Scene, config: AppConfig) -> None:
    if scene.audio is None:
        raise ValueError(f"{scene.id}: cannot replan without a known audio duration")
    duration = scene.audio.duration
    count = shot_count_for_duration(
        duration, config.timeline.min_shot_seconds, config.timeline.max_shot_seconds
    )
    llm = LLMClient(config.llm)
    scene.shots = plan_scene(
        scene,
        project.characters,
        config,
        llm,
        shot_count=count,
        scene_duration=duration,
        language=project.language,
        previous_camera=_previous_camera(project, project.scenes.index(scene)),
        extra_constraints=(
            f"Hard constraint: no single shot may exceed {config.timeline.max_shot_seconds:.1f} "
            "seconds of screen time; the scene's real narration audio duration is now known, "
            f"split it accordingly. {_format_constraints(project)}"
        ).strip(),
    )
