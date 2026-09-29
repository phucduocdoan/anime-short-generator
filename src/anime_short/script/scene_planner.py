"""Plan stage: split one scene into shots with structured visual fields (SPEC §5, §9)."""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, create_model

from anime_short.config import AppConfig
from anime_short.schemas import CameraMove, Character, Scene, Shot, VisualSpec
from anime_short.script import prompts
from anime_short.script.llm import LLMClient

SHOT_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_TYPOGRAPHIC = str.maketrans("\u2018\u2019\u201c\u201d\u2014", "''\"\"-")


def _english_text(text: str) -> str:
    """The planner sometimes copies the (Vietnamese) narration into `text`; image models render
    anything but plain English as gibberish, so such text is dropped."""
    text = text.translate(_TYPOGRAPHIC).strip()
    return text if text.isascii() else ""


class ShotDraft(BaseModel):
    subject: str
    action: str = ""
    setting: str = ""
    lighting: str = ""
    shot_type: str = ""
    mood: str = ""
    extra: str = ""
    text: str = Field(max_length=40)
    motion_prompt: str = ""
    camera: CameraMove
    characters: list[str] = Field(default_factory=list)
    weight: float = 1.0


def _plan_schema(shot_count: int) -> type[BaseModel]:
    """A `{shots: [...]}` model with the array length pinned to `shot_count` via JSON Schema
    `minItems`/`maxItems`, which the local vLLM server enforces at decode time (checked directly).
    """
    return create_model(
        "PlanDraft",
        shots=(list[ShotDraft], Field(min_length=shot_count, max_length=shot_count)),
    )


def estimate_scene_duration(narration: str, words_per_second: float) -> float:
    words = len(narration.split())
    return words / words_per_second if words_per_second else 0.0


def shot_count_for_duration(
    duration: float, min_shot_seconds: float, max_shot_seconds: float
) -> int:
    """Smallest shot count so no shot exceeds `max_shot_seconds`, without going under
    `min_shot_seconds` on average (SPEC §9).
    """
    if duration <= 0:
        return 1
    count = max(1, math.ceil(duration / max_shot_seconds))
    while count > 1 and duration / count < min_shot_seconds:
        count -= 1
    return count


def build_shots(
    scene_id: str, draft_shots: list[ShotDraft], previous_camera: CameraMove | None = None
) -> list[Shot]:
    """`previous_camera` is the camera of the shot right before this scene; a `static` shot that
    directly follows another `static` one becomes a slow `push_in`, so the video never holds still
    for two shots in a row (the planner LLM otherwise picks `static` for most shots).
    """
    shots = []
    for i, sd in enumerate(draft_shots):
        camera = sd.camera
        if camera == CameraMove.static and previous_camera == CameraMove.static:
            camera = CameraMove.push_in
        previous_camera = camera
        letter = SHOT_LETTERS[i] if i < len(SHOT_LETTERS) else str(i)
        visual = VisualSpec(
            subject=sd.subject,
            action=sd.action,
            setting=sd.setting,
            lighting=sd.lighting,
            shot_type=sd.shot_type,
            mood=sd.mood,
            extra=sd.extra,
            text=_english_text(sd.text),
        )
        shots.append(
            Shot(
                id=f"{scene_id}_{letter}",
                visual=visual,
                motion_prompt=sd.motion_prompt,
                camera=camera,
                characters=list(sd.characters),
                weight=sd.weight,
            )
        )
    return shots


def plan_scene(
    scene: Scene,
    characters: dict[str, Character],
    config: AppConfig,
    llm: LLMClient,
    *,
    shot_count: int | None = None,
    scene_duration: float | None = None,
    extra_constraints: str = "",
    language: str = "vi",
    previous_camera: CameraMove | None = None,
) -> list[Shot]:
    duration = (
        scene_duration
        if scene_duration is not None
        else estimate_scene_duration(scene.narration, config.script.words_per_second)
    )
    count = (
        shot_count
        if shot_count is not None
        else min(
            config.script.max_shots_per_scene,
            shot_count_for_duration(
                duration, config.timeline.min_shot_seconds, config.timeline.max_shot_seconds
            ),
        )
    )
    if previous_camera is not None:
        # Each scene is planned in its own call, so the LLM can't vary the camera across scenes
        # unless it is told what came right before.
        extra_constraints = (
            f"The shot right before this scene uses camera `{previous_camera.value}`; open this "
            f"scene with a different camera move. {extra_constraints}"
        ).strip()
    system = prompts.plan_system(characters, count, duration, extra_constraints)
    user = prompts.plan_user(scene, language)
    draft = llm.complete_structured(
        system=system,
        user=user,
        schema=_plan_schema(count),
        max_tokens=250 + 350 * count,
    )
    return build_shots(scene.id, draft.shots, previous_camera)
