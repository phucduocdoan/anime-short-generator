"""Prompt text for the script/plan LLM calls (SPEC §5). Templates live in `templates/*.txt`
(repo root, `config.TEMPLATES_DIR`) so wording can be iterated on without touching code.
"""

from __future__ import annotations

from anime_short.config import TEMPLATES_DIR, AppConfig
from anime_short.schemas import Character, ContentSpec, Scene

LANGUAGE_NAMES = {"vi": "Vietnamese", "en": "English"}


def _load(name: str) -> str:
    return (TEMPLATES_DIR / name).read_text(encoding="utf-8")


def _language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def _character_list(characters: dict[str, Character]) -> str:
    if not characters:
        return "(none)"
    lines = []
    for cid, c in characters.items():
        age = f", age {c.age}" if c.age is not None else ""
        lines.append(f"- {cid}: {c.name}{age} — {c.description}")
    return "\n".join(lines)


def script_system(
    content: ContentSpec, language: str, config: AppConfig, need_characters: bool
) -> str:
    words_per_second = config.script.words_per_second
    target_words = round(content.duration_seconds * words_per_second)
    hook_words = max(3, round(config.script.hook_seconds * words_per_second))
    min_scenes = max(3, round(content.duration_seconds / 12))
    max_scenes = max(min_scenes + 1, round(content.duration_seconds / 7))
    scene_hint = f"{min_scenes}-{max_scenes} scenes"
    words_per_scene = round(target_words / ((min_scenes + max_scenes) / 2))
    character_instructions = (
        _load("script_characters.txt")
        if need_characters
        else "- Do not invent new characters; the `characters` list must be empty.\n"
    )
    return _load("script_system.txt").format(
        duration_seconds=content.duration_seconds,
        language_name=_language_name(language),
        hook_words=hook_words,
        target_words=target_words,
        words_per_second=words_per_second,
        words_per_scene=words_per_scene,
        scene_hint=scene_hint,
        character_instructions=character_instructions,
        tone=", ".join(content.tone) or "engaging",
        audience=content.audience or "general",
    )


def script_user(content: ContentSpec) -> str:
    return _load("script_user.txt").format(topic=content.topic.strip())


def plan_system(
    characters: dict[str, Character],
    shot_count: int,
    scene_duration: float,
    extra_constraints: str = "",
) -> str:
    return _load("plan_system.txt").format(
        shot_count=shot_count,
        scene_duration=scene_duration,
        extra_constraints=extra_constraints,
        character_list=_character_list(characters),
    )


def plan_user(scene: Scene, language: str) -> str:
    return _load("plan_user.txt").format(
        language_name=_language_name(language),
        narration=scene.narration,
        visual_description=scene.visual_description,
        emotion=scene.emotion or "neutral",
    )
