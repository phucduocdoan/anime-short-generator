"""Script stage: topic -> hook, scenes, optional character bible (SPEC §5)."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, Field, create_model

from anime_short.config import AppConfig
from anime_short.schemas import Character, ContentSpec, Scene
from anime_short.script import prompts
from anime_short.script.llm import LLMClient


class CharacterDraft(BaseModel):
    id: str = Field(description="short lowercase ascii id, e.g. 'minh'")
    name: str
    age: int | None = None
    tags: str
    description: str
    outfit_tags: str = ""
    outfit_description: str = ""


class SceneDraft(BaseModel):
    narration: str
    visual_description: str
    emotion: str = ""
    importance: Literal["low", "normal", "high"] = "normal"


class ScriptDraft(BaseModel):
    characters: list[CharacterDraft] = Field(default_factory=list, max_length=2)
    scenes: list[SceneDraft] = Field(min_length=1)
    ending: str = ""


def _script_schema(need_characters: bool) -> type[ScriptDraft]:
    """Pin `characters` to non-empty when the project has no bible yet — the model has been
    unreliable about honoring this as prompt wording alone, so enforce it structurally instead
    (same pattern as the per-scene shot count in `scene_planner._plan_schema`).
    """
    if not need_characters:
        return ScriptDraft
    return create_model(
        "ScriptDraftWithCharacters",
        __base__=ScriptDraft,
        characters=(list[CharacterDraft], Field(min_length=1, max_length=2)),
    )


def generate_script(
    content: ContentSpec, language: str, config: AppConfig, llm: LLMClient, need_characters: bool
) -> ScriptDraft:
    system = prompts.script_system(content, language, config, need_characters)
    user = prompts.script_user(content)
    # ~ a few hundred tokens of overhead plus room per scene (narration + english visual text).
    est_scenes = max(4, round(content.duration_seconds / 8))
    max_tokens = 400 + 220 * est_scenes
    schema = _script_schema(need_characters)
    return llm.complete_structured(system=system, user=user, schema=schema, max_tokens=max_tokens)


def _slugify_id(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:20] or "char"


def _unique_id(name_or_id: str, existing: Iterable[str]) -> str:
    base = _slugify_id(name_or_id)
    existing = set(existing)
    cid = base
    i = 2
    while cid in existing:
        cid = f"{base}_{i}"
        i += 1
    return cid


def characters_from_draft(draft: ScriptDraft) -> dict[str, Character]:
    characters: dict[str, Character] = {}
    for cdraft in draft.characters:
        cid = _unique_id(cdraft.id or cdraft.name, characters.keys())
        characters[cid] = Character(
            name=cdraft.name,
            age=cdraft.age,
            tags=cdraft.tags,
            description=cdraft.description,
            outfit_tags=cdraft.outfit_tags,
            outfit_description=cdraft.outfit_description,
        )
    return characters


def scenes_from_draft(draft: ScriptDraft) -> list[Scene]:
    return [
        Scene(
            id=f"scene_{i:02d}",
            narration=sd.narration.strip(),
            visual_description=sd.visual_description.strip(),
            emotion=sd.emotion,
            importance=sd.importance,
        )
        for i, sd in enumerate(draft.scenes, start=1)
    ]
