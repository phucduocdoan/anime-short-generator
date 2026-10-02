"""Build image prompts from a shot, character bible and style (SPEC §6).

The LLM never writes character appearance; it only references character ids in
`shot.characters`. This module injects the appearance (from the character bible) and the
`VisualSpec` fields into either a Danbooru-style tag string or a natural-language description,
depending on the provider's `prompt_style`.
"""

from __future__ import annotations

import re

from anime_short.config import StyleBible
from anime_short.schemas import Character, PromptStyle, Shot

_FRAMING_TAGS = ["9:16 aspect ratio"]
_FRAMING_NATURAL = "9:16 aspect ratio."

# Image models fill signs with invented (often Chinese or gibberish) lettering unless told not
# to; asking for pictograms works better than only forbidding letters.
_SIGN_HINTS = {
    "neon",
    "sign",
    "signs",
    "signage",
    "billboard",
    "billboards",
    "poster",
    "posters",
    "storefront",
    "storefronts",
}
_NO_SIGN_TEXT = (
    "Every neon sign and shop sign is a simple glowing pictogram or abstract shape (a coffee cup, "
    "a star, stripes); signs never contain letters, numbers or characters of any language. Apart "
    "from any writing explicitly requested above, there is no text anywhere in the image."
)

_VISUAL_FIELDS = ("setting", "lighting", "shot_type", "subject", "action", "mood", "extra")

# Heuristic only: if a shot has no characters and none of these appear in the visual fields,
# we assume it is a scenery/object shot and make that explicit in the prompt.
_HUMAN_HINTS = {
    "person",
    "people",
    "man",
    "men",
    "woman",
    "women",
    "boy",
    "girl",
    "child",
    "children",
    "human",
    "figure",
    "silhouette",
    "crowd",
    "couple",
    "lover",
    "wanderer",
    "traveler",
    "hand",
    "hands",
    "finger",
    "fingers",
    "fingertip",
    "fingertips",
    "arm",
    "arms",
    "palm",
    "face",
    "stranger",
    "passenger",
    "student",
    "teacher",
    "soldier",
    "officer",
    "commuter",
    "he",
    "she",
    "his",
    "her",
}


def build(
    shot: Shot,
    characters: dict[str, Character],
    style: StyleBible,
    prompt_style: PromptStyle,
) -> tuple[str, str]:
    shot_characters = [characters[cid] for cid in shot.characters if cid in characters]
    if prompt_style == PromptStyle.tags:
        return _build_tags(shot, shot_characters, style)
    return _build_natural(shot, shot_characters, style)


def _visual_text(shot: Shot) -> str:
    return " ".join(
        getattr(shot.visual, field) for field in _VISUAL_FIELDS if getattr(shot.visual, field)
    )


def _implies_humans(shot: Shot) -> bool:
    words = set(re.findall(r"[a-z]+", _visual_text(shot).lower()))
    words |= {w[:-1] for w in words if w.endswith("s")}  # plurals: figures, couples, lovers
    return bool(words & _HUMAN_HINTS)


def _mentions_signs(shot: Shot) -> bool:
    words = set(re.findall(r"[a-z]+", _visual_text(shot).lower()))
    return bool(words & _SIGN_HINTS)


def _split_to_tags(text: str) -> list[str]:
    return [part.strip().lower() for part in text.split(",") if part.strip()]


def _dedupe(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for tag in tags:
        key = tag.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(tag)
    return out


def _build_tags(shot: Shot, shot_characters: list[Character], style: StyleBible) -> tuple[str, str]:
    tags = _split_to_tags(style.positive_tags)

    for character in shot_characters:
        tags += _split_to_tags(character.tags)
        if character.outfit_tags:
            tags += _split_to_tags(character.outfit_tags)

    if not shot_characters and not _implies_humans(shot):
        tags.append("no humans")

    for field in _VISUAL_FIELDS:
        value = getattr(shot.visual, field)
        if value:
            tags += _split_to_tags(value)

    if shot.visual.text:
        tags += ["english text", f'text "{shot.visual.text}"']
    tags += _FRAMING_TAGS

    prompt = ", ".join(_dedupe(tags))
    negative = style.negative_tags.strip()
    return prompt, negative


def _build_natural(
    shot: Shot, shot_characters: list[Character], style: StyleBible
) -> tuple[str, str]:
    sentences = [style.positive_natural.strip()]

    for character in shot_characters:
        desc = f"{character.name} is {character.description}"
        if character.outfit_description:
            desc += f", {character.outfit_description}"
        sentences.append(desc + ".")

    if not shot_characters and not _implies_humans(shot):
        sentences.append("There are no people in the scene.")

    v = shot.visual
    if v.shot_type:
        shot_type = v.shot_type.strip().rstrip(".")
        if not shot_type.lower().endswith("shot"):
            shot_type += " shot"
        sentences.append(f"{shot_type.capitalize()}.")
    subject_line = v.subject
    if v.action:
        subject_line = f"{subject_line}, {v.action}" if subject_line else v.action
    if subject_line:
        sentences.append(subject_line.strip().capitalize() + ".")
    if v.setting:
        sentences.append(f"Setting: {v.setting}.")
    if v.lighting:
        sentences.append(f"Lighting: {v.lighting}.")
    if v.mood:
        sentences.append(f"Mood: {v.mood}.")
    if v.extra:
        sentences.append(v.extra.strip().capitalize() + ".")
    if v.text:
        sentences.append(f'The only writing in the image is the English text "{v.text}".')

    sentences.append(_FRAMING_NATURAL)
    # Last, where the models follow it best (placed mid-prompt, Gemini still lettered the signs).
    if _mentions_signs(shot):
        sentences.append(_NO_SIGN_TEXT)

    prompt = " ".join(s for s in sentences if s)
    negative = style.negative_natural.strip()
    return prompt, negative
