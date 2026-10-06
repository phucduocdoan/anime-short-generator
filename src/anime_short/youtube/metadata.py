from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

from anime_short.config import AppConfig
from anime_short.schemas import ProjectSpec
from anime_short.script.llm import LLMClient

log = logging.getLogger(__name__)


class VideoMetadata(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=5000)
    tags: list[str] = Field(default_factory=list, max_length=15)


def default_metadata(
    project: ProjectSpec,
    title: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
) -> VideoMetadata:
    """Build conservative YouTube metadata from the finished project."""
    base_title = _clean_title(title or project.title or project.content.topic)
    base_description = description or _description(project)
    base_tags = _dedupe(
        [
            *(tags or []),
            "shorts",
            "anime shorts",
            "life quotes",
            "philosophy",
            "love quotes",
            project.content.format,
            project.language,
            *_keywords(project.content.topic),
        ]
    )
    return VideoMetadata(
        title=base_title[:100],
        description=base_description[:5000],
        tags=[tag[:500] for tag in base_tags[:15]],
    )


def upload_metadata(
    project: ProjectSpec,
    config: AppConfig,
    title: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
) -> VideoMetadata:
    """Build upload metadata from a base template, optionally varied by the project LLM."""
    fallback = default_metadata(project, title=title, description=description, tags=tags)
    if config.youtube.metadata_provider != "llm":
        return fallback
    try:
        generated = llm_metadata(project, config, fallback)
    except Exception as exc:
        log.warning("YouTube metadata LLM failed; using template fallback: %s", exc)
        return fallback

    return _apply_overrides(generated, fallback, title=title, description=description, tags=tags)


def llm_metadata(project: ProjectSpec, config: AppConfig, fallback: VideoMetadata) -> VideoMetadata:
    """Ask the configured structured-output LLM for varied YouTube metadata."""
    llm_config = config.llm.model_copy(
        update={
            "timeout_seconds": min(config.llm.timeout_seconds, 30),
            "max_retries": min(config.llm.max_retries, 1),
        }
    )
    client = LLMClient(llm_config)
    metadata = client.complete_structured(
        system=(
            "You write YouTube metadata for vertical anime Shorts in the philosophy, life quotes, "
            "and love quotes niche. Use the provided base template as the channel style, but do "
            "not copy it word for word. Be tasteful, specific, and search-friendly. Avoid "
            "clickbait, false claims, emoji, ALL CAPS, and overpromising. Return only valid JSON."
        ),
        user=_metadata_prompt(project, fallback),
        schema=VideoMetadata,
        max_tokens=900,
    )
    return _normalise_metadata(metadata, fallback)


def _description(project: ProjectSpec) -> str:
    lines = [
        project.content.topic.strip(),
        "",
        "A quiet vertical anime Short about love, life, and the thoughts we keep to ourselves.",
        "",
        "#Shorts #AnimeShorts #LifeQuotes",
    ]
    if project.ending:
        lines.insert(2, project.ending.strip())
        lines.insert(3, "")
    return "\n".join(line for line in lines if line is not None).strip()


def _metadata_prompt(project: ProjectSpec, fallback: VideoMetadata) -> str:
    narration = "\n".join(f"- {scene.narration}" for scene in project.scenes[:20])
    if not narration:
        narration = "- " + project.content.topic.strip()
    ending = project.ending or ""
    tone = ", ".join(project.content.tone) if project.content.tone else "quiet, reflective"
    return f"""Base template:
Title: {fallback.title}
Description:
{fallback.description}
Tags: {", ".join(fallback.tags)}

Project:
- channel/content type: vertical anime Short
- title: {project.title}
- topic: {project.content.topic}
- format: {project.content.format}
- language: {project.language}
- tone: {tone}
- ending: {ending}

Narration lines:
{narration}

Write:
- `title`: one natural English title, <= 100 characters. No hashtags in title.
- `description`: 2-4 short lines, then a hashtag line. Mention the emotional idea, not a plot recap.
- `tags`: 10-15 lowercase tags. Include shorts/anime/life/love/philosophy tags when relevant.
"""


def _normalise_metadata(metadata: VideoMetadata, fallback: VideoMetadata) -> VideoMetadata:
    title = _clean_title(metadata.title)[:100] or fallback.title
    description = re.sub(r"\n{3,}", "\n\n", metadata.description.strip())[:5000]
    if not description:
        description = fallback.description
    tags = _dedupe([*metadata.tags, *fallback.tags])[:15]
    return VideoMetadata(title=title, description=description, tags=[tag[:500] for tag in tags])


def _apply_overrides(
    generated: VideoMetadata,
    fallback: VideoMetadata,
    title: str | None,
    description: str | None,
    tags: list[str] | None,
) -> VideoMetadata:
    final_title = _clean_title(title or generated.title)[:100]
    final_description = (description or generated.description).strip()[:5000]
    final_tags = _dedupe([*(tags or []), *generated.tags, *fallback.tags])[:15]
    return VideoMetadata(
        title=final_title,
        description=final_description,
        tags=[tag[:500] for tag in final_tags],
    )


def _clean_title(value: str) -> str:
    title = re.sub(r"\s+", " ", value).strip()
    return title or "Anime Short"


def _keywords(topic: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9]+", topic.lower())
    return [word for word in words if len(word) >= 3][:6]


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = value.strip().lower()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out
