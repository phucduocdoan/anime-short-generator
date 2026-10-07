from __future__ import annotations

import random
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel, Field

from anime_short.config import REPO_ROOT, AppConfig
from anime_short.pipeline import runner
from anime_short.script.llm import LLMClient

DEFAULT_THEME = (
    "quiet anime Shorts about winter love, letting go, self-worth, and gentle life "
    "philosophy for tired hearts"
)
DEFAULT_SLOTS = ("08:15", "12:15", "16:45", "20:15", "22:45")


class QueueIdea(BaseModel):
    title: str = Field(min_length=1, max_length=80)
    topic: str = Field(min_length=20, max_length=500)
    tone: list[str] = Field(default_factory=list, max_length=5)


class QueuePlan(BaseModel):
    ideas: list[QueueIdea]


def queue_dir(config: AppConfig) -> Path:
    base = config.workspace_dir
    if not base.is_absolute():
        base = REPO_ROOT / base
    return base / "queue"


def queue_path(config: AppConfig, day: str) -> Path:
    return queue_dir(config) / f"daily_{day}.yaml"


def plan_day(
    config: AppConfig,
    *,
    day: str,
    count: int,
    theme: str = DEFAULT_THEME,
    timezone_name: str | None = None,
    duration_seconds: float = 45,
) -> dict[str, Any]:
    timezone_name = timezone_name or config.youtube.publish_timezone
    ideas = generate_ideas(config, theme=theme, count=count)
    schedules = schedule_times(
        day=day,
        count=count,
        timezone_name=timezone_name,
        slots=DEFAULT_SLOTS,
    )
    items = []
    for index, idea in enumerate(ideas[:count], start=1):
        title = idea.title.strip()
        project_id = _unique_project_id(day, index, title)
        spec = build_spec(
            project_id=project_id,
            title=title,
            topic=idea.topic,
            tone=idea.tone,
            duration_seconds=duration_seconds,
        )
        items.append(
            {
                "id": f"{day}_{index:02d}",
                "status": "pending",
                "project_id": project_id,
                "title": title,
                "topic": idea.topic,
                "tone": idea.tone,
                "scheduled_at": schedules[index - 1].isoformat(),
                "spec": spec,
                "project_dir": None,
                "url": None,
                "error": None,
            }
        )
    return {
        "date": day,
        "timezone": timezone_name,
        "theme": theme,
        "created_at": datetime.now(ZoneInfo(timezone_name)).isoformat(),
        "items": items,
    }


def save_queue(config: AppConfig, queue: dict[str, Any]) -> Path:
    path = queue_path(config, queue["date"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(queue, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def load_queue(config: AppConfig, day: str) -> dict[str, Any]:
    path = queue_path(config, day)
    if not path.exists():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_spec(config: AppConfig, item: dict[str, Any]) -> Path:
    path = queue_dir(config) / "specs" / f"{item['project_id']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(item["spec"], allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def generate_ideas(config: AppConfig, *, theme: str, count: int) -> list[QueueIdea]:
    try:
        llm_config = config.llm.model_copy(
            update={
                "timeout_seconds": min(config.llm.timeout_seconds, 45),
                "max_retries": min(config.llm.max_retries, 1),
            }
        )
        plan = LLMClient(llm_config).complete_structured(
            system=(
                "You plan daily YouTube Shorts for an anime quotes channel. Ideas must be "
                "emotionally distinct, non-repetitive, and suitable for 35-55 second narrated "
                "quote videos. Avoid generic slogans and avoid reusing the same phrasing."
            ),
            user=(
                f"Create {count} video ideas for today.\n"
                f"Theme: {theme}\n"
                "Audience: English-speaking viewers age 18-35 who like quiet love, life, "
                "healing, and philosophy Shorts.\n"
                "Return concise titles and concrete topics."
            ),
            schema=QueuePlan,
            max_tokens=1200,
        )
        if len(plan.ideas) < count:
            raise RuntimeError(f"LLM returned only {len(plan.ideas)} ideas, need {count}")
        return plan.ideas[:count]
    except Exception as exc:
        raise RuntimeError("daily topic planning requires LLM creativity and failed") from exc


def schedule_times(
    *,
    day: str,
    count: int,
    timezone_name: str,
    slots: tuple[str, ...] = DEFAULT_SLOTS,
    now: datetime | None = None,
) -> list[datetime]:
    tz = ZoneInfo(timezone_name)
    now = now.astimezone(tz) if now else datetime.now(tz)
    target_date = datetime.strptime(day, "%Y-%m-%d").date()
    planned = [
        datetime.combine(target_date, _parse_time(slot), tzinfo=tz)
        + timedelta(minutes=random.randint(-18, 18))
        for slot in slots
    ]
    minimum = now + timedelta(minutes=20)
    future = sorted(t for t in planned if t > minimum)
    if len(future) >= count:
        return future[:count]
    if target_date == now.date():
        start = max(minimum, datetime.combine(target_date, time(13, 0), tzinfo=tz))
        end = datetime.combine(target_date, time(23, 30), tzinfo=tz)
        if start < end:
            future = _spread_times(start, end, count)
    if len(future) < count:
        start = datetime.combine(target_date, time(8, 15), tzinfo=tz)
        end = datetime.combine(target_date, time(22, 45), tzinfo=tz)
        future = _spread_times(start, end, count)
    return future[:count]


def build_spec(
    *,
    project_id: str,
    title: str,
    topic: str,
    tone: list[str],
    duration_seconds: float,
) -> dict[str, Any]:
    return {
        "project": {"id": project_id, "title": title},
        "content": {
            "format": "quotes",
            "topic": topic,
            "duration_seconds": duration_seconds,
            "tone": tone or ["tender", "reflective", "hopeful"],
            "audience": "18-35, US",
            "language": "en",
        },
        "llm": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "model": "gemini-3.8-flash",
            "api_key_env": "GEMINI_API_KEY",
            "temperature": 1.0,
            "reasoning_effort": "low",
        },
        "script": {"words_per_second": 2.8, "max_scene_seconds": 6},
        "timeline": {
            "scene_gap_seconds": 1.0,
            "tail_seconds": 1.5,
            "max_total_seconds": 58.5,
        },
        "voice": {
            "provider": "gemini",
            "voice": "Enceladus",
            "style": (
                "slow and soft, low, warm and intimate, like an older friend talking quietly "
                "at 2 a.m. Gentle, a little melancholic, never forceful. Short pause between lines."
            ),
        },
        "visual": {"provider": "gemini", "style": "lofi_quotes"},
        "video": {"provider": "animegen"},
        "subtitle": {
            "enabled": True,
            "style": "quote",
            "font_name": "Mali",
            "max_words": 16,
            "max_chars": 90,
        },
        "music": {"enabled": True, "provider": "library", "volume_db": -8, "duck": False},
        "render": {"film_grain": 14, "transition_seconds": 0.6, "vignette": True, "dust": 8},
    }


def _unique_project_id(day: str, index: int, title: str) -> str:
    return f"{day.replace('-', '_')}_{index:02d}_{runner.slugify(title, max_len=28)}"


def _parse_time(value: str) -> time:
    hour, minute = value.split(":", maxsplit=1)
    return time(int(hour), int(minute))


def _spread_times(start: datetime, end: datetime, count: int) -> list[datetime]:
    if count <= 1:
        return [start]
    span = (end - start) / (count - 1)
    return [
        start + span * index + timedelta(minutes=random.randint(-12, 12))
        for index in range(count)
    ]
