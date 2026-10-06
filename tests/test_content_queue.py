from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from anime_short.config import load_config
from anime_short.content_queue import (
    QueueIdea,
    QueuePlan,
    load_queue,
    plan_day,
    save_queue,
    schedule_times,
    write_spec,
)


def test_plan_day_uses_llm_ideas_and_writes_specs(tmp_path, monkeypatch):
    config = load_config({"workspace_dir": str(tmp_path)})

    class FakeLLM:
        def __init__(self, _config):
            pass

        def complete_structured(self, **_kwargs):
            return QueuePlan(
                ideas=[
                    QueueIdea(
                        title="Winter Love One",
                        topic="A quiet winter love topic with enough detail for a Short.",
                        tone=["tender"],
                    ),
                    QueueIdea(
                        title="Letting Go Two",
                        topic="A reflective letting go topic with enough detail for a Short.",
                        tone=["healing"],
                    ),
                ]
            )

    monkeypatch.setattr("anime_short.content_queue.LLMClient", FakeLLM)

    queue = plan_day(
        config,
        day="2026-10-06",
        count=2,
        theme="winter love",
        timezone_name="Asia/Ho_Chi_Minh",
    )
    path = save_queue(config, queue)
    loaded = load_queue(config, "2026-10-06")
    spec_path = write_spec(config, loaded["items"][0])

    assert path.exists()
    assert spec_path.exists()
    assert len(loaded["items"]) == 2
    assert loaded["items"][0]["status"] == "pending"
    assert loaded["items"][0]["spec"]["content"]["format"] == "quotes"
    assert loaded["items"][0]["project_id"].startswith("2026_10_06_01_")


def test_plan_day_fails_when_llm_returns_too_few_ideas(tmp_path, monkeypatch):
    import pytest

    config = load_config({"workspace_dir": str(tmp_path)})

    class FakeLLM:
        def __init__(self, _config):
            pass

        def complete_structured(self, **_kwargs):
            return QueuePlan(
                ideas=[
                    QueueIdea(
                        title="Only One",
                        topic="A single topic is not enough for the requested daily plan.",
                    )
                ]
            )

    monkeypatch.setattr("anime_short.content_queue.LLMClient", FakeLLM)

    with pytest.raises(RuntimeError, match="requires LLM creativity"):
        plan_day(config, day="2026-10-06", count=2)


def test_schedule_times_keeps_today_times_in_future():
    tz = ZoneInfo("Asia/Ho_Chi_Minh")
    now = datetime(2026, 10, 6, 20, 0, tzinfo=tz)

    times = schedule_times(
        day="2026-10-06",
        count=2,
        timezone_name="Asia/Ho_Chi_Minh",
        now=now,
    )

    assert len(times) == 2
    assert all(item > now for item in times)
