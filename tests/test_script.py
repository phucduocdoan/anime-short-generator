"""Unit tests for the script + plan stages (SPEC §5). No network: every LLM call goes through a
fake client, either at the raw chat-completion level (`FakeRawClient`, for `LLMClient` itself) or
at the `complete_structured` level (`FakeLLM`, for generator/scene_planner/stage tests).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from anime_short.config import AppConfig, ImageConfig, LLMConfig, TTSConfig, VideoConfig
from anime_short.schemas import AudioInfo, CameraMove, Character, ContentSpec, ProjectSpec, Scene
from anime_short.script import generator, stage
from anime_short.script.llm import LLMClient, LLMError
from anime_short.script.scene_planner import (
    ShotDraft,
    build_shots,
    estimate_scene_duration,
    plan_scene,
    shot_count_for_duration,
)

# --- fixtures ---------------------------------------------------------------------------------


def make_config() -> AppConfig:
    return AppConfig(
        llm=LLMConfig(base_url="http://unused", model="fake-model", max_retries=2),
        tts=TTSConfig(provider="vieneu"),
        image=ImageConfig(provider="anima"),
        video=VideoConfig(provider="animegen"),
    )


def make_project(**kwargs) -> ProjectSpec:
    # 4s -> a 17-word target, which SCRIPT_PAYLOAD's 19 narration words meet on the first attempt
    content = ContentSpec(topic="a boy and a message", duration_seconds=4)
    defaults = dict(id="p1", title="Title", content=content)
    defaults.update(kwargs)
    return ProjectSpec(**defaults)


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


class FakeRawClient:
    """Fakes the slice of the `openai` client `LLMClient` calls directly."""

    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs) -> FakeResponse:
        # snapshot messages: `LLMClient` mutates the same list across retries
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return FakeResponse(self.contents.pop(0))


def _list_bounds(field) -> tuple[int | None, int | None]:
    min_len = next((m.min_length for m in field.metadata if hasattr(m, "min_length")), None)
    max_len = next((m.max_length for m in field.metadata if hasattr(m, "max_length")), None)
    return min_len, max_len


class FakeLLM:
    """Fakes `LLMClient.complete_structured` directly, for generator/scene_planner/stage tests."""

    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete_structured(self, *, system, user, schema, max_tokens):
        self.calls.append(dict(system=system, user=user, schema=schema, max_tokens=max_tokens))
        return schema.model_validate(self.responses.pop(0))


SCRIPT_PAYLOAD = {
    "characters": [
        {
            "id": "minh",
            "name": "Minh",
            "age": 22,
            "tags": "1boy, short black hair, brown eyes",
            "description": "a young Vietnamese man with short black hair and brown eyes",
            "outfit_tags": "navy hoodie",
            "outfit_description": "wearing a navy hoodie",
        }
    ],
    "scenes": [
        {
            "narration": "2 gio sang, dien thoai cua Minh bong sang len.",
            "visual_description": "close-up of a phone glowing in a dark bedroom",
            "emotion": "mystery",
            "importance": "high",
        },
        {
            "narration": "Mot tin nhan tu chinh anh, 10 nam sau.",
            "visual_description": "Minh's hand hesitating over the notification",
            "emotion": "shock",
            "importance": "normal",
        },
    ],
    "ending": "Ban se lam gi neu nhan duoc tin nhan do?",
}


def shot_payload(n: int) -> dict:
    return {
        "shots": [
            {
                "subject": "a young man",
                "action": "sits up",
                "setting": "dark bedroom",
                "lighting": "cold blue glow",
                "shot_type": "close-up",
                "mood": "tense",
                "extra": "",
                "text": "",
                "motion_prompt": "slow push in, curtains drift",
                "camera": "push_in",
                "characters": ["minh"],
                "weight": 1.0,
            }
            for _ in range(n)
        ]
    }


# --- LLMClient (retry / validation) -----------------------------------------------------------


class Person(BaseModel):
    name: str
    age: int


def test_complete_structured_returns_valid_response():
    raw = FakeRawClient([json.dumps({"name": "Bob", "age": 30})])
    client = LLMClient(LLMConfig(base_url="x", model="m"), client=raw)
    result = client.complete_structured(system="s", user="u", schema=Person, max_tokens=50)
    assert result == Person(name="Bob", age=30)
    assert len(raw.calls) == 1
    # response_format carries a json_schema derived from the Pydantic model
    assert raw.calls[0]["response_format"]["type"] == "json_schema"


def test_complete_structured_retries_on_invalid_json_then_succeeds():
    raw = FakeRawClient(["not json at all", json.dumps({"name": "Alice", "age": 5})])
    client = LLMClient(LLMConfig(base_url="x", model="m", max_retries=2), client=raw)
    result = client.complete_structured(system="s", user="u", schema=Person, max_tokens=50)
    assert result == Person(name="Alice", age=5)
    assert len(raw.calls) == 2
    # the retry feeds the bad output + error back as extra messages
    assert len(raw.calls[1]["messages"]) == len(raw.calls[0]["messages"]) + 2


def test_complete_structured_retries_on_schema_mismatch():
    raw = FakeRawClient([json.dumps({"name": "Alice"}), json.dumps({"name": "Alice", "age": 5})])
    client = LLMClient(LLMConfig(base_url="x", model="m", max_retries=2), client=raw)
    result = client.complete_structured(system="s", user="u", schema=Person, max_tokens=50)
    assert result.age == 5
    assert len(raw.calls) == 2


def test_complete_structured_raises_after_max_retries():
    raw = FakeRawClient(["bad"] * 3)
    client = LLMClient(LLMConfig(base_url="x", model="m", max_retries=2), client=raw)
    with pytest.raises(LLMError):
        client.complete_structured(system="s", user="u", schema=Person, max_tokens=50)
    assert len(raw.calls) == 3  # 1 initial attempt + 2 retries


# --- generator (script stage) ------------------------------------------------------------------


def test_generate_script_returns_validated_draft():
    llm = FakeLLM([SCRIPT_PAYLOAD])
    config = make_config()
    draft = generator.generate_script(
        ContentSpec(topic="x", duration_seconds=4), "vi", config, llm, need_characters=True
    )
    assert len(draft.scenes) == 2
    assert draft.characters[0].id == "minh"
    assert len(llm.calls) == 1


def test_generate_script_re_asks_when_narration_is_too_short():
    short = {**SCRIPT_PAYLOAD, "scenes": SCRIPT_PAYLOAD["scenes"][:1]}  # 10 words < 0.85 * 17
    llm = FakeLLM([short, SCRIPT_PAYLOAD])
    draft = generator.generate_script(
        ContentSpec(topic="x", duration_seconds=4), "vi", make_config(), llm, need_characters=True
    )
    assert len(draft.scenes) == 2
    assert len(llm.calls) == 2
    assert "had 10 words" in llm.calls[1]["user"]


def test_generate_script_re_asks_when_narration_is_too_long():
    llm = FakeLLM([SCRIPT_PAYLOAD] * 3)  # 19 words > 1.2 * 13 (3s target)
    generator.generate_script(
        ContentSpec(topic="x", duration_seconds=3), "vi", make_config(), llm, need_characters=True
    )
    assert "fewer or shorter scenes" in llm.calls[1]["user"]


def test_generate_script_keeps_longest_draft_when_always_short():
    one = {**SCRIPT_PAYLOAD, "scenes": SCRIPT_PAYLOAD["scenes"][:1]}
    llm = FakeLLM([one, one, one])
    draft = generator.generate_script(
        ContentSpec(topic="x", duration_seconds=10), "vi", make_config(), llm, need_characters=True
    )
    assert len(llm.calls) == generator.LENGTH_ATTEMPTS
    assert generator.narration_words(draft) == 10


def test_scenes_from_draft_ids_are_sequential():
    draft = generator.ScriptDraft.model_validate(SCRIPT_PAYLOAD)
    scenes = generator.scenes_from_draft(draft)
    assert [s.id for s in scenes] == ["scene_01", "scene_02"]
    assert scenes[0].narration == SCRIPT_PAYLOAD["scenes"][0]["narration"]


def test_characters_from_draft_dedupes_ids():
    draft = generator.ScriptDraft.model_validate(
        {
            **SCRIPT_PAYLOAD,
            "characters": [
                {**SCRIPT_PAYLOAD["characters"][0], "id": "minh"},
                {**SCRIPT_PAYLOAD["characters"][0], "id": "minh", "name": "Minh Two"},
            ],
        }
    )
    characters = generator.characters_from_draft(draft)
    assert set(characters) == {"minh", "minh_2"}
    assert characters["minh_2"].name == "Minh Two"


# --- scene_planner (plan stage) -----------------------------------------------------------------


def test_shot_count_for_duration_respects_bounds():
    assert shot_count_for_duration(0, 1.5, 5.0) == 1
    assert shot_count_for_duration(4.0, 1.5, 5.0) == 1
    assert shot_count_for_duration(12.0, 1.5, 5.0) == 3  # ceil(12/5)
    # max_shot_seconds is generous here, so 1 shot already satisfies both bounds
    assert shot_count_for_duration(12.0, 5.0, 100.0) == 1
    # max_shot_seconds forces 3 shots, but min_shot_seconds pulls it back down to 2
    assert shot_count_for_duration(12.0, 5.0, 4.5) == 2


def test_estimate_scene_duration():
    assert estimate_scene_duration("mot hai ba bon", 2.0) == 2.0


def test_build_shots_ids_use_letters():
    shots = build_shots("scene_01", [ShotDraft(subject="a", camera="pan_left", text="")] * 2)
    assert [s.id for s in shots] == ["scene_01_a", "scene_01_b"]


def test_build_shots_never_keeps_two_static_shots_in_a_row():
    drafts = [ShotDraft(subject=c, camera="static", text="") for c in "abc"]

    assert [s.camera for s in build_shots("scene_01", drafts)] == ["static", "push_in", "static"]
    assert [s.camera for s in build_shots("scene_02", drafts[:1], CameraMove.static)] == ["push_in"]


def test_build_shots_keeps_only_plain_english_text():
    drafts = [
        ShotDraft(subject="a", camera="static", text="I\u2019m here"),
        ShotDraft(subject="b", camera="push_in", text="Tin nh\u1eafn t\u1eeb anh"),
    ]
    assert [s.visual.text for s in build_shots("scene_01", drafts)] == ["I'm here", ""]


def test_run_plan_passes_previous_scene_camera(monkeypatch, tmp_path):
    static_payload = shot_payload(1)
    static_payload["shots"][0]["camera"] = "static"
    llm = FakeLLM([static_payload, static_payload])
    monkeypatch.setattr(stage, "LLMClient", lambda cfg: llm)
    project = make_project(
        scenes=[Scene(id="scene_01", narration="a"), Scene(id="scene_02", narration="b c")]
    )

    stage.run_plan(project, tmp_path, make_config())

    assert [s.shots[0].camera for s in project.scenes] == ["static", "push_in"]


def test_plan_scene_requests_exact_shot_count_and_builds_shots():
    config = make_config()
    scene = Scene(id="scene_01", narration=" ".join(["tu"] * 27))  # 10s @ 2.7 wps
    characters = {"minh": Character(name="Minh", tags="1boy", description="a boy")}
    expected_count = shot_count_for_duration(
        estimate_scene_duration(scene.narration, config.script.words_per_second),
        config.timeline.min_shot_seconds,
        config.timeline.max_shot_seconds,
    )
    expected_count = min(expected_count, config.script.max_shots_per_scene)
    llm = FakeLLM([shot_payload(expected_count)])

    shots = plan_scene(scene, characters, config, llm)

    assert len(shots) == expected_count
    assert shots[0].id == "scene_01_a"
    assert shots[0].characters == ["minh"]
    schema = llm.calls[0]["schema"]
    field = schema.model_fields["shots"]
    assert _list_bounds(field) == (expected_count, expected_count)


def test_plan_scene_uses_explicit_shot_count_for_replan():
    config = make_config()
    scene = Scene(id="scene_02", narration="mot cau ngan")
    llm = FakeLLM([shot_payload(4)])

    shots = plan_scene(scene, {}, config, llm, shot_count=4, scene_duration=13.0)

    assert len(shots) == 4
    assert [s.id for s in shots] == [f"scene_02_{c}" for c in "abcd"]


# --- stage.py (orchestration: skip/force/resume) ------------------------------------------------


def test_run_script_populates_project_and_is_skipped_without_force(monkeypatch, tmp_path):
    llm = FakeLLM([SCRIPT_PAYLOAD])
    monkeypatch.setattr(stage, "LLMClient", lambda cfg: llm)
    project = make_project()
    config = make_config()

    stage.run_script(project, tmp_path, config)

    assert len(project.scenes) == 2
    assert project.hook == project.scenes[0].narration
    assert project.ending == SCRIPT_PAYLOAD["ending"]
    assert "minh" in project.characters

    # second call without force: no new LLM call, scenes untouched
    stage.run_script(project, tmp_path, config)
    assert len(llm.calls) == 1


def test_run_script_does_not_overwrite_existing_characters(monkeypatch, tmp_path):
    llm = FakeLLM([SCRIPT_PAYLOAD])
    monkeypatch.setattr(stage, "LLMClient", lambda cfg: llm)
    project = make_project(
        characters={"an": Character(name="An", tags="1girl", description="a girl")}
    )
    stage.run_script(project, tmp_path, make_config())
    assert set(project.characters) == {"an"}  # LLM's "minh" is ignored: bible already exists


def test_run_script_force_regenerates(monkeypatch, tmp_path):
    llm = FakeLLM([SCRIPT_PAYLOAD, SCRIPT_PAYLOAD])
    monkeypatch.setattr(stage, "LLMClient", lambda cfg: llm)
    project = make_project()
    config = make_config()
    stage.run_script(project, tmp_path, config)
    stage.run_script(project, tmp_path, config, force=True)
    assert len(llm.calls) == 2


def test_run_plan_skips_scenes_with_existing_shots(monkeypatch, tmp_path):
    llm = FakeLLM([shot_payload(1)])
    monkeypatch.setattr(stage, "LLMClient", lambda cfg: llm)
    already_planned = build_shots("scene_01", [ShotDraft(subject="x", camera="static", text="")])
    project = make_project(
        scenes=[
            Scene(id="scene_01", narration="a", shots=already_planned),
            Scene(id="scene_02", narration="b c"),
        ]
    )
    stage.run_plan(project, tmp_path, make_config())
    assert len(llm.calls) == 1  # only scene_02 was planned
    assert project.scenes[0].shots == already_planned
    assert len(project.scenes[1].shots) == 1


def test_run_plan_force_replans_all(monkeypatch, tmp_path):
    llm = FakeLLM([shot_payload(1), shot_payload(1)])
    monkeypatch.setattr(stage, "LLMClient", lambda cfg: llm)
    project = make_project(
        scenes=[Scene(id="scene_01", narration="a"), Scene(id="scene_02", narration="b c")]
    )
    stage.run_plan(project, tmp_path, make_config(), force=True)
    assert len(llm.calls) == 2


def test_replan_scene_picks_shot_count_from_actual_duration(monkeypatch):
    config = make_config()  # max_shot_seconds=5.0 by default
    llm = FakeLLM([shot_payload(3)])  # ceil(12.0 / 5.0) == 3
    monkeypatch.setattr(stage, "LLMClient", lambda cfg: llm)
    scene = Scene(
        id="scene_01",
        narration="long scene",
        audio=AudioInfo(path="audio/scene_01.wav", duration=12.0),
    )
    project = make_project(scenes=[scene])

    stage.replan_scene(project, scene, config)

    assert len(scene.shots) == 3
    field = llm.calls[0]["schema"].model_fields["shots"]
    assert _list_bounds(field) == (3, 3)


def test_replan_scene_requires_audio_duration():
    config = make_config()
    scene = Scene(id="scene_01", narration="x")
    project = make_project(scenes=[scene])
    with pytest.raises(ValueError, match="audio duration"):
        stage.replan_scene(project, scene, config)
