"""Unit tests for the images stage (SPEC §6, §14): prompt builder, seed determinism, and
stage skip/force behaviour against a fake `ImageGenerator` (no GPU/network).
"""

from __future__ import annotations

from pathlib import Path

from anime_short.config import AppConfig, ImageConfig, LLMConfig, StyleBible, TTSConfig, VideoConfig
from anime_short.image import stage as stage_mod
from anime_short.image.prompt_builder import build
from anime_short.image.stage import deterministic_seed, run_images
from anime_short.schemas import (
    Character,
    ContentSpec,
    ImageResult,
    ProjectSpec,
    PromptStyle,
    Scene,
    Shot,
    VisualSpec,
)

STYLE = StyleBible(
    name="test_style",
    positive_tags="masterpiece, best quality, score_7, safe",
    positive_natural="A high quality anime illustration.",
    negative_tags="worst quality, low quality, blurry",
    negative_natural="low quality, blurry",
)

MINH = Character(
    name="Minh",
    age=22,
    tags="1boy, solo, short black hair, brown eyes",
    description="a slim 22-year-old man with short black hair and brown eyes",
    outfit_tags="dark navy hoodie",
    outfit_description="wearing a dark navy hoodie",
)


def make_shot(**overrides) -> Shot:
    defaults: dict = dict(
        id="scene_01_a",
        visual=VisualSpec(
            subject="a young man",
            action="looking at his phone",
            setting="a rainy street at night",
            lighting="neon lights, phone screen glow",
            shot_type="close-up",
            mood="tense",
        ),
        characters=["minh"],
    )
    defaults.update(overrides)
    return Shot(**defaults)


def make_config() -> AppConfig:
    return AppConfig(
        llm=LLMConfig(base_url="http://unused", model="fake-model"),
        tts=TTSConfig(provider="vieneu"),
        image=ImageConfig(provider="fake"),
        video=VideoConfig(provider="animegen"),
    )


def make_project() -> ProjectSpec:
    return ProjectSpec(
        id="proj_1",
        title="Test",
        style="cinematic_anime",
        content=ContentSpec(topic="test"),
        characters={"minh": MINH},
        scenes=[Scene(id="scene_01", narration="...", shots=[make_shot(id="scene_01_a")])],
    )


# --- prompt_builder: tags style ----------------------------------------------------------------


def test_tags_includes_style_and_character_tags():
    prompt, negative = build(make_shot(), {"minh": MINH}, STYLE, PromptStyle.tags)
    assert "masterpiece" in prompt
    assert "1boy" in prompt
    assert "dark navy hoodie" in prompt
    assert "close-up" in prompt
    assert "9:16 aspect ratio" in prompt
    assert negative == STYLE.negative_tags


def test_tags_are_deduped():
    prompt, _ = build(make_shot(), {"minh": MINH}, STYLE, PromptStyle.tags)
    tags = [t.strip() for t in prompt.split(",")]
    assert len(tags) == len(set(tags))


def test_tags_no_humans_for_scenery_shot():
    shot = make_shot(
        characters=[], visual=VisualSpec(subject="an empty train platform", setting="dusk")
    )
    prompt, _ = build(shot, {}, STYLE, PromptStyle.tags)
    assert "no humans" in prompt


def test_tags_no_humans_omitted_when_visual_implies_person():
    shot = make_shot(characters=[], visual=VisualSpec(subject="a crowd of people waiting"))
    prompt, _ = build(shot, {}, STYLE, PromptStyle.tags)
    assert "no humans" not in prompt


# --- prompt_builder: natural style --------------------------------------------------------------


def test_natural_includes_character_description():
    prompt, negative = build(make_shot(), {"minh": MINH}, STYLE, PromptStyle.natural)
    assert "slim 22-year-old man" in prompt
    assert "dark navy hoodie" in prompt
    assert negative == STYLE.negative_natural


def test_natural_no_characters_states_no_people():
    shot = make_shot(characters=[], visual=VisualSpec(subject="an empty train platform"))
    prompt, _ = build(shot, {}, STYLE, PromptStyle.natural)
    assert "no people" in prompt.lower()


# --- deterministic_seed ---------------------------------------------------------------------


def test_deterministic_seed_stable_and_distinct():
    a = deterministic_seed("proj_1", "scene_01_a")
    b = deterministic_seed("proj_1", "scene_01_a")
    c = deterministic_seed("proj_1", "scene_01_b")
    assert a == b
    assert a != c
    assert isinstance(a, int)


# --- run_images stage ------------------------------------------------------------------------


class FakeGenerator:
    name = "fake"
    prompt_style = PromptStyle.tags

    def __init__(self):
        self.loaded = False
        self.calls = 0

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def generate(self, prompt, negative_prompt, width, height, seed, output_path):
        self.calls += 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-png")
        return ImageResult(path=output_path, seed=seed, width=width, height=height)


def test_run_images_generates_skips_and_forces(tmp_path: Path, monkeypatch):
    fake = FakeGenerator()
    monkeypatch.setattr(stage_mod, "get_image_generator", lambda config: fake)

    config = make_config()
    project = make_project()

    run_images(project, tmp_path, config, force=False)
    assert fake.calls == 1
    shot = project.scenes[0].shots[0]
    assert shot.image is not None
    assert shot.image.provider == "fake"
    assert shot.image.path == "images/scene_01_a.png"
    assert (tmp_path / "images" / "scene_01_a.png").exists()
    assert (tmp_path / "project.json").exists()

    run_images(project, tmp_path, config, force=False)
    assert fake.calls == 1, "existing image should be skipped without --force"

    run_images(project, tmp_path, config, force=True)
    assert fake.calls == 2, "force should regenerate"


def test_run_images_regenerates_when_png_exists_but_shot_image_unset(tmp_path: Path, monkeypatch):
    """A leftover PNG on disk (crash before save_project, or a `--force` run of a later
    stage) must not cause a permanent skip: `shot.image` is the source of truth, not the
    file's mere existence.
    """
    fake = FakeGenerator()
    monkeypatch.setattr(stage_mod, "get_image_generator", lambda config: fake)

    config = make_config()
    project = make_project()

    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "scene_01_a.png").write_bytes(b"leftover-png")

    run_images(project, tmp_path, config, force=False)

    assert fake.calls == 1, "should regenerate when shot.image is unset, despite the stale png"
    shot = project.scenes[0].shots[0]
    assert shot.image is not None
    assert shot.image.provider == "fake"


def test_run_images_uses_deterministic_seed_when_shot_seed_unset(tmp_path: Path, monkeypatch):
    fake = FakeGenerator()
    monkeypatch.setattr(stage_mod, "get_image_generator", lambda config: fake)

    config = make_config()
    project = make_project()

    run_images(project, tmp_path, config, force=False)

    shot = project.scenes[0].shots[0]
    assert shot.image.seed == deterministic_seed("proj_1", "scene_01_a")
