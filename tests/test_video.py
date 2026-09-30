"""Unit tests for the videos stage (SPEC §7): shared postprocessing, the still_motion CPU
fallback, and stage skip/force/VRAM-fallback behaviour against fake generators (no GPU/network).
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

from anime_short.config import AppConfig, ImageConfig, LLMConfig, TTSConfig, VideoConfig
from anime_short.schemas import (
    CameraMove,
    ContentSpec,
    ImageInfo,
    ProjectSpec,
    Scene,
    Shot,
    VideoMode,
    VisualSpec,
)
from anime_short.utils.gpu import InsufficientVRAM
from anime_short.utils.media import probe_video
from anime_short.video import stage as stage_mod
from anime_short.video.postprocess import encode_raw_frames, normalize_clip, target_frame_count
from anime_short.video.stage import run_videos
from anime_short.video.still_motion import StillMotionGenerator

# --- postprocess ---------------------------------------------------------------------------


def test_target_frame_count():
    assert target_frame_count(5.0, 24) == 120
    assert target_frame_count(2.5, 24) == 60


def test_encode_raw_frames_produces_expected_video(tmp_path: Path):
    width, height, fps, n_frames = 64, 96, 12, 18
    frames = (np.full((height, width, 3), i * 10 % 256, dtype=np.uint8) for i in range(n_frames))
    out_path = tmp_path / "clip.mp4"

    result = encode_raw_frames(frames, out_path, width, height, fps)

    assert out_path.exists()
    assert result.width == width
    assert result.height == height
    assert result.fps == fps
    probe = probe_video(out_path)
    assert probe.width == width
    assert probe.height == height
    assert probe.fps == fps
    assert not probe.has_audio
    assert abs(probe.duration - n_frames / fps) < 0.05


def test_normalize_clip_stretches_short_source_to_duration(tmp_path: Path):
    # animegen caps at 81 frames @16fps (~5.06s); a 5.5s shot must still get a full-length clip.
    src = tmp_path / "src.mp4"
    encode_raw_frames((np.zeros((96, 64, 3), np.uint8) for _ in range(16)), src, 64, 96, 16)

    result = normalize_clip(src, tmp_path / "out.mp4", 120, 160, 24, 1.5)

    assert result.duration == pytest.approx(36 / 24)
    probe = probe_video(tmp_path / "out.mp4")
    assert (probe.width, probe.height) == (120, 160)
    assert probe.duration == pytest.approx(1.5, abs=0.05)


# --- still_motion ---------------------------------------------------------------------------


def _make_keyframe(path: Path) -> Path:
    from PIL import Image

    Image.new("RGB", (864, 1536), color=(120, 60, 200)).save(path)
    return path


@pytest.mark.parametrize("camera", list(CameraMove))
def test_still_motion_generates_normalized_clip(tmp_path: Path, camera):
    keyframe = _make_keyframe(tmp_path / "keyframe.png")
    out_path = tmp_path / f"clip_{camera}.mp4"
    generator = StillMotionGenerator(width=120, height=160, fps=12)

    result = generator.generate(
        image=keyframe, prompt="", camera=camera, duration=1.0, output_path=out_path
    )

    assert out_path.exists()
    probe = probe_video(out_path)
    assert probe.width == 120
    assert probe.height == 160
    assert probe.fps == 12
    assert not probe.has_audio
    expected_frames = target_frame_count(1.0, 12)
    assert abs(probe.duration - expected_frames / 12) < 0.05
    assert result.width == 120
    assert result.height == 160


# --- stage: skip / force / fallback -----------------------------------------------------------


def make_config(provider: str = "animegen") -> AppConfig:
    return AppConfig(
        llm=LLMConfig(base_url="http://unused", model="fake-model"),
        tts=TTSConfig(provider="vieneu"),
        image=ImageConfig(provider="fake"),
        video=VideoConfig(provider=provider, animegen={"min_free_vram_gb": 24}),
    )


def make_shot(**overrides) -> Shot:
    defaults: dict = dict(
        id="scene_01_a",
        visual=VisualSpec(subject="a young man", setting="a rainy street"),
        camera=CameraMove.push_in,
        duration=2.0,
        image=ImageInfo(path="images/scene_01_a.png", seed=1, provider="fake", prompt="p"),
    )
    defaults.update(overrides)
    return Shot(**defaults)


def make_project(shot: Shot | None = None) -> ProjectSpec:
    return ProjectSpec(
        id="proj_1",
        title="Test",
        style="cinematic_anime",
        content=ContentSpec(topic="test"),
        scenes=[Scene(id="scene_01", narration="...", shots=[shot or make_shot()])],
    )


class FakeVideoGenerator:
    name = "fake"

    def __init__(self):
        self.loaded = False
        self.calls = 0
        self.width = self.height = self.fps = 0

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def generate(self, image, prompt, camera, duration, output_path):
        from anime_short.schemas import VideoResult

        self.calls += 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-mp4")
        return VideoResult(path=output_path, duration=duration, fps=24, width=1080, height=1920)


def _prep_project_image(tmp_path: Path, project: ProjectSpec) -> None:
    img_path = tmp_path / "images" / "scene_01_a.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img_path.write_bytes(b"fake-png")


def test_run_videos_skips_and_forces_with_still_motion_provider(tmp_path: Path, monkeypatch):
    fake = FakeVideoGenerator()
    monkeypatch.setattr(stage_mod, "get_video_generator", lambda config, name: fake)

    config = make_config(provider="still_motion")
    project = make_project()
    _prep_project_image(tmp_path, project)

    run_videos(project, tmp_path, config, force=False)
    assert fake.calls == 1
    shot = project.scenes[0].shots[0]
    assert shot.video is not None
    assert shot.video.mode == VideoMode.still_motion
    assert shot.video.provider == "still_motion"
    assert shot.video.path == "clips/scene_01_a.mp4"
    assert (tmp_path / "clips" / "scene_01_a.mp4").exists()
    assert (tmp_path / "project.json").exists()

    run_videos(project, tmp_path, config, force=False)
    assert fake.calls == 1, "existing clip should be skipped without --force"

    run_videos(project, tmp_path, config, force=True)
    assert fake.calls == 2, "force should regenerate"


def test_run_videos_falls_back_to_still_motion_on_insufficient_vram(tmp_path: Path, monkeypatch):
    class OomOnLoadGenerator:
        name = "animegen"

        def load(self):
            raise InsufficientVRAM("not enough free VRAM")

        def unload(self):
            pass

        def generate(self, *args, **kwargs):  # pragma: no cover - never reached
            raise AssertionError("should not be called")

    fallback = FakeVideoGenerator()

    def fake_get_video_generator(config, name):
        return OomOnLoadGenerator() if name == "animegen" else fallback

    monkeypatch.setattr(stage_mod, "get_video_generator", fake_get_video_generator)

    config = make_config(provider="animegen")
    project = make_project()
    _prep_project_image(tmp_path, project)

    run_videos(project, tmp_path, config, force=False)

    assert fallback.calls == 1
    shot = project.scenes[0].shots[0]
    assert shot.video is not None
    assert shot.video.mode == VideoMode.still_motion
    assert shot.video.provider == "still_motion"
    assert (tmp_path / "clips" / "scene_01_a.mp4").exists()


def test_run_videos_skips_shots_without_image_or_duration(tmp_path: Path, monkeypatch):
    fake = FakeVideoGenerator()
    monkeypatch.setattr(stage_mod, "get_video_generator", lambda config, name: fake)

    config = make_config(provider="still_motion")
    shot = make_shot(image=None)
    project = make_project(shot)

    run_videos(project, tmp_path, config, force=False)

    assert fake.calls == 0
    assert project.scenes[0].shots[0].video is None


def test_run_videos_regenerates_clip_without_state(tmp_path: Path, monkeypatch):
    # A clip file left over from an old keyframe/duration must not be reused: images and
    # timeline clear `shot.video`, and only state + file together mean "done".
    fake = FakeVideoGenerator()
    monkeypatch.setattr(stage_mod, "get_video_generator", lambda config, name: fake)
    project = make_project()
    _prep_project_image(tmp_path, project)
    stale = tmp_path / "clips" / "scene_01_a.mp4"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")

    run_videos(project, tmp_path, make_config(provider="still_motion"))

    assert fake.calls == 1
    assert project.scenes[0].shots[0].video is not None


def test_run_videos_keeps_text_shots_on_still_motion(tmp_path: Path, monkeypatch):
    i2v, still = FakeVideoGenerator(), FakeVideoGenerator()
    monkeypatch.setattr(
        stage_mod, "get_video_generator", lambda c, name: i2v if name == "animegen" else still
    )
    sessions = []
    monkeypatch.setattr(
        stage_mod, "gpu_session", lambda generator, min_free_gb: sessions.append(1) or nullcontext()
    )
    text_shot = make_shot(
        id="scene_01_b", visual=VisualSpec(subject="phone screen", text="Are you there?")
    )
    screen_close_up = make_shot(
        id="scene_01_c",
        visual=VisualSpec(subject="minh", shot_type="Close-up", extra="phone screen lit"),
    )
    held_phone = make_shot(
        id="scene_01_d",
        visual=VisualSpec(subject="minh", shot_type="medium shot", extra="phone screen glowing"),
    )
    project = make_project()
    project.scenes[0].shots += [text_shot, screen_close_up, held_phone]
    _prep_project_image(tmp_path, project)

    run_videos(project, tmp_path, make_config(provider="animegen"))

    modes = [shot.video.mode for shot in project.scenes[0].shots]
    assert modes == [VideoMode.i2v, VideoMode.still_motion, VideoMode.still_motion, VideoMode.i2v]
    assert (i2v.calls, still.calls, len(sessions)) == (2, 2, 1)

    run_videos(project, tmp_path, make_config(provider="animegen"))  # all done: no model load
    assert (i2v.calls, still.calls, len(sessions)) == (2, 2, 1)

    run_videos(project, tmp_path, make_config(provider="animegen"), force=True)
    assert (i2v.calls, still.calls, len(sessions)) == (4, 4, 2)
