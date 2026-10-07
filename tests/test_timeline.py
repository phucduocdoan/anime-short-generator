from __future__ import annotations

import pytest

from anime_short.config import AppConfig, load_config
from anime_short.pipeline.timeline import (
    compute_cut_points,
    durations_from_points,
    merge_short_durations,
    plan_scene_shots,
    resolve_scene,
    resolve_spacing,
    run_timeline,
    snap_to_words,
)
from anime_short.schemas import (
    AudioInfo,
    ContentSpec,
    ProjectSpec,
    Scene,
    Shot,
    StageStatus,
    VisualSpec,
    WordTiming,
)


def make_shot(shot_id: str, weight: float = 1.0) -> Shot:
    return Shot(id=shot_id, visual=VisualSpec(subject="a girl"), weight=weight)


def make_scene(
    scene_id: str, duration: float, n_shots: int, words: list[WordTiming] | None = None
) -> Scene:
    shots = [make_shot(f"{scene_id}_{chr(ord('a') + i)}") for i in range(n_shots)]
    return Scene(
        id=scene_id,
        narration="xin chao the gioi",
        audio=AudioInfo(path=f"audio/{scene_id}.wav", duration=duration, words=words or []),
        shots=shots,
    )


@pytest.fixture
def config() -> AppConfig:
    return load_config(
        {
            "llm": {"base_url": "http://x", "model": "m"},
            "tts": {"provider": "vieneu"},
            "image": {"provider": "anima"},
            "video": {"provider": "animegen"},
        }
    )


# --- pure helpers -----------------------------------------------------------------------------


def test_compute_cut_points_equal_weights():
    points = compute_cut_points(10.0, [1.0, 1.0, 1.0])
    assert points == pytest.approx([10 / 3, 20 / 3])


def test_compute_cut_points_single_shot():
    assert compute_cut_points(10.0, [1.0]) == []


def test_compute_cut_points_weighted():
    points = compute_cut_points(9.0, [1.0, 2.0])
    assert points == pytest.approx([3.0])


def test_durations_from_points():
    assert durations_from_points([3.0, 6.0], 10.0) == pytest.approx([3.0, 3.0, 4.0])


def test_snap_to_words_picks_nearest():
    word_ends = [2.9, 5.1, 7.0]
    snapped = snap_to_words([3.0, 6.0], word_ends, 10.0)
    assert snapped == pytest.approx([2.9, 5.1])


def test_snap_to_words_no_words_is_noop():
    assert snap_to_words([3.0, 6.0], [], 10.0) == [3.0, 6.0]


def test_snap_to_words_keeps_strictly_increasing():
    # both points would naively snap to the same word end
    snapped = snap_to_words([3.0, 3.1], [3.05], 10.0)
    assert snapped[0] < snapped[1]


def test_merge_short_durations_merges_forward():
    groups = merge_short_durations([0.5, 2.0, 2.0], min_shot_seconds=1.5)
    assert groups == [[0, 1], [2]]


def test_merge_short_durations_merges_last_into_previous():
    groups = merge_short_durations([2.0, 2.0, 0.5], min_shot_seconds=1.5)
    assert groups == [[0], [1, 2]]


def test_merge_short_durations_keeps_single_shot():
    assert merge_short_durations([0.2], min_shot_seconds=1.5) == [[0]]


def test_merge_short_durations_chains():
    # three tiny shots collapse down to one
    groups = merge_short_durations([0.2, 0.2, 0.2], min_shot_seconds=1.5)
    assert groups == [[0, 1, 2]]


# --- plan_scene_shots --------------------------------------------------------------------------


def test_plan_scene_shots_equal_weights_no_words():
    scene = make_scene("scene_01", duration=9.0, n_shots=3)
    durations, groups = plan_scene_shots(scene, min_shot_seconds=1.5)
    assert durations == pytest.approx([3.0, 3.0, 3.0])
    assert groups == [[0], [1], [2]]


def test_plan_scene_shots_merges_short_shot():
    scene = make_scene("scene_01", duration=4.0, n_shots=3)  # 1.33s each, below 1.5 min
    durations, groups = plan_scene_shots(scene, min_shot_seconds=1.5)
    assert sum(durations) == pytest.approx(4.0)
    assert all(d >= 1.5 - 1e-9 or len(groups) == 1 for d in durations)
    assert len(groups) < 3


def test_plan_scene_shots_snaps_to_words():
    words = [
        WordTiming(word="xin", start=0.0, end=1.0),
        WordTiming(word="chao", start=1.0, end=2.9),
        WordTiming(word="gioi", start=2.9, end=6.0),
    ]
    scene = make_scene("scene_01", duration=6.0, n_shots=2, words=words)
    durations, _groups = plan_scene_shots(scene, min_shot_seconds=1.5)
    assert sum(durations) == pytest.approx(6.0)
    assert durations[0] == pytest.approx(2.9, abs=0.02)


def test_plan_scene_shots_durations_sum_exactly():
    scene = make_scene("scene_01", duration=7.3, n_shots=3)
    durations, _ = plan_scene_shots(scene, min_shot_seconds=1.5)
    assert sum(durations) == pytest.approx(7.3)


# --- resolve_scene (replan hook) ----------------------------------------------------------------


def test_resolve_scene_replans_when_shot_too_long(config, monkeypatch):
    scene = make_scene("scene_01", duration=12.0, n_shots=1)  # 12s, way over max 5.0s
    project = ProjectSpec(id="p", title="t", content=ContentSpec(topic="t"), scenes=[scene])

    calls = []

    def fake_replan(project_, scene_, config_):
        calls.append(scene_.id)
        scene_.shots = [make_shot("scene_01_a"), make_shot("scene_01_b"), make_shot("scene_01_c")]

    import anime_short.script.stage as script_stage_module

    monkeypatch.setattr(script_stage_module, "replan_scene", fake_replan, raising=False)

    durations = resolve_scene(project, scene, config)
    assert calls == ["scene_01"]
    assert len(durations) == 3
    assert sum(durations) == pytest.approx(12.0)


def test_resolve_scene_logs_and_keeps_when_still_too_long(config, monkeypatch, caplog):
    scene = make_scene("scene_01", duration=12.0, n_shots=1)
    project = ProjectSpec(id="p", title="t", content=ContentSpec(topic="t"), scenes=[scene])

    def fake_replan(project_, scene_, config_):
        pass  # still a single shot afterwards -> still too long

    import anime_short.script.stage as script_stage_module

    monkeypatch.setattr(script_stage_module, "replan_scene", fake_replan, raising=False)

    with caplog.at_level("WARNING"):
        durations = resolve_scene(project, scene, config)
    assert durations == pytest.approx([12.0])
    assert "still has a shot" in caplog.text


# --- run_timeline (integration) ------------------------------------------------------------------


def test_run_timeline_sets_absolute_starts_and_stages(config, tmp_path):
    scenes = [
        make_scene("scene_01", duration=4.0, n_shots=2),
        make_scene("scene_02", duration=3.0, n_shots=1),
    ]
    project = ProjectSpec(id="p", title="t", content=ContentSpec(topic="t"), scenes=scenes)

    run_timeline(project, tmp_path, config)

    assert project.stages["timeline"] == StageStatus.done
    assert project.scenes[0].start == pytest.approx(0.0)
    gap = config.timeline.scene_gap_seconds
    assert project.scenes[1].start == pytest.approx(4.0 + gap)

    all_shots = project.all_shots()
    assert all_shots[0].start == pytest.approx(0.0)
    for shot in all_shots:
        assert shot.start is not None
        assert shot.duration is not None

    tail = config.timeline.tail_seconds
    last_shot = project.scenes[-1].shots[-1]
    total = last_shot.start + last_shot.duration
    assert total == pytest.approx(4.0 + gap + 3.0 + tail)

    # project.json is saved
    assert (tmp_path / "project.json").exists()


def test_run_timeline_shrinks_silence_to_fit_max_total(tmp_path):
    config = load_config(
        {
            "llm": {"base_url": "http://x", "model": "m"},
            "tts": {"provider": "vieneu"},
            "image": {"provider": "anima"},
            "video": {"provider": "animegen"},
            "timeline": {
                "scene_gap_seconds": 1.0,
                "tail_seconds": 1.5,
                "max_total_seconds": 58.5,
            },
        }
    )
    scenes = [make_scene(f"scene_{i:02d}", duration=4.0, n_shots=1) for i in range(1, 15)]
    project = ProjectSpec(id="p", title="t", content=ContentSpec(topic="t"), scenes=scenes)

    run_timeline(project, tmp_path, config)

    last_shot = project.scenes[-1].shots[-1]
    assert last_shot.start is not None
    assert last_shot.duration is not None
    assert last_shot.start + last_shot.duration == pytest.approx(58.5)
    assert project.scenes[1].start == pytest.approx(4.0 + (58.5 - 56.0 - 1.5) / 13)


def test_resolve_spacing_fails_when_voice_alone_exceeds_max(config):
    config.timeline.max_total_seconds = 10.0
    scenes = [make_scene("scene_01", duration=11.0, n_shots=1)]
    project = ProjectSpec(id="p", title="t", content=ContentSpec(topic="t"), scenes=scenes)

    with pytest.raises(ValueError, match="voice audio is 11.00s"):
        resolve_spacing(project, config)


def test_run_timeline_skips_when_already_done(config, tmp_path):
    scenes = [make_scene("scene_01", duration=4.0, n_shots=1)]
    project = ProjectSpec(id="p", title="t", content=ContentSpec(topic="t"), scenes=scenes)
    project.stages["timeline"] = StageStatus.done
    run_timeline(project, tmp_path, config)
    assert project.scenes[0].start is None  # untouched
    assert not (tmp_path / "project.json").exists()


def test_run_timeline_requires_audio(config, tmp_path):
    scene = make_scene("scene_01", duration=1.0, n_shots=1)
    scene.audio = None
    project = ProjectSpec(id="p", title="t", content=ContentSpec(topic="t"), scenes=[scene])
    with pytest.raises(ValueError):
        run_timeline(project, tmp_path, config)


def test_run_timeline_force_clears_clip_only_when_duration_changes(config, tmp_path):
    from anime_short.schemas import VideoInfo, VideoMode

    scene = make_scene("scene_01", duration=4.0, n_shots=1)
    project = ProjectSpec(id="p", title="t", content=ContentSpec(topic="t"), scenes=[scene])
    run_timeline(project, tmp_path, config)
    shot = project.scenes[0].shots[0]
    shot.video = VideoInfo(path="clips/x.mp4", mode=VideoMode.i2v, provider="animegen")

    run_timeline(project, tmp_path, config, force=True)
    assert shot.video is not None, "same duration: clip is still valid"

    scene.audio.duration = 3.0
    run_timeline(project, tmp_path, config, force=True)
    assert shot.video is None
