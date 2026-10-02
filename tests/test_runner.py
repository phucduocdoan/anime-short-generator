from pathlib import Path

import pytest
from typer.testing import CliRunner

from anime_short.config import load_config, load_yaml
from anime_short.pipeline import runner
from anime_short.pipeline.state import load_project, save_project
from anime_short.schemas import STAGES, StageStatus

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "tin_nhan_10_nam.yaml"


def test_slugify_vietnamese():
    assert runner.slugify("Tin nhắn từ 10 năm sau") == "tin_nhan_tu_10_nam_sau"
    assert runner.slugify("Đường về") == "duong_ve"
    assert runner.slugify("!!!") == "project"


def test_spec_to_project_example():
    project = runner.spec_to_project(load_yaml(EXAMPLE))
    assert project.id == "tin_nhan_10_nam"
    assert project.language == "vi"
    assert project.content.duration_seconds == 45
    assert project.characters["minh"].tags.startswith("1boy")
    assert project.voice.provider == "vieneu"
    assert all(project.stages[s] == StageStatus.pending for s in STAGES)


def test_spec_to_overrides_example_is_valid_config():
    overrides = runner.spec_to_overrides(load_yaml(EXAMPLE))
    assert overrides["image"] == {"provider": "zanime"}
    config = load_config(overrides)
    assert config.image.provider == "zanime"
    assert config.music.enabled is False


def test_max_scene_seconds_maps_to_timeline():
    overrides = runner.spec_to_overrides(
        {"content": {"topic": "x"}, "script": {"max_scene_seconds": 4}}
    )
    assert load_config(overrides).timeline.max_shot_seconds == 4


def test_spec_maps_words_per_second_and_merges_timeline():
    overrides = runner.spec_to_overrides(
        {
            "content": {"topic": "x"},
            "script": {"max_scene_seconds": 6, "words_per_second": 2.8},
            "timeline": {"scene_gap_seconds": 0.7},
        }
    )
    config = load_config(overrides)
    assert config.script.words_per_second == 2.8
    assert config.timeline.max_shot_seconds == 6
    assert config.timeline.scene_gap_seconds == 0.7


def test_spec_render_section_overrides_render_config():
    overrides = runner.spec_to_overrides(
        {"content": {"topic": "x"}, "render": {"film_grain": 14, "vignette": True}}
    )
    config = load_config(overrides)
    assert config.render.film_grain == 14
    assert config.render.vignette is True
    assert config.render.crf == 18


@pytest.fixture
def workspace_config(tmp_path):
    return load_config({"workspace_dir": str(tmp_path)})


def test_create_project_writes_state_and_refuses_duplicates(workspace_config):
    spec = {"content": {"topic": "Chuyến tàu cuối cùng"}}
    project, project_dir = runner.create_project(spec, workspace_config)
    assert project.id == "chuyen_tau_cuoi_cung"
    assert load_project(project_dir).title == "Chuyến tàu cuối cùng"
    assert load_yaml(project_dir / runner.SPEC_FILE) == spec
    for sub in ("audio", "images", "clips", "subtitles", "output"):
        assert (project_dir / sub).is_dir()
    with pytest.raises(FileExistsError):
        runner.create_project(spec, workspace_config)


def test_run_pending_skips_done_and_marks_failure(workspace_config, monkeypatch):
    project, project_dir = runner.create_project({"content": {"topic": "t"}}, workspace_config)
    project.stages["script"] = StageStatus.done
    save_project(project, project_dir)

    calls = []

    def fake(stage):
        def run(project, project_dir, config, force):
            calls.append(stage)
            if stage == "images":
                raise RuntimeError("boom")

        return run

    monkeypatch.setattr(runner, "stage_function", fake)
    with pytest.raises(RuntimeError):
        runner.run_pending(project_dir)

    assert calls == ["plan", "tts", "timeline", "images"]
    stages = load_project(project_dir).stages
    assert stages["plan"] == StageStatus.done
    assert stages["images"] == StageStatus.failed
    assert stages["videos"] == StageStatus.pending

    calls.clear()
    monkeypatch.setattr(runner, "stage_function", lambda stage: lambda *a: calls.append(stage))
    runner.run_pending(project_dir)
    assert calls == ["images", "videos", "subtitles", "music", "render"]


def test_stage_functions_are_importable_names():
    for stage in STAGES:
        assert stage in runner.STAGE_FUNCTIONS


def test_cli_lists_all_commands():
    from anime_short.cli import app

    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (*STAGES, "generate", "resume", "status"):
        assert command in result.output


@pytest.mark.parametrize("force", [True, False])
def test_running_a_stage_resets_later_stages(workspace_config, monkeypatch, force):
    # Without --force a stage still regenerates invalidated items, so later stages must rerun.
    project, project_dir = runner.create_project({"content": {"topic": "t"}}, workspace_config)
    for stage in STAGES:
        project.stages[stage] = StageStatus.done
    monkeypatch.setattr(runner, "stage_function", lambda stage: lambda *a: None)

    runner.run_stage(project, project_dir, workspace_config, "tts", force=force)

    stages = load_project(project_dir).stages
    assert stages["plan"] == StageStatus.done
    assert stages["tts"] == StageStatus.done
    assert all(stages[s] == StageStatus.pending for s in STAGES[STAGES.index("tts") + 1 :])
