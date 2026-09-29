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
    assert overrides["image"] == {"provider": "anima"}
    config = load_config(overrides)
    assert config.image.provider == "anima"
    assert config.music.enabled is False


def test_max_scene_seconds_maps_to_timeline():
    overrides = runner.spec_to_overrides(
        {"content": {"topic": "x"}, "script": {"max_scene_seconds": 4}}
    )
    assert load_config(overrides).timeline.max_shot_seconds == 4


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
    assert calls == ["images", "videos", "subtitles", "render"]


def test_stage_functions_are_importable_names():
    for stage in STAGES:
        assert stage in runner.STAGE_FUNCTIONS


def test_cli_lists_all_commands():
    from anime_short.cli import app

    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (*STAGES, "generate", "resume", "status"):
        assert command in result.output
