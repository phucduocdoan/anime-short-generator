from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from anime_short import cli
from anime_short.cli import _resolve_publish_at, app
from anime_short.config import load_config
from anime_short.pipeline import runner
from anime_short.pipeline.state import load_project, save_project
from anime_short.schemas import ContentSpec, ProjectSpec
from anime_short.utils.media import VideoProbe
from anime_short.youtube.auth import _token_has_scopes
from anime_short.youtube.metadata import VideoMetadata, default_metadata, upload_metadata
from anime_short.youtube.validation import validate_short_video


def test_youtube_config_defaults_to_people_and_blogs():
    config = load_config()
    assert config.youtube.category_id == "22"
    assert config.youtube.privacy_status == "private"
    assert config.youtube.metadata_provider == "llm"


def test_spec_youtube_section_overrides_defaults():
    overrides = runner.spec_to_overrides(
        {"content": {"topic": "x"}, "youtube": {"category_id": "24"}}
    )
    assert load_config(overrides).youtube.category_id == "24"


def test_token_scope_check_uses_saved_token_scopes(tmp_path):
    token = tmp_path / "token.json"
    token.write_text(
        '{"scopes": ["https://www.googleapis.com/auth/youtube.upload"]}',
        encoding="utf-8",
    )

    assert _token_has_scopes(token, ["https://www.googleapis.com/auth/youtube.upload"])
    assert not _token_has_scopes(
        token,
        [
            "https://www.googleapis.com/auth/youtube.upload",
            "https://www.googleapis.com/auth/yt-analytics.readonly",
        ],
    )


def test_default_metadata_is_short_safe():
    project = ProjectSpec(
        id="winter_love",
        title="Winter Love Letters",
        language="en",
        content=ContentSpec(topic="winter love philosophy", format="quotes"),
        ending="Some warmth only arrives after the cold.",
    )

    metadata = default_metadata(project)

    assert metadata.title == "Winter Love Letters"
    assert "winter love philosophy" in metadata.description
    assert "shorts" in metadata.tags
    assert "quotes" in metadata.tags


def test_upload_metadata_uses_llm_then_keeps_template_tags(monkeypatch):
    project = ProjectSpec(
        id="winter_love",
        title="Winter Love Letters",
        language="en",
        content=ContentSpec(topic="winter love philosophy", format="quotes"),
    )
    config = load_config()

    class FakeLLM:
        def __init__(self, _config):
            pass

        def complete_structured(self, **_kwargs):
            return VideoMetadata(
                title="When Winter Teaches Love",
                description="A quiet thought for the season.\n\n#Shorts #LoveQuotes",
                tags=["winter love", "quiet thoughts"],
            )

    monkeypatch.setattr("anime_short.youtube.metadata.LLMClient", FakeLLM)

    metadata = upload_metadata(project, config)

    assert metadata.title == "When Winter Teaches Love"
    assert "quiet thought" in metadata.description
    assert "winter love" in metadata.tags
    assert "shorts" in metadata.tags


def test_upload_metadata_falls_back_when_llm_fails(monkeypatch):
    project = ProjectSpec(
        id="winter_love",
        title="Winter Love Letters",
        language="en",
        content=ContentSpec(topic="winter love philosophy", format="quotes"),
    )
    config = load_config()

    class BrokenLLM:
        def __init__(self, _config):
            pass

        def complete_structured(self, **_kwargs):
            raise RuntimeError("offline")

    monkeypatch.setattr("anime_short.youtube.metadata.LLMClient", BrokenLLM)

    metadata = upload_metadata(project, config)

    assert metadata.title == "Winter Love Letters"
    assert "winter love philosophy" in metadata.description


def test_validate_short_video_rejects_wide_video(tmp_path, monkeypatch):
    path = tmp_path / "final.mp4"
    path.write_bytes(b"x")
    monkeypatch.setattr(
        "anime_short.youtube.validation.probe_video",
        lambda _: VideoProbe(width=1920, height=1080, fps=24, duration=45, has_audio=True),
    )

    with pytest.raises(ValueError, match="square or vertical"):
        validate_short_video(path)


def test_validate_short_video_rejects_too_long_video(tmp_path, monkeypatch):
    path = tmp_path / "final.mp4"
    path.write_bytes(b"x")
    monkeypatch.setattr(
        "anime_short.youtube.validation.probe_video",
        lambda _: VideoProbe(width=1080, height=1920, fps=24, duration=181, has_audio=True),
    )

    with pytest.raises(ValueError, match="at most 180"):
        validate_short_video(path)


def test_resolve_publish_date_uses_configured_time_and_timezone():
    publish_at = _resolve_publish_at(
        publish_at=None,
        publish_date="2026-10-06",
        publish_time="19:00",
        timezone_name="Asia/Ho_Chi_Minh",
    )

    assert publish_at == datetime(2026, 10, 6, 19, 0, tzinfo=publish_at.tzinfo)
    assert str(publish_at.tzinfo) == "Asia/Ho_Chi_Minh"


def test_upload_dry_run_validates_and_skips_oauth(tmp_path, monkeypatch):
    config = load_config(
        {"workspace_dir": str(tmp_path), "youtube": {"metadata_provider": "template"}}
    )
    project, project_dir = runner.create_project({"content": {"topic": "winter love"}}, config)
    output = Path("output") / "final.mp4"
    (project_dir / output).write_bytes(b"x")
    project.output = output.as_posix()
    save_project(project, project_dir)

    monkeypatch.setattr("anime_short.cli.load_config", lambda: config)
    monkeypatch.setattr("anime_short.pipeline.runner.load_config", lambda _: config)
    monkeypatch.setattr(
        "anime_short.cli.validate_short_video",
        lambda _: VideoProbe(width=1080, height=1920, fps=24, duration=54, has_audio=True),
    )

    result = CliRunner().invoke(app, ["upload", project.id, "--dry-run"])

    assert result.exit_code == 0
    assert "dry run" in result.output
    assert load_project(project_dir).youtube_uploads == []


def test_run_auto_uploads_after_render_by_default(tmp_path, monkeypatch):
    project = ProjectSpec(
        id="p",
        title="P",
        content=ContentSpec(topic="winter love"),
        output="output/final.mp4",
    )
    config = load_config({"workspace_dir": str(tmp_path)})
    calls = []

    monkeypatch.setattr(cli.runner, "run_pending", lambda _: project)
    monkeypatch.setattr(cli.runner, "project_config", lambda _: config)
    monkeypatch.setattr(cli, "_upload_project", lambda *args, **kwargs: calls.append(kwargs))

    cli._run(tmp_path / "p")

    assert calls
    assert calls[0]["dry_run"] is False
    assert calls[0]["skip_existing"] is True


def test_run_no_upload_flag_skips_auto_upload(tmp_path, monkeypatch):
    project = ProjectSpec(
        id="p",
        title="P",
        content=ContentSpec(topic="winter love"),
        output="output/final.mp4",
    )
    config = load_config({"workspace_dir": str(tmp_path)})

    monkeypatch.setattr(cli.runner, "run_pending", lambda _: project)
    monkeypatch.setattr(cli.runner, "project_config", lambda _: config)
    monkeypatch.setattr(
        cli,
        "_upload_project",
        lambda *args, **kwargs: pytest.fail("auto-upload should be skipped"),
    )

    cli._run(tmp_path / "p", upload_after_render=False)
