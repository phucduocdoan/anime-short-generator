from __future__ import annotations

import json

from anime_short.config import load_config
from anime_short.pipeline import runner
from anime_short.pipeline.state import save_project
from anime_short.youtube.monitor import (
    analytics_dir,
    collect_uploaded_videos,
    load_latest_snapshot,
    recommend,
    render_report,
    save_snapshot,
    write_report,
)


def test_collect_uploaded_videos_reads_project_uploads(tmp_path):
    config = load_config({"workspace_dir": str(tmp_path)})
    project, project_dir = runner.create_project({"content": {"topic": "winter love"}}, config)
    project.youtube_uploads.append(
        {
            "video_id": "abc123",
            "url": "https://youtu.be/abc123",
            "title": "Winter Love",
            "privacy_status": "public",
            "uploaded_at": "2026-10-06T04:32:09+00:00",
        }
    )
    save_project(project, project_dir)

    videos = collect_uploaded_videos(config)

    assert len(videos) == 1
    assert videos[0].project_id == project.id
    assert videos[0].video_id == "abc123"
    assert videos[0].topic == "winter love"


def test_snapshot_roundtrip_and_report(tmp_path):
    config = load_config({"workspace_dir": str(tmp_path)})
    snapshot = {
        "fetched_at": "2026-10-06T00:00:00+00:00",
        "date": "2026-10-06",
        "days": 7,
        "videos": [
            {
                "project_id": "winter_love",
                "project_title": "Winter Love Letters",
                "topic": "winter love",
                "content_format": "quotes",
                "video_id": "abc123",
                "url": "https://youtu.be/abc123",
                "upload_title": "Winter Love",
                "youtube_title": "Winter Love",
                "youtube_privacy_status": "public",
                "views": 42,
                "likes": 4,
                "comments": 1,
            }
        ],
    }

    out_dir = analytics_dir(config)
    latest = save_snapshot(snapshot, out_dir)
    report = write_report(snapshot, out_dir)

    assert latest.exists()
    assert json.loads(latest.read_text())["videos"][0]["views"] == 42
    assert load_latest_snapshot(out_dir)["date"] == "2026-10-06"
    assert report.exists()
    assert "Winter Love" in report.read_text()
    assert "| 42 | 4 | 1 | public | winter_love | Winter Love |" in render_report(snapshot)


def test_recommend_uses_best_topic():
    snapshot = {
        "videos": [
            {
                "topic": "winter love",
                "project_title": "Winter Love",
                "content_format": "quotes",
                "youtube_privacy_status": "public",
                "views": 100,
            },
            {
                "topic": "quiet life",
                "project_title": "Quiet Life",
                "content_format": "quotes",
                "youtube_privacy_status": "public",
                "views": 12,
            },
        ]
    }

    recs = recommend(snapshot)

    assert "winter love" in recs[0]
    assert any("quotes" in rec for rec in recs)


def test_report_command_requires_snapshot(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from anime_short import cli
    from anime_short.cli import app

    config = load_config({"workspace_dir": str(tmp_path)})
    monkeypatch.setattr(cli, "load_config", lambda: config)

    result = CliRunner().invoke(app, ["monitor", "report"])

    assert result.exit_code == 1
    assert "run `anime-short monitor fetch`" in result.output
