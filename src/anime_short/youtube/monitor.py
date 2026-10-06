from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from anime_short.config import REPO_ROOT, AppConfig
from anime_short.pipeline.state import load_project
from anime_short.youtube.auth import (
    ANALYTICS_READONLY_SCOPE,
    UPLOAD_SCOPE,
    YOUTUBE_READONLY_SCOPE,
    get_credentials,
)


@dataclass(frozen=True)
class UploadedVideo:
    project_id: str
    project_title: str
    topic: str
    content_format: str
    video_id: str
    url: str
    upload_title: str
    uploaded_at: str | None
    recorded_privacy_status: str | None


def analytics_dir(config: AppConfig) -> Path:
    base = config.workspace_dir
    if not base.is_absolute():
        base = REPO_ROOT / base
    return base / "analytics"


def collect_uploaded_videos(config: AppConfig) -> list[UploadedVideo]:
    base = config.workspace_dir
    if not base.is_absolute():
        base = REPO_ROOT / base
    if not base.exists():
        return []

    videos: list[UploadedVideo] = []
    for project_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        project_json = project_dir / "project.json"
        if not project_json.exists():
            continue
        try:
            project = load_project(project_dir)
        except Exception:
            continue
        for upload in project.youtube_uploads:
            video_id = str(upload.get("video_id") or "").strip()
            if not video_id:
                continue
            videos.append(
                UploadedVideo(
                    project_id=project.id,
                    project_title=project.title,
                    topic=project.content.topic,
                    content_format=project.content.format,
                    video_id=video_id,
                    url=str(upload.get("url") or f"https://youtu.be/{video_id}"),
                    upload_title=str(upload.get("title") or project.title),
                    uploaded_at=upload.get("uploaded_at"),
                    recorded_privacy_status=upload.get("privacy_status"),
                )
            )
    return videos


def fetch_monitor_snapshot(
    config: AppConfig,
    days: int,
    headless: bool = False,
    oauth_port: int = 8080,
) -> dict[str, Any]:
    videos = collect_uploaded_videos(config)
    fetched_at = datetime.now(UTC)
    snapshot = {
        "fetched_at": fetched_at.isoformat(),
        "date": fetched_at.date().isoformat(),
        "days": days,
        "videos": [],
    }
    if not videos:
        return snapshot

    creds = get_credentials(
        token_file=config.youtube.token_file,
        client_secret_file=config.youtube.client_secret_file,
        headless=headless,
        port=oauth_port,
        scopes=[UPLOAD_SCOPE, YOUTUBE_READONLY_SCOPE, ANALYTICS_READONLY_SCOPE],
    )

    from googleapiclient.discovery import build

    youtube = build("youtube", "v3", credentials=creds)
    analytics = build("youtubeAnalytics", "v2", credentials=creds)
    details = _fetch_video_details(youtube, [video.video_id for video in videos])

    start_date = (fetched_at.date() - timedelta(days=max(days - 1, 0))).isoformat()
    end_date = fetched_at.date().isoformat()
    for video in videos:
        item = details.get(video.video_id, {})
        stats = item.get("statistics", {})
        snippet = item.get("snippet", {})
        status = item.get("status", {})
        content_details = item.get("contentDetails", {})
        row = {
            **asdict(video),
            "youtube_title": snippet.get("title"),
            "published_at": snippet.get("publishedAt"),
            "youtube_privacy_status": status.get("privacyStatus"),
            "upload_status": status.get("uploadStatus"),
            "duration": content_details.get("duration"),
            "views": _int(stats.get("viewCount")),
            "likes": _int(stats.get("likeCount")),
            "comments": _int(stats.get("commentCount")),
            "analytics": _fetch_video_analytics(
                analytics,
                video.video_id,
                start_date=start_date,
                end_date=end_date,
            ),
        }
        snapshot["videos"].append(row)
    return snapshot


def save_snapshot(snapshot: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    latest = out_dir / "latest.json"
    latest.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "daily_snapshots.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
    return latest


def load_latest_snapshot(out_dir: Path) -> dict[str, Any]:
    latest = out_dir / "latest.json"
    if not latest.exists():
        raise FileNotFoundError(f"no monitor snapshot at {latest}; run `anime-short monitor fetch`")
    return json.loads(latest.read_text(encoding="utf-8"))


def write_report(snapshot: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_date = snapshot.get("date") or date.today().isoformat()
    path = out_dir / f"report_{report_date}.md"
    path.write_text(render_report(snapshot), encoding="utf-8")
    return path


def render_report(snapshot: dict[str, Any]) -> str:
    videos = sorted(snapshot.get("videos", []), key=lambda v: v.get("views") or 0, reverse=True)
    lines = [
        f"# YouTube Monitor Report - {snapshot.get('date', '')}",
        "",
        f"Fetched at: `{snapshot.get('fetched_at', '')}`",
        f"Window: last {snapshot.get('days', '-')} day(s)",
        "",
        "## Videos",
        "",
        "| Views | Likes | Comments | Privacy | Project | Title | URL |",
        "|---:|---:|---:|---|---|---|---|",
    ]
    for video in videos:
        lines.append(
            "| {views} | {likes} | {comments} | {privacy} | {project} | {title} | {url} |".format(
                views=video.get("views", 0),
                likes=video.get("likes", 0),
                comments=video.get("comments", 0),
                privacy=video.get("youtube_privacy_status")
                or video.get("recorded_privacy_status")
                or "-",
                project=video.get("project_id", "-"),
                title=(video.get("youtube_title") or video.get("upload_title") or "-").replace(
                    "|", "/"
                ),
                url=video.get("url", "-"),
            )
        )
    lines += ["", "## Recommendations", ""]
    lines += [f"- {item}" for item in recommend(snapshot)]
    return "\n".join(lines) + "\n"


def recommend(snapshot: dict[str, Any]) -> list[str]:
    videos = snapshot.get("videos", [])
    if not videos:
        return ["No uploaded videos found yet. Upload a few Shorts before comparing topics."]

    public = [
        v
        for v in videos
        if (v.get("youtube_privacy_status") or v.get("recorded_privacy_status")) == "public"
    ]
    sample = public or videos
    top = max(sample, key=lambda v: v.get("views") or 0)
    recs = [
        "Make the next batch around the strongest current topic: "
        f"`{top.get('topic') or top.get('project_title')}`.",
    ]

    by_format: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for video in sample:
        by_format[video.get("content_format") or "unknown"].append(video)
    best_format = max(
        by_format.items(),
        key=lambda item: sum(v.get("views") or 0 for v in item[1]) / max(len(item[1]), 1),
    )[0]
    recs.append(f"Keep testing `{best_format}` format; it is the best current sample.")

    low_view = [v for v in sample if (v.get("views") or 0) <= 10]
    if low_view:
        recs.append(
            "Review hooks/titles for low-view uploads after 24-48h before changing the whole niche."
        )
    recs.append(
        "Do not delete or privatize videos automatically; use this report as a decision aid."
    )
    return recs


def _fetch_video_details(youtube, video_ids: list[str]) -> dict[str, dict[str, Any]]:
    details: dict[str, dict[str, Any]] = {}
    for chunk in _chunks(video_ids, 50):
        response = (
            youtube.videos()
            .list(part="snippet,statistics,status,contentDetails", id=",".join(chunk))
            .execute()
        )
        for item in response.get("items", []):
            details[item["id"]] = item
    return details


def _fetch_video_analytics(
    analytics,
    video_id: str,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    metrics = (
        "views,estimatedMinutesWatched,averageViewDuration,"
        "averageViewPercentage,subscribersGained"
    )
    try:
        response = (
            analytics.reports()
            .query(
                ids="channel==MINE",
                startDate=start_date,
                endDate=end_date,
                metrics=metrics,
                filters=f"video=={video_id}",
            )
            .execute()
        )
    except Exception as exc:
        return {"error": str(exc)}
    rows = response.get("rows") or []
    if not rows:
        return {}
    values = rows[0]
    keys = [header["name"] for header in response.get("columnHeaders", [])]
    return dict(zip(keys, values, strict=False))


def _chunks(values: list[str], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
