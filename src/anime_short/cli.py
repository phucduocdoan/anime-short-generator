"""`anime-short` command line (SPEC §13)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from anime_short.config import load_config
from anime_short.pipeline import runner
from anime_short.pipeline.state import load_project, save_project
from anime_short.schemas import STAGES
from anime_short.utils.logging import setup_logging
from anime_short.youtube.metadata import upload_metadata
from anime_short.youtube.validation import validate_short_video

app = typer.Typer(no_args_is_help=True, add_completion=False)
monitor_app = typer.Typer(no_args_is_help=True, add_completion=False)
queue_app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(monitor_app, name="monitor")
app.add_typer(queue_app, name="queue")
console = Console()

ProjectId = Annotated[str, typer.Argument(help="Project id (directory name under workspace/)")]
Force = Annotated[bool, typer.Option("--force", help="Regenerate items that already exist")]


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    setup_logging(verbose)


def _project_dir(project_id: str) -> Path:
    project_dir = load_config().project_dir(project_id)
    if not (project_dir / "project.json").exists():
        raise typer.BadParameter(f"no project at {project_dir}")
    return project_dir


@app.command()
def generate(
    spec: Annotated[Path | None, typer.Argument(exists=True, dir_okay=False)] = None,
    topic: Annotated[str | None, typer.Option(help="Topic, when no spec file is given")] = None,
    duration: Annotated[float, typer.Option(help="Target duration in seconds")] = 45,
    style: Annotated[str, typer.Option()] = "cinematic_anime",
    upload: Annotated[
        bool,
        typer.Option("--upload/--no-upload", help="Upload to YouTube after render finishes"),
    ] = True,
    headless: Annotated[
        bool,
        typer.Option("--headless/--no-headless", help="OAuth headless mode for auto-upload"),
    ] = False,
    oauth_port: Annotated[int, typer.Option("--oauth-port", help="OAuth redirect port")] = 8080,
) -> None:
    """Create a project from spec.yaml (or --topic) and run the whole pipeline."""
    if spec is not None:
        project, project_dir = runner.create_project_from_file(spec)
    elif topic:
        spec_data = {
            "content": {"topic": topic, "duration_seconds": duration},
            "visual": {"style": style},
        }
        project, project_dir = runner.create_project(spec_data)
    else:
        raise typer.BadParameter("give a spec file or --topic")
    console.print(f"project [bold]{project.id}[/] at {project_dir}")
    _run(project_dir, upload_after_render=upload, headless=headless, oauth_port=oauth_port)


@app.command()
def resume(
    project_id: ProjectId,
    upload: Annotated[
        bool,
        typer.Option("--upload/--no-upload", help="Upload to YouTube after render finishes"),
    ] = True,
    headless: Annotated[
        bool,
        typer.Option("--headless/--no-headless", help="OAuth headless mode for auto-upload"),
    ] = False,
    oauth_port: Annotated[int, typer.Option("--oauth-port", help="OAuth redirect port")] = 8080,
) -> None:
    """Run every stage that is not done yet (item-level skip inside each stage)."""
    _run(
        _project_dir(project_id),
        upload_after_render=upload,
        headless=headless,
        oauth_port=oauth_port,
    )


def _run(
    project_dir: Path,
    upload_after_render: bool = True,
    headless: bool = False,
    oauth_port: int = 8080,
) -> None:
    try:
        project = runner.run_pending(project_dir)
    except runner.ProjectLocked as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from None
    if project.output:
        console.print(f"[green]done:[/] {project_dir / project.output}")
    config = runner.project_config(project_dir)
    if upload_after_render and config.youtube.auto_upload_after_render:
        if project.youtube_uploads:
            console.print("[yellow]upload skipped:[/] project already has a YouTube upload")
            return
        _upload_project(
            project_dir,
            title=None,
            description=None,
            tags=[],
            privacy=None,
            scheduled_at=None,
            headless=headless,
            oauth_port=oauth_port,
            dry_run=False,
            skip_existing=True,
        )


def _stage_command(stage: str) -> None:
    def command(project_id: ProjectId, force: Force = False) -> None:
        project_dir = _project_dir(project_id)
        config = runner.project_config(project_dir)
        try:
            runner.run_stage(load_project(project_dir), project_dir, config, stage, force)
        except runner.ProjectLocked as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=1) from None

    command.__doc__ = f"Run only the {stage!r} stage."
    app.command(name=stage)(command)


for _stage in STAGES:
    _stage_command(_stage)


@app.command()
def status(project_id: ProjectId) -> None:
    """Show stage status and per-shot artifacts."""
    project_dir = _project_dir(project_id)
    project = load_project(project_dir)

    stages = Table("stage", "status", title=f"{project.id} — {project.title}")
    for stage in STAGES:
        stages.add_row(stage, project.stages.get(stage, "pending"))
    console.print(stages)

    shots = Table("shot", "start", "dur", "audio", "image", "video")
    for scene in project.scenes:
        audio = f"{scene.audio.duration:.2f}s" if scene.audio else "-"
        for i, shot in enumerate(scene.shots):
            shots.add_row(
                shot.id,
                f"{shot.start:.2f}" if shot.start is not None else "-",
                f"{shot.duration:.2f}" if shot.duration is not None else "-",
                audio if i == 0 else "",
                shot.image.provider if shot.image else "-",
                f"{shot.video.mode}" if shot.video else "-",
            )
    console.print(shots)
    if project.output:
        console.print(f"output: {project_dir / project.output}")


@app.command("youtube-auth")
def youtube_auth(
    headless: Annotated[
        bool,
        typer.Option(
            "--headless/--no-headless",
            help="Print the OAuth URL instead of opening a browser",
        ),
    ] = False,
    port: Annotated[int, typer.Option("--port", help="Local OAuth redirect port")] = 8080,
) -> None:
    """Authenticate YouTube upload access and save the OAuth token."""
    config = load_config()
    if headless:
        console.print(
            Panel(
                "Open a local tunnel before approving the browser URL:\n\n"
                f"ssh -L {port}:localhost:{port} <user>@<server-ip>\n\n"
                f"Then open the printed URL and complete the redirect to localhost:{port}.",
                title="Headless YouTube OAuth",
            )
        )
    try:
        from anime_short.youtube.auth import get_credentials, resolve_repo_path

        get_credentials(
            token_file=config.youtube.token_file,
            client_secret_file=config.youtube.client_secret_file,
            headless=headless,
            port=port,
        )
        console.print(
            f"[green]authenticated:[/] {resolve_repo_path(config.youtube.token_file)}"
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from None


@app.command("upload")
def upload(
    project_id: ProjectId,
    title: Annotated[str | None, typer.Option("--title", help="Override YouTube title")] = None,
    description: Annotated[
        str | None, typer.Option("--description", help="Override YouTube description")
    ] = None,
    tag: Annotated[
        list[str] | None,
        typer.Option("--tag", help="Extra tag; may be passed multiple times"),
    ] = None,
    privacy: Annotated[
        str | None,
        typer.Option("--privacy", help="private, unlisted, or public"),
    ] = None,
    publish_at: Annotated[
        str | None,
        typer.Option("--publish-at", help="Schedule time, e.g. '2026-10-06 19:00'"),
    ] = None,
    publish_date: Annotated[
        str | None,
        typer.Option("--publish-date", help="Schedule date YYYY-MM-DD using youtube.publish_time"),
    ] = None,
    timezone_name: Annotated[
        str | None,
        typer.Option("--timezone", help="IANA timezone for schedule parsing"),
    ] = None,
    headless: Annotated[
        bool,
        typer.Option("--headless/--no-headless", help="OAuth headless mode"),
    ] = False,
    oauth_port: Annotated[int, typer.Option("--oauth-port", help="OAuth redirect port")] = 8080,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate and print metadata without uploading"),
    ] = False,
) -> None:
    """Upload a finished project's output/final.mp4 to YouTube."""
    project_dir = _project_dir(project_id)
    config = runner.project_config(project_dir)
    scheduled_at = _resolve_publish_at(
        publish_at=publish_at,
        publish_date=publish_date,
        publish_time=config.youtube.publish_time,
        timezone_name=timezone_name or config.youtube.publish_timezone,
    )
    privacy_status = privacy or config.youtube.privacy_status
    if privacy_status not in {"private", "unlisted", "public"}:
        raise typer.BadParameter("--privacy must be private, unlisted, or public")
    if scheduled_at and privacy_status != "private":
        console.print("[yellow]scheduled uploads are sent as private until publish time[/]")
    if scheduled_at and scheduled_at.astimezone(UTC) <= datetime.now(UTC) + timedelta(minutes=15):
        raise typer.BadParameter("--publish-at must be at least 15 minutes in the future")

    result = _upload_project(
        project_dir,
        title=title,
        description=description,
        tags=tag or [],
        privacy=privacy_status,
        scheduled_at=scheduled_at,
        headless=headless,
        oauth_port=oauth_port,
        dry_run=dry_run,
        skip_existing=False,
    )
    if result is not None:
        console.print(f"[green]uploaded:[/] {result.url}")


def _upload_project(
    project_dir: Path,
    title: str | None,
    description: str | None,
    tags: list[str],
    privacy: str | None,
    scheduled_at: datetime | None,
    headless: bool,
    oauth_port: int,
    dry_run: bool,
    skip_existing: bool,
):
    config = runner.project_config(project_dir)
    privacy_status = privacy or config.youtube.privacy_status
    if privacy_status not in {"private", "unlisted", "public"}:
        raise typer.BadParameter("YouTube privacy must be private, unlisted, or public")

    try:
        with runner._project_lock(project_dir, "upload"):
            project = load_project(project_dir)
            if skip_existing and project.youtube_uploads:
                console.print("[yellow]upload skipped:[/] project already has a YouTube upload")
                return None
            if not project.output:
                console.print("[red]project has no rendered output; run render/resume first[/]")
                raise typer.Exit(code=1)

            video_path = project_dir / project.output
            probe = validate_short_video(video_path)
            metadata = upload_metadata(
                project,
                config,
                title=title,
                description=description,
                tags=tags,
            )
            _print_upload_summary(video_path, probe, metadata, privacy_status, scheduled_at)
            if not probe.has_audio:
                console.print("[yellow]warning:[/] video has no audio stream")
            if probe.duration > 60:
                console.print("[yellow]note:[/] this Short is over 60 seconds")

            if dry_run:
                console.print("[cyan]dry run:[/] upload skipped")
                return None

            from anime_short.youtube.service import YouTubeService

            service = YouTubeService(config.youtube, headless=headless, oauth_port=oauth_port)
            result = service.upload(
                video_path=video_path,
                metadata=metadata,
                privacy_status=privacy_status,
                publish_at=scheduled_at,
            )
            project.youtube_uploads.append(
                {
                    "video_id": result.video_id,
                    "url": result.url,
                    "title": metadata.title,
                    "privacy_status": result.privacy_status,
                    "publish_at": result.publish_at.isoformat() if result.publish_at else None,
                    "uploaded_at": datetime.now(UTC).isoformat(),
                }
            )
            save_project(project, project_dir)
    except runner.ProjectLocked as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from None

    return result


def _resolve_publish_at(
    publish_at: str | None,
    publish_date: str | None,
    publish_time: str,
    timezone_name: str,
) -> datetime | None:
    if not publish_at and not publish_date:
        return None
    tz = ZoneInfo(timezone_name)
    value = publish_at or f"{publish_date} {publish_time}"
    assert value is not None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=tz)
        except ValueError:
            pass
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise typer.BadParameter(
            "--publish-at must be ISO-like or 'YYYY-MM-DD HH:MM'"
        ) from None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=tz)


def _print_upload_summary(
    video_path: Path,
    probe,
    metadata,
    privacy_status: str,
    publish_at: datetime | None,
) -> None:
    scheduled = publish_at.isoformat() if publish_at else "-"
    tags = ", ".join(metadata.tags) if metadata.tags else "-"
    console.print(
        Panel(
            f"file: {video_path}\n"
            f"video: {probe.width}x{probe.height}, {probe.duration:.2f}s, {probe.fps:.2f} fps\n"
            f"title: {metadata.title}\n"
            f"description:\n{metadata.description}\n"
            f"tags: {tags}\n"
            f"privacy: {privacy_status}\n"
            f"publish_at: {scheduled}",
            title="YouTube Upload",
        )
    )


@monitor_app.command("fetch")
def monitor_fetch(
    days: Annotated[int, typer.Option("--days", help="Analytics window in days")] = 7,
    headless: Annotated[
        bool,
        typer.Option("--headless/--no-headless", help="OAuth headless mode"),
    ] = False,
    oauth_port: Annotated[int, typer.Option("--oauth-port", help="OAuth redirect port")] = 8080,
) -> None:
    """Fetch YouTube stats for uploaded project videos and save a local snapshot."""
    from anime_short.youtube.monitor import (
        analytics_dir,
        fetch_monitor_snapshot,
        save_snapshot,
        write_report,
    )

    config = load_config()
    out_dir = analytics_dir(config)
    snapshot = fetch_monitor_snapshot(
        config,
        days=days,
        headless=headless,
        oauth_port=oauth_port,
    )
    snapshot_path = save_snapshot(snapshot, out_dir)
    report_path = write_report(snapshot, out_dir)
    _print_monitor_table(snapshot)
    console.print(f"[green]snapshot:[/] {snapshot_path}")
    console.print(f"[green]report:[/] {report_path}")


@monitor_app.command("report")
def monitor_report() -> None:
    """Render a markdown report from the latest local monitor snapshot."""
    from anime_short.youtube.monitor import analytics_dir, load_latest_snapshot, write_report

    config = load_config()
    out_dir = analytics_dir(config)
    try:
        snapshot = load_latest_snapshot(out_dir)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from None
    report_path = write_report(snapshot, out_dir)
    _print_monitor_table(snapshot)
    console.print(f"[green]report:[/] {report_path}")


@monitor_app.command("recommend")
def monitor_recommend() -> None:
    """Print recommendations from the latest local monitor snapshot."""
    from anime_short.youtube.monitor import analytics_dir, load_latest_snapshot, recommend

    config = load_config()
    try:
        snapshot = load_latest_snapshot(analytics_dir(config))
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from None
    for item in recommend(snapshot):
        console.print(f"- {item}")


def _print_monitor_table(snapshot) -> None:
    table = Table("views", "likes", "comments", "privacy", "project", "title", title="YouTube")
    videos = sorted(
        snapshot.get("videos", []),
        key=lambda v: v.get("views") or 0,
        reverse=True,
    )
    for video in videos:
        table.add_row(
            str(video.get("views", 0)),
            str(video.get("likes", 0)),
            str(video.get("comments", 0)),
            video.get("youtube_privacy_status")
            or video.get("recorded_privacy_status")
            or "-",
            video.get("project_id", "-"),
            video.get("youtube_title") or video.get("upload_title") or "-",
        )
    console.print(table)


@queue_app.command("plan-day")
def queue_plan_day(
    day: Annotated[
        str | None,
        typer.Option("--date", help="Plan date YYYY-MM-DD; defaults to today in YouTube timezone"),
    ] = None,
    count: Annotated[int, typer.Option("--count", help="Number of videos to plan")] = 5,
    theme: Annotated[str, typer.Option("--theme", help="Daily theme prompt")] = "",
    duration: Annotated[float, typer.Option("--duration", help="Target seconds per Short")] = 45,
) -> None:
    """Create a daily video queue with generated topics and scheduled upload times."""
    from anime_short.content_queue import DEFAULT_THEME, plan_day, save_queue

    config = load_config()
    timezone_name = config.youtube.publish_timezone
    plan_date = day or datetime.now(ZoneInfo(timezone_name)).date().isoformat()
    queue = plan_day(
        config,
        day=plan_date,
        count=count,
        theme=theme or DEFAULT_THEME,
        timezone_name=timezone_name,
        duration_seconds=duration,
    )
    path = save_queue(config, queue)
    console.print(f"[green]queue:[/] {path}")
    _print_queue_table(queue)


@queue_app.command("status")
def queue_status(
    day: Annotated[
        str | None,
        typer.Option("--date", help="Plan date YYYY-MM-DD; defaults to today in YouTube timezone"),
    ] = None,
) -> None:
    """Show the daily queue."""
    from anime_short.content_queue import load_queue

    config = load_config()
    plan_date = day or datetime.now(ZoneInfo(config.youtube.publish_timezone)).date().isoformat()
    try:
        queue = load_queue(config, plan_date)
    except FileNotFoundError as exc:
        console.print(f"[red]queue not found:[/] {exc}")
        raise typer.Exit(code=1) from None
    _print_queue_table(queue)


@queue_app.command("run")
def queue_run(
    day: Annotated[
        str | None,
        typer.Option("--date", help="Plan date YYYY-MM-DD; defaults to today in YouTube timezone"),
    ] = None,
    limit: Annotated[int | None, typer.Option("--limit", help="Max pending items to run")] = None,
    headless: Annotated[
        bool,
        typer.Option("--headless/--no-headless", help="OAuth headless mode for uploads"),
    ] = False,
    oauth_port: Annotated[int, typer.Option("--oauth-port", help="OAuth redirect port")] = 28080,
) -> None:
    """Generate pending queued videos and upload them with their scheduled publish times."""
    from anime_short.content_queue import load_queue, save_queue, write_spec

    config = load_config()
    plan_date = day or datetime.now(ZoneInfo(config.youtube.publish_timezone)).date().isoformat()
    try:
        queue = load_queue(config, plan_date)
    except FileNotFoundError as exc:
        console.print(f"[red]queue not found:[/] {exc}")
        raise typer.Exit(code=1) from None

    pending = [
        item
        for item in queue.get("items", [])
        if item.get("status") in {"pending", "failed"}
    ]
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        console.print("[yellow]no pending queue items[/]")
        return

    for item in pending:
        console.print(f"[cyan]queue item:[/] {item['id']} {item['title']}")
        try:
            spec_path = write_spec(config, item)
            project_dir = _ensure_queue_project(spec_path, item["project_id"])
            item["project_dir"] = str(project_dir)
            project = runner.run_pending(project_dir)
            if project.output:
                console.print(f"[green]done:[/] {project_dir / project.output}")
            scheduled_at = datetime.fromisoformat(item["scheduled_at"])
            minimum = datetime.now(scheduled_at.tzinfo) + timedelta(minutes=15)
            upload_time = scheduled_at if scheduled_at > minimum else None
            privacy = "private" if upload_time else "public"
            result = _upload_project(
                project_dir,
                title=None,
                description=None,
                tags=[],
                privacy=privacy,
                scheduled_at=upload_time,
                headless=headless,
                oauth_port=oauth_port,
                dry_run=False,
                skip_existing=True,
            )
            if result is not None:
                item["url"] = result.url
            item["status"] = "uploaded"
            item["error"] = None
        except Exception as exc:
            item["status"] = "failed"
            item["error"] = str(exc)
            save_queue(config, queue)
            console.print(f"[red]failed:[/] {exc}")
            raise
        save_queue(config, queue)
    _print_queue_table(queue)


def _ensure_queue_project(spec_path: Path, project_id: str) -> Path:
    project_dir = load_config().project_dir(project_id)
    if (project_dir / "project.json").exists():
        return project_dir
    project, project_dir = runner.create_project_from_file(spec_path)
    console.print(f"project [bold]{project.id}[/] at {project_dir}")
    return project_dir


def _print_queue_table(queue: dict) -> None:
    table = Table("id", "status", "scheduled", "project", "title", "url", title="Daily Queue")
    for item in queue.get("items", []):
        table.add_row(
            item.get("id", "-"),
            item.get("status", "-"),
            item.get("scheduled_at", "-"),
            item.get("project_id", "-"),
            item.get("title", "-"),
            item.get("url") or "-",
        )
    console.print(table)


if __name__ == "__main__":
    app()
