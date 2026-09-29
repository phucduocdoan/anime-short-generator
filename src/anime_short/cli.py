"""`anime-short` command line (SPEC §13)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from anime_short.config import load_config
from anime_short.pipeline import runner
from anime_short.pipeline.state import load_project
from anime_short.schemas import STAGES
from anime_short.utils.logging import setup_logging

app = typer.Typer(no_args_is_help=True, add_completion=False)
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
    _run(project_dir)


@app.command()
def resume(project_id: ProjectId) -> None:
    """Run every stage that is not done yet (item-level skip inside each stage)."""
    _run(_project_dir(project_id))


def _run(project_dir: Path) -> None:
    project = runner.run_pending(project_dir)
    if project.output:
        console.print(f"[green]done:[/] {project_dir / project.output}")


def _stage_command(stage: str) -> None:
    def command(project_id: ProjectId, force: Force = False) -> None:
        project_dir = _project_dir(project_id)
        config = runner.project_config(project_dir)
        runner.run_stage(load_project(project_dir), project_dir, config, stage, force)

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


if __name__ == "__main__":
    app()
