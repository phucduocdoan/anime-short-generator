"""Music stage: one background track per project. `music.provider: library` picks a track from the
`music.library` folder of downloaded free tracks; `acestep` generates one long enough for the whole
video. With no provider, the render stage uses the fixed `music.path`."""

from __future__ import annotations

import logging
import random
from pathlib import Path

from anime_short.config import REPO_ROOT, AppConfig
from anime_short.image.stage import deterministic_seed
from anime_short.music.acestep import AceStepClient
from anime_short.pipeline.state import save_project
from anime_short.render import ffmpeg as ff
from anime_short.schemas import ProjectSpec
from anime_short.utils.media import run_ffmpeg

log = logging.getLogger(__name__)

MUSIC_FILE = "music/music.mp3"
LIBRARY_MUSIC_FILE = "music/music.wav"
# Library tracks come from many sources, 20+ dB apart; level them to about what ACE-Step makes
# (-16 LUFS) so `music.volume_db` means the same for every track. Level in mono, as the render
# mixes: a wide stereo track loses up to ~8 dB when downmixed, a narrow one ~3 dB.
LIBRARY_LUFS = -16.0
LIBRARY_TRUE_PEAK = -1.5
_SILENCE = "silenceremove=start_periods=1:start_threshold=-50dB"
TRIM_SILENCE = f"{_SILENCE},areverse,{_SILENCE},areverse"
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
# Generated tracks end abruptly; the render trims and fades out well before the track's end.
EXTRA_SECONDS = 8.0


def total_duration(project: ProjectSpec) -> float:
    shots = [shot for scene in project.scenes for shot in scene.shots]
    if not shots or any(shot.start is None or shot.duration is None for shot in shots):
        raise ValueError("run the timeline stage before music")
    return max(shot.start + shot.duration for shot in shots)


def run_music(
    project: ProjectSpec, project_dir: Path, config: AppConfig, force: bool = False
) -> None:
    music = config.music
    if not (music.enabled and music.provider):
        return
    if project.music and (project_dir / project.music).exists() and not force:
        return
    if music.provider == "library":
        project.music = pick_from_library(project.id, REPO_ROOT / music.library, project_dir)
        save_project(project, project_dir)
        return
    if music.provider != "acestep":
        raise ValueError(f"unknown music provider {music.provider!r}")
    duration = round(total_duration(project) + EXTRA_SECONDS)
    client = AceStepClient(music.acestep)
    try:
        client.generate(duration, deterministic_seed(project.id, "music"), project_dir / MUSIC_FILE)
    finally:
        client.close()
    project.music = MUSIC_FILE
    save_project(project, project_dir)


def pick_from_library(project_id: str, library: Path, project_dir: Path) -> str:
    """Pick one track from `library` by the project id (the same project always gets the same
    track) and write it into the project at a common loudness. The render loops it if it is
    shorter than the video."""
    tracks = sorted(p for p in library.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES)
    if not tracks:
        raise ValueError(f"no audio files in music library {library}")
    track = random.Random(deterministic_seed(project_id, "music")).choice(tracks)
    log.info("music: %s", track.name)
    out = project_dir / LIBRARY_MUSIC_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    # Downloaded tracks can open or close with seconds of silence: no music under the first
    # lines, or a gap where the render loops the track.
    trimmed = out.with_name("music_trimmed.wav")
    run_ffmpeg(["-i", str(track), "-af", TRIM_SILENCE, "-ac", "1", str(trimmed)])
    measured = ff.measure_loudness(trimmed, LIBRARY_LUFS, LIBRARY_TRUE_PEAK)
    run_ffmpeg(
        ff.build_loudnorm_apply_args(trimmed, measured, LIBRARY_LUFS, LIBRARY_TRUE_PEAK, out)
    )
    trimmed.unlink()
    return LIBRARY_MUSIC_FILE
