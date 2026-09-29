"""Concat shot clips, mux with the final audio, and burn subtitles (SPEC §11 steps 1, 4, 6)."""

from __future__ import annotations

from pathlib import Path

from anime_short.config import FONTS_DIR, AppConfig
from anime_short.render import ffmpeg as ff
from anime_short.render.audio import build_final_audio
from anime_short.schemas import ProjectSpec
from anime_short.utils.media import run_ffmpeg


def total_timeline_duration(project: ProjectSpec) -> float:
    last_shot = project.scenes[-1].shots[-1]
    assert last_shot.start is not None and last_shot.duration is not None
    return last_shot.start + last_shot.duration


def concat_shots(project: ProjectSpec, project_dir: Path, out_dir: Path) -> Path:
    clip_paths = [project_dir / shot.video.path for shot in project.all_shots() if shot.video]
    list_path = out_dir / "concat.txt"
    ff.write_concat_file(clip_paths, list_path)
    out = out_dir / "video_concat.mp4"
    run_ffmpeg(ff.build_concat_video_args(list_path, out))
    return out


def compose(project: ProjectSpec, project_dir: Path, config: AppConfig) -> Path:
    out_dir = project_dir / "output" / "_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    total_duration = total_timeline_duration(project)

    video = concat_shots(project, project_dir, out_dir)
    audio = build_final_audio(project, project_dir, config, total_duration, out_dir)

    ass_path = project_dir / "subtitles" / "final.ass"
    final_path = project_dir / "output" / "final.mp4"
    run_ffmpeg(
        ff.build_final_encode_args(
            video,
            audio,
            ass_path,
            FONTS_DIR,
            config.render.crf,
            config.render.audio_bitrate,
            config.render.fade_seconds,
            total_duration,
            final_path,
        )
    )
    return final_path
