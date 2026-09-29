"""Voice + music audio pipeline for the render stage (SPEC §11 steps 2, 3 and 5)."""

from __future__ import annotations

from pathlib import Path

from anime_short.config import AppConfig
from anime_short.render import ffmpeg as ff
from anime_short.schemas import ProjectSpec
from anime_short.utils.media import run_ffmpeg


def build_voice_track(
    project: ProjectSpec, project_dir: Path, total_duration: float, out_dir: Path
) -> Path:
    voice_paths = [project_dir / scene.audio.path for scene in project.scenes if scene.audio]
    starts = [scene.start for scene in project.scenes if scene.audio]
    out = out_dir / "voice_mix.wav"
    run_ffmpeg(ff.build_voice_mix_args(voice_paths, starts, total_duration, out))
    return out


def duck_music(
    music_path: Path, voice_path: Path, total_duration: float, volume_db: float, out_dir: Path
) -> Path:
    out = out_dir / "music_ducked.wav"
    # Size the sidechain-compressor threshold to the voice track's actual peak rather than a
    # fixed absolute value, so ducking reliably triggers regardless of upstream voice gain.
    # 0.0011 stays just above sidechaincompress's documented minimum (0.000976563): a lower
    # floor makes ffmpeg reject the filter outright ("out of range") for a very quiet voice track.
    voice_peak = ff.measure_voice_peak(voice_path)
    threshold = max(voice_peak * 0.3, 0.0011)
    run_ffmpeg(
        ff.build_music_duck_args(music_path, voice_path, total_duration, volume_db, out, threshold)
    )
    return out


def mix_voice_and_music(voice_path: Path, music_path: Path, out_dir: Path) -> Path:
    out = out_dir / "audio_mixed.wav"
    run_ffmpeg(ff.build_audio_mix_args(voice_path, music_path, out))
    return out


def normalize_loudness(
    input_path: Path, target_lufs: float, true_peak: float, out_dir: Path
) -> Path:
    """Two-pass loudnorm: measure, then apply with the measured stats for an accurate result."""
    measured = ff.measure_loudness(input_path, target_lufs, true_peak)
    out = out_dir / "audio_normalized.wav"
    run_ffmpeg(ff.build_loudnorm_apply_args(input_path, measured, target_lufs, true_peak, out))
    return out


def build_final_audio(
    project: ProjectSpec,
    project_dir: Path,
    config: AppConfig,
    total_duration: float,
    out_dir: Path,
) -> Path:
    audio = build_voice_track(project, project_dir, total_duration, out_dir)
    music_cfg = config.music
    if music_cfg.enabled and music_cfg.path:
        music_path = Path(music_cfg.path)
        if not music_path.is_absolute():
            music_path = project_dir / music_path
        ducked = duck_music(music_path, audio, total_duration, music_cfg.volume_db, out_dir)
        audio = mix_voice_and_music(audio, ducked, out_dir)
    return normalize_loudness(
        audio, config.render.loudness_lufs, config.render.true_peak_db, out_dir
    )
