"""Pure FFmpeg argument-list builders for the render stage (SPEC §11).

Execution normally goes through `utils.media.run_ffmpeg` (which resolves the configured binary
via `check_ffmpeg()`). Exceptions needing their own subprocess calls: `measure_loudness` and
`measure_voice_peak` (need the raw stderr of a successful run, which `run_ffmpeg` discards).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from anime_short.utils.media import check_ffmpeg


def write_concat_file(paths: list[Path], list_path: Path) -> None:
    lines = [f"file '{p.resolve().as_posix()}'" for p in paths]
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_concat_video_args(list_path: Path, output_path: Path) -> list[str]:
    return ["-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(output_path)]


def build_voice_mix_args(
    voice_paths: list[Path], starts: list[float], total_duration: float, output_path: Path
) -> list[str]:
    """Place each scene's voice WAV at its absolute `start`, mixed into a track that is exactly
    `total_duration` long (silence-padded/trimmed at the end)."""
    args: list[str] = []
    for path in voice_paths:
        args += ["-i", str(path)]
    filters = []
    labels = []
    for i, start in enumerate(starts):
        delay_ms = max(0, round(start * 1000))
        filters.append(f"[{i}:a]adelay={delay_ms}:all=1[v{i}]")
        labels.append(f"[v{i}]")
    filters.append(
        "".join(labels) + f"amix=inputs={len(labels)}:normalize=0,"
        f"apad=whole_dur={total_duration:.3f},atrim=0:{total_duration:.3f}[aout]"
    )
    args += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[aout]",
        "-ar",
        "48000",
        "-ac",
        "1",
        str(output_path),
    ]
    return args


def build_music_duck_args(
    music_path: Path,
    voice_path: Path,
    total_duration: float,
    volume_db: float,
    output_path: Path,
    threshold: float = 0.05,
) -> list[str]:
    """Loop/trim music to `total_duration`, apply `volume_db`, then duck it under the voice
    track with a sidechain compressor.

    `threshold` (linear amplitude, 0-1) should normally be derived from the voice track's
    actual level via `measure_voice_peak` rather than left at its default: a fixed absolute
    threshold can silently fail to trigger the compressor at all if the voice track happens
    to sit quieter than that, e.g. because of upstream TTS gain or `amix` normalisation.
    """
    # sidechaincompress ends its output as soon as EITHER input reaches EOF. The voice track can
    # hit EOF at or slightly before `total_duration` (e.g. rounding in its own trim/encode step),
    # which would otherwise truncate the ducked output early. Padding the sidechain input with
    # silence keeps it from being the one that runs out first, so the ducked track's length
    # tracks the main [music] input instead, which is already trimmed to exactly `total_duration`.
    filters = (
        f"[0:a]aloop=loop=-1:size=2147483647,atrim=0:{total_duration:.3f},"
        f"asetpts=PTS-STARTPTS,volume={volume_db}dB[music];"
        f"[1:a]apad[side];"
        f"[music][side]sidechaincompress=threshold={threshold}:ratio=8:attack=5:release=400[ducked]"
    )
    return [
        "-stream_loop",
        "-1",
        "-i",
        str(music_path),
        "-i",
        str(voice_path),
        "-filter_complex",
        filters,
        "-map",
        "[ducked]",
        "-t",
        f"{total_duration:.3f}",
        "-ar",
        "48000",
        "-ac",
        "1",
        str(output_path),
    ]


def build_audio_mix_args(voice_path: Path, music_path: Path, output_path: Path) -> list[str]:
    return [
        "-i",
        str(voice_path),
        "-i",
        str(music_path),
        "-filter_complex",
        "[0:a][1:a]amix=inputs=2:normalize=0[aout]",
        "-map",
        "[aout]",
        "-ar",
        "48000",
        "-ac",
        "1",
        str(output_path),
    ]


def build_loudnorm_measure_args(
    input_path: Path, target_lufs: float, true_peak: float
) -> list[str]:
    return [
        "-i",
        str(input_path),
        "-af",
        f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=11:print_format=json",
        "-f",
        "null",
        "-",
    ]


def parse_loudnorm_json(stderr: str) -> dict:
    start, end = stderr.rfind("{"), stderr.rfind("}")
    return json.loads(stderr[start : end + 1])


def measure_loudness(input_path: Path, target_lufs: float, true_peak: float) -> dict:
    """Run ffmpeg's loudnorm analysis pass and parse its stats (two-pass loudnorm, step 1)."""
    args = build_loudnorm_measure_args(input_path, target_lufs, true_peak)
    proc = subprocess.run(
        [check_ffmpeg(), "-y", "-hide_banner", *args], capture_output=True, text=True
    )
    return parse_loudnorm_json(proc.stderr)


def measure_voice_peak(input_path: Path) -> float:
    """Return `input_path`'s peak amplitude as a linear 0-1 value, via ffmpeg's `astats` filter.

    Used to size the music-ducking sidechain-compressor threshold relative to the voice
    track's *actual* level (see `build_music_duck_args`), instead of a fixed absolute value
    that could sit above a quieter-than-expected voice track and never trigger compression.
    """
    proc = subprocess.run(
        [
            check_ffmpeg(),
            "-hide_banner",
            "-i",
            str(input_path),
            "-af",
            "astats=metadata=0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    peak_db = -99.0
    for line in proc.stderr.splitlines():
        if "Peak level dB:" in line:
            try:
                peak_db = float(line.rsplit(":", 1)[1])
            except ValueError:
                continue
    return 10 ** (peak_db / 20)


def build_loudnorm_apply_args(
    input_path: Path, measured: dict, target_lufs: float, true_peak: float, output_path: Path
) -> list[str]:
    loudnorm = (
        f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=11:"
        f"measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
        f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:"
        f"offset={measured['target_offset']}:linear=true:print_format=summary"
    )
    return ["-i", str(input_path), "-af", loudnorm, "-ar", "48000", str(output_path)]


def _escape_filter_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace(":", r"\:")


def build_final_encode_args(
    video_path: Path,
    audio_path: Path,
    ass_path: Path,
    fonts_dir: Path,
    crf: int,
    audio_bitrate: str,
    fade_seconds: float,
    total_duration: float,
    output_path: Path,
) -> list[str]:
    """Burn subtitles, apply fade in/out, and encode the final H.264/AAC mp4 in one pass."""
    fade_out_start = max(0.0, total_duration - fade_seconds)
    vf = (
        f"ass='{_escape_filter_path(ass_path)}':fontsdir='{_escape_filter_path(fonts_dir)}',"
        f"fade=t=in:st=0:d={fade_seconds},fade=t=out:st={fade_out_start:.3f}:d={fade_seconds}"
    )
    af = f"afade=t=in:st=0:d={fade_seconds},afade=t=out:st={fade_out_start:.3f}:d={fade_seconds}"
    return [
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-vf",
        vf,
        "-af",
        af,
        "-t",
        f"{total_duration:.3f}",
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
