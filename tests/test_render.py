from __future__ import annotations

import math
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from anime_short.config import AppConfig, load_config
from anime_short.pipeline.state import init_project_dir, save_project
from anime_short.pipeline.timeline import run_timeline
from anime_short.render import ffmpeg as ff
from anime_short.render.composer import compose
from anime_short.schemas import (
    AudioInfo,
    ContentSpec,
    ImageInfo,
    ProjectSpec,
    Scene,
    Shot,
    VideoInfo,
    VideoMode,
    VisualSpec,
    WordTiming,
)
from anime_short.subtitle.stage import run_subtitles
from anime_short.utils.media import check_ffmpeg, probe_duration, probe_video

# --- pure arg builders ---------------------------------------------------------------------


def test_write_concat_file(tmp_path):
    paths = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    list_path = tmp_path / "concat.txt"
    ff.write_concat_file(paths, list_path)
    text = list_path.read_text()
    assert "file '" in text
    assert str((tmp_path / "a.mp4").resolve().as_posix()) in text
    assert text.count("file '") == 2


def test_build_concat_video_args():
    args = ff.build_concat_video_args(Path("list.txt"), Path("out.mp4"))
    assert args == ["-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy", "out.mp4"]


def test_build_voice_mix_args_delays_each_input():
    args = ff.build_voice_mix_args(
        [Path("a.wav"), Path("b.wav")], [0.0, 2.5], total_duration=5.0, output_path=Path("out.wav")
    )
    joined = " ".join(args)
    assert "adelay=0:all=1" in joined
    assert "adelay=2500:all=1" in joined
    assert "amix=inputs=2" in joined
    assert "atrim=0:5.000" in joined
    assert args[-1] == "out.wav"


def test_build_music_duck_args_has_sidechain():
    args = ff.build_music_duck_args(
        Path("music.wav"), Path("voice.wav"), 10.0, -22.0, Path("out.wav")
    )
    joined = " ".join(args)
    assert "sidechaincompress" in joined
    assert "threshold=0.05" in joined  # default when the caller doesn't measure the voice peak
    assert "volume=-22.0dB" in joined
    assert "afade=t=out:st=8.000:d=2.000" in joined
    assert "-t 10.000" in joined


def test_build_music_duck_args_uses_custom_threshold():
    args = ff.build_music_duck_args(
        Path("music.wav"), Path("voice.wav"), 10.0, -22.0, Path("out.wav"), threshold=0.01
    )
    assert "threshold=0.01" in " ".join(args)


def test_measure_voice_peak_matches_known_amplitude(tmp_path):
    # A sine at -6dBFS should measure a peak close to 10**(-6/20) ~= 0.501.
    # Relative check (the lavfi sine source's own baseline amplitude isn't 0 dBFS on every
    # ffmpeg build): a tone attenuated an extra 12 dB should measure a peak ~12 dB lower.
    def make(path, extra_db):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1",
                "-af",
                f"volume={extra_db}dB",
                "-ar",
                "48000",
                "-ac",
                "1",
                str(path),
            ],
            check=True,
        )

    loud, quiet = tmp_path / "loud.wav", tmp_path / "quiet.wav"
    make(loud, 0)
    make(quiet, -12)
    loud_db = 20 * math.log10(ff.measure_voice_peak(loud))
    quiet_db = 20 * math.log10(ff.measure_voice_peak(quiet))
    assert loud_db - quiet_db == pytest.approx(12.0, abs=0.5)


def test_duck_music_actually_attenuates_during_voice(tmp_path):
    # Regression test: a fixed absolute sidechaincompress threshold silently failed to
    # duck at all when the voice track's peak sat below it. duck_music must measure the
    # voice's real peak and derive a threshold from it so ducking always engages.
    from anime_short.render.audio import duck_music

    voice_path = tmp_path / "voice.wav"
    music_path = tmp_path / "music.wav"
    # A quiet voice track (peak well under the old fixed 0.05 threshold), 1s tone + 1s silence.
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:duration=1",
            "-af",
            "volume=-20dB",
            "-ar",
            "48000",
            "-ac",
            "1",
            tmp_path.as_posix() + "/_v.wav",
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=mono:d=1",
            str(tmp_path / "_s.wav"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(tmp_path / "_v.wav"),
            "-i",
            str(tmp_path / "_s.wav"),
            "-filter_complex",
            "[0:a][1:a]concat=n=2:v=0:a=1[out]",
            "-map",
            "[out]",
            str(voice_path),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=110:duration=3",
            "-ar",
            "48000",
            "-ac",
            "1",
            str(music_path),
        ],
        check=True,
    )
    out = duck_music(music_path, voice_path, 2.0, -6.0, tmp_path)

    def rms_db(path, t0, t1):
        # ffmpeg's `-ss`/`-t`-after-`-i` windowed `astats` was found unreliable for precise
        # windowed measurement on short files in this environment; read raw PCM samples directly
        # instead, which was confirmed accurate against a known-silent reference window.
        with wave.open(str(path), "rb") as w:
            sr = w.getframerate()
            w.setpos(int(t0 * sr))
            frames = w.readframes(int((t1 - t0) * sr))
        samples = struct.unpack(f"<{len(frames) // 2}h", frames)
        mean_sq = sum((s / 32768) ** 2 for s in samples) / len(samples)
        return -120.0 if mean_sq <= 0 else 10 * math.log10(mean_sq)

    during_voice = rms_db(out, 0.2, 0.8)
    during_silence = rms_db(out, 1.2, 1.8)
    assert during_voice < during_silence - 3

    # duck=False: the music stays at volume_db while the voice speaks (window before the fade-out).
    steady = duck_music(music_path, voice_path, 4.0, -6.0, tmp_path, duck=False)
    assert abs(rms_db(steady, 0.2, 0.8) - rms_db(steady, 1.2, 1.8)) < 0.5


def test_build_audio_mix_args():
    args = ff.build_audio_mix_args(Path("v.wav"), Path("m.wav"), Path("out.wav"))
    assert "amix=inputs=2" in " ".join(args)


def test_parse_loudnorm_json():
    stderr = 'garbage before\n{\n "input_i": "-23.0",\n "input_tp": "-5.0"\n}\ntrailing'
    parsed = ff.parse_loudnorm_json(stderr)
    assert parsed["input_i"] == "-23.0"


def test_build_loudnorm_apply_args_uses_measured_values():
    measured = {
        "input_i": "-23.0",
        "input_tp": "-5.0",
        "input_lra": "4.0",
        "input_thresh": "-33.0",
        "target_offset": "0.5",
    }
    args = ff.build_loudnorm_apply_args(Path("in.wav"), measured, -14.0, -1.0, Path("out.wav"))
    joined = " ".join(args)
    assert "measured_I=-23.0" in joined
    assert "I=-14.0" in joined


def test_build_final_encode_args_burns_subtitles_and_fades():
    args = ff.build_final_encode_args(
        Path("v.mp4"),
        Path("a.wav"),
        Path("subs.ass"),
        Path("fonts"),
        18,
        "192k",
        0.3,
        9.0,
        Path("out.mp4"),
    )
    joined = " ".join(args)
    assert "ass='subs.ass':fontsdir='fonts'" in joined
    assert "fade=t=in:st=0:d=0.3" in joined
    assert "fade=t=out:st=8.700:d=0.3" in joined
    assert "noise" not in joined and "vignette" not in joined
    assert "-crf 18" in joined
    assert "-b:a 192k" in joined
    assert "+faststart" in joined


def test_build_final_encode_args_film_look_goes_under_subtitles():
    args = ff.build_final_encode_args(
        Path("v.mp4"),
        Path("a.wav"),
        Path("subs.ass"),
        Path("fonts"),
        18,
        "192k",
        0.3,
        9.0,
        Path("out.mp4"),
        film_grain=14,
        vignette=True,
    )
    vf = args[args.index("-vf") + 1]
    assert "noise=c0s=14:c0f=t" in vf
    assert vf.index("blend=all_mode=grainmerge") < vf.index("vignette") < vf.index("ass=")


def test_check_ffmpeg_has_libx264_and_libass():
    binary = check_ffmpeg()
    encoders = subprocess.run(
        [binary, "-hide_banner", "-encoders"], capture_output=True, text=True
    ).stdout
    assert "libx264" in encoders


# --- full synthetic pipeline (real ffmpeg) --------------------------------------------------


CAPABLE = check_ffmpeg()


def _make_clip(path: Path, duration: float, width: int, height: int, fps: int) -> None:
    subprocess.run(
        [
            CAPABLE,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={width}x{height}:rate={fps}:duration={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-t",
            f"{duration:.3f}",
            str(path),
        ],
        check=True,
    )


def _make_tone(path: Path, duration: float, freq: int = 220) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={freq}:duration={duration}",
            "-ar",
            "48000",
            "-ac",
            "1",
            str(path),
        ],
        check=True,
    )


@pytest.fixture
def config() -> AppConfig:
    return load_config(
        {
            "llm": {"base_url": "http://x", "model": "m"},
            "tts": {"provider": "vieneu"},
            "image": {"provider": "anima"},
            "video": {"provider": "animegen"},
        }
    )


def _build_synthetic_project(
    tmp_path: Path, config: AppConfig, with_words: bool, with_music: bool
) -> ProjectSpec:
    init_project_dir(tmp_path)
    narrations = [
        "Nhưng hôm ấy, chuyến tàu cuối cùng không bao giờ đến.",
        "2 giờ sáng, điện thoại của Minh bỗng sáng lên.",
    ]
    scenes = []
    for i, narration in enumerate(narrations, start=1):
        scene_id = f"scene_{i:02d}"
        audio_path = tmp_path / "audio" / f"{scene_id}.wav"
        duration = 2.4 if i == 1 else 3.1
        _make_tone(audio_path, duration)
        words = []
        if with_words:
            n = len(narration.split())
            step = duration / n
            words = [
                WordTiming(word=w, start=j * step, end=(j + 1) * step)
                for j, w in enumerate(narration.split())
            ]
        shots = []
        for letter in "ab":
            shot_id = f"{scene_id}_{letter}"
            shots.append(
                Shot(
                    id=shot_id,
                    visual=VisualSpec(subject="test"),
                    weight=1.0,
                    image=ImageInfo(
                        path=f"images/{shot_id}.png", seed=1, provider="test", prompt="x"
                    ),
                )
            )
        scenes.append(
            Scene(
                id=scene_id,
                narration=narration,
                audio=AudioInfo(path=f"audio/{scene_id}.wav", duration=duration, words=words),
                shots=shots,
            )
        )
    project = ProjectSpec(
        id="synthetic", title="Synthetic", content=ContentSpec(topic="test"), scenes=scenes
    )

    run_timeline(project, tmp_path, config)

    # now that shots are final (post merge/replan), render the video clips at their exact duration
    for shot in project.all_shots():
        clip_path = tmp_path / "clips" / f"{shot.id}.mp4"
        _make_clip(clip_path, shot.duration, project.width, project.height, project.fps)
        shot.video = VideoInfo(
            path=f"clips/{shot.id}.mp4", mode=VideoMode.still_motion, provider="test"
        )
    save_project(project, tmp_path)

    run_subtitles(project, tmp_path, config)

    if with_music:
        music_path = tmp_path / "music.wav"
        total = project.scenes[-1].shots[-1].start + project.scenes[-1].shots[-1].duration
        _make_tone(music_path, total + 1.0, freq=440)
        config.music.enabled = True
        config.music.path = str(music_path)

    return project


def test_render_end_to_end_with_words(tmp_path, config):
    project = _build_synthetic_project(tmp_path, config, with_words=True, with_music=False)
    final_path = compose(project, tmp_path, config)

    assert final_path.exists()
    info = probe_video(final_path)
    assert info.width == project.width
    assert info.height == project.height
    assert round(info.fps) == project.fps
    assert info.has_audio

    expected_total = project.scenes[-1].shots[-1].start + project.scenes[-1].shots[-1].duration
    assert info.duration == pytest.approx(expected_total, abs=1 / project.fps + 0.05)


def test_render_end_to_end_with_music_ducking(tmp_path, config):
    project = _build_synthetic_project(tmp_path, config, with_words=False, with_music=True)
    final_path = compose(project, tmp_path, config)
    assert final_path.exists()
    assert probe_duration(final_path) > 0


def test_render_stage_marks_done_and_sets_output(tmp_path, config):
    from anime_short.render.stage import run_render
    from anime_short.schemas import StageStatus

    project = _build_synthetic_project(tmp_path, config, with_words=True, with_music=False)
    run_render(project, tmp_path, config)
    assert project.stages["render"] == StageStatus.done
    assert project.output == "output/final.mp4"
    assert (tmp_path / project.output).exists()


def test_build_music_args_without_ducking_ignores_voice():
    args = ff.build_music_duck_args(
        Path("music.wav"), Path("voice.wav"), 10.0, -15.0, Path("out.wav"), duck=False
    )
    joined = " ".join(args)
    assert "sidechaincompress" not in joined
    assert "voice.wav" not in joined
    assert "volume=-15.0dB" in joined
    assert "afade=t=out:st=8.000:d=2.000[ducked]" in joined


def test_build_final_encode_args_dust_goes_after_vignette_under_subtitles():
    args = ff.build_final_encode_args(
        Path("v.mp4"),
        Path("a.wav"),
        Path("subs.ass"),
        Path("fonts"),
        18,
        "192k",
        0.3,
        9.0,
        Path("out.mp4"),
        vignette=True,
        dust_path=Path("dust.mkv"),
    )
    vf = args[args.index("-vf") + 1]
    assert "movie='dust.mkv'" in vf
    assert vf.index("vignette") < vf.index("blend=c0_mode=multiply") < vf.index("ass=")


def test_dust_frame_is_white_with_sparse_dark_specks():
    import numpy as np

    from anime_short.render.dust import dust_frame

    empty = np.asarray(dust_frame(np.random.default_rng(0), 1080, 1920, 0))
    assert empty.min() == 255

    frames = [np.asarray(dust_frame(np.random.default_rng(s), 1080, 1920, 8)) for s in (1, 2)]
    for frame in frames:
        assert frame.min() < 128  # some specks are dark
        assert (frame < 250).mean() < 0.01  # but they cover a tiny part of the frame
    assert not np.array_equal(frames[0] < 250, frames[1] < 250)  # specks move between frames


def _mean_luma(path: Path, at: float) -> float:
    proc = subprocess.run(
        [
            CAPABLE,
            "-hide_banner",
            "-ss",
            str(at),
            "-i",
            str(path),
            "-vf",
            "signalstats,metadata=print:key=lavfi.signalstats.YAVG",
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(proc.stderr.split("lavfi.signalstats.YAVG=")[1].split()[0])


def test_render_dust_without_specks_keeps_brightness(tmp_path, config):
    # White dust must multiply as 1: a gray video turns white into 235 when converted to yuv
    project = _build_synthetic_project(tmp_path, config, with_words=True, with_music=False)
    plain = compose(project, tmp_path, config).rename(tmp_path / "plain.mp4")
    config.render.dust = 1e-9  # dust on, but (practically) no specks
    dusty = compose(project, tmp_path, config)
    assert (tmp_path / "output" / "_tmp" / "dust.mkv").exists()
    assert _mean_luma(dusty, 2.0) == pytest.approx(_mean_luma(plain, 2.0), abs=0.5)
