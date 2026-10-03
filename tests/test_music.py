from __future__ import annotations

import json

import httpx
import pytest

from anime_short.config import load_config
from anime_short.music import stage
from anime_short.music.acestep import AceStepClient
from anime_short.render import audio
from anime_short.render.ffmpeg import measure_loudness
from anime_short.schemas import ContentSpec, ProjectSpec, Scene, Shot, VisualSpec
from anime_short.utils.media import probe_duration, run_ffmpeg


def make_project() -> ProjectSpec:
    shots = [
        Shot(id="scene_01_a", visual=VisualSpec(subject="sky"), start=0.0, duration=4.0),
        Shot(id="scene_01_b", visual=VisualSpec(subject="lake"), start=4.0, duration=6.5),
    ]
    scene = Scene(id="scene_01", narration="x", shots=shots)
    return ProjectSpec(id="p", title="t", content=ContentSpec(topic="t"), scenes=[scene])


def fake_server(requests: list[tuple[str, dict]], polls_before_done: int = 1) -> httpx.Client:
    state = {"polls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        requests.append((request.url.path, body))
        if request.url.path == "/release_task":
            return httpx.Response(200, json={"data": {"task_id": "t1"}})
        if request.url.path == "/query_result":
            state["polls"] += 1
            done = state["polls"] > polls_before_done
            result = json.dumps([{"file": "/v1/audio?path=/tmp/a.mp3"}]) if done else "[]"
            return httpx.Response(200, json={"data": [{"status": int(done), "result": result}]})
        if request.url.path == "/v1/audio":
            return httpx.Response(200, content=b"mp3-bytes")
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://acestep")


def test_acestep_client_releases_polls_and_downloads(tmp_path):
    requests: list[tuple[str, dict]] = []
    client = AceStepClient(
        {"prompt": "soft piano", "bpm": 70, "poll_seconds": 0}, client=fake_server(requests)
    )
    out = client.generate(18.0, 42, tmp_path / "m.mp3")

    assert out.read_bytes() == b"mp3-bytes"
    assert [path for path, _ in requests] == [
        "/release_task",
        "/query_result",
        "/query_result",
        "/v1/audio",
    ]
    task = requests[0][1]
    assert task["prompt"] == "soft piano"
    assert task["lyrics"] == "[Instrumental]"
    assert task["audio_duration"] == 18.0
    assert (task["seed"], task["use_random_seed"], task["batch_size"]) == (42, False, 1)


def test_run_music_generates_a_track_longer_than_the_video(tmp_path, monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, options):
            pass

        def generate(self, duration, seed, output_path):
            calls.append(duration)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"x")
            return output_path

        def close(self):
            pass

    monkeypatch.setattr(stage, "AceStepClient", FakeClient)
    config = load_config({"music": {"enabled": True, "provider": "acestep"}})
    project = make_project()

    stage.run_music(project, tmp_path, config)
    assert project.music == stage.MUSIC_FILE
    assert calls == [round(10.5 + stage.EXTRA_SECONDS)]

    stage.run_music(project, tmp_path, config)  # the track exists: skipped
    assert len(calls) == 1


def _tone(path, seconds, volume, silence=0.0):
    """A sine tone (lavfi amplitude 1/8, times `volume`) with `silence` seconds of silence
    before and after it."""
    pad = f"adelay={silence * 1000:.0f},apad=pad_dur={silence}"
    run_ffmpeg(
        ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
        + ["-af", f"volume={volume},{pad}", str(path)]
    )


def _lufs(path):
    return float(measure_loudness(path, -16, -1.5)["input_i"])


def test_run_music_levels_a_library_track_chosen_by_project_id(tmp_path, monkeypatch):
    library = tmp_path / "lib"
    library.mkdir()
    _tone(library / "quiet.mp3", 3, 0.1, silence=2)  # peak -38 dBFS
    _tone(library / "loud.WAV", 4, 6)  # peak -2.5 dBFS
    (library / "notes.txt").write_text("not audio")
    monkeypatch.setattr(stage, "REPO_ROOT", tmp_path)
    config = load_config({"music": {"enabled": True, "provider": "library", "library": "lib"}})

    durations = set()
    for project_id in ("p1", "p2", "p3", "p4", "p5", "p6"):
        project = make_project()
        project.id = project_id
        stage.run_music(project, tmp_path / project_id, config)
        assert project.music == stage.LIBRARY_MUSIC_FILE
        out = tmp_path / project_id / project.music
        assert abs(_lufs(out) - stage.LIBRARY_LUFS) < 1.0
        durations.add(round(probe_duration(out)))

        again = make_project()
        again.id = project_id
        stage.run_music(again, tmp_path / "again", config)  # same project id: same track
        assert round(probe_duration(tmp_path / "again" / again.music)) == round(probe_duration(out))
    # both tracks get picked across projects; the quiet one without its 2 s of silence each side
    assert durations == {3, 4}


def test_run_music_library_without_tracks_fails(tmp_path, monkeypatch):
    (tmp_path / "lib").mkdir()
    monkeypatch.setattr(stage, "REPO_ROOT", tmp_path)
    config = load_config({"music": {"enabled": True, "provider": "library", "library": "lib"}})
    with pytest.raises(ValueError, match="no audio files"):
        stage.run_music(make_project(), tmp_path / "p", config)


def test_run_music_does_nothing_without_a_provider(tmp_path):
    project = make_project()
    stage.run_music(project, tmp_path, load_config({"music": {"enabled": True, "path": "m.mp3"}}))
    assert project.music is None


def test_final_audio_prefers_the_generated_track(tmp_path, monkeypatch):
    used = []
    monkeypatch.setattr(audio, "build_voice_track", lambda *a: tmp_path / "voice.wav")
    monkeypatch.setattr(
        audio, "duck_music", lambda music, *a: used.append(music) or tmp_path / "d.wav"
    )
    monkeypatch.setattr(audio, "mix_voice_and_music", lambda *a: tmp_path / "mix.wav")
    monkeypatch.setattr(audio, "normalize_loudness", lambda *a: tmp_path / "norm.wav")
    config = load_config({"music": {"enabled": True, "path": "fixed.mp3"}})
    project = make_project()
    project.music = "music/music.mp3"

    audio.build_final_audio(project, tmp_path, config, 10.5, tmp_path)
    assert used == [tmp_path / "music/music.mp3"]


@pytest.mark.parametrize("music", [None, "music/music.mp3"])
def test_final_audio_skips_music_when_disabled(tmp_path, monkeypatch, music):
    monkeypatch.setattr(audio, "build_voice_track", lambda *a: tmp_path / "voice.wav")
    monkeypatch.setattr(audio, "duck_music", lambda *a: pytest.fail("music must not be mixed"))
    monkeypatch.setattr(audio, "normalize_loudness", lambda *a: tmp_path / "norm.wav")
    project = make_project()
    project.music = music
    audio.build_final_audio(project, tmp_path, load_config({}), 10.5, tmp_path)
