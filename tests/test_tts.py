"""Unit tests for the TTS stage (SPEC §8): silence trimming, word-timing offsets,
alignment conversion, the ElevenLabs HTTP client (mocked), the provider factory,
and the stage's skip/force behaviour. No network, no GPU.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf

from anime_short.config import load_config
from anime_short.schemas import (
    AudioResult,
    Character,
    ContentSpec,
    ProjectSpec,
    Scene,
    VoiceSpec,
    WordTiming,
)
from anime_short.tts import audio, get_tts
from anime_short.tts.elevenlabs import ElevenLabsProvider, alignment_to_words
from anime_short.tts.stage import run_tts
from anime_short.tts.vieneu import VieNeuProvider

# --- audio.py ---------------------------------------------------------------------------


def _tone_with_silence(sr: int, silence_s: float, tone_s: float) -> np.ndarray:
    silence = np.zeros(int(silence_s * sr), dtype=np.float32)
    tone = np.full(int(tone_s * sr), 0.5, dtype=np.float32)
    return np.concatenate([silence, tone, silence])


def test_trim_silence_caps_leading_and_trailing_edges():
    sr = 1000
    samples = _tone_with_silence(sr, silence_s=0.5, tone_s=1.0)

    trimmed, offset = audio.trim_silence(samples, sr, max_edge_silence=0.15)

    assert offset == pytest.approx(0.35)  # 0.5s silence - 0.15s kept
    first_tone_idx = np.flatnonzero(np.abs(trimmed) > 0.01)[0]
    last_tone_idx = np.flatnonzero(np.abs(trimmed) > 0.01)[-1]
    assert first_tone_idx / sr <= 0.15
    assert (trimmed.size - 1 - last_tone_idx) / sr <= 0.15


def test_trim_silence_leaves_short_edges_untouched():
    sr = 1000
    samples = _tone_with_silence(sr, silence_s=0.05, tone_s=0.2)

    trimmed, offset = audio.trim_silence(samples, sr, max_edge_silence=0.15)

    assert offset == 0.0
    assert trimmed.size == samples.size


def test_trim_silence_all_silent_returns_unchanged():
    samples = np.zeros(500, dtype=np.float32)
    trimmed, offset = audio.trim_silence(samples, 1000)
    assert offset == 0.0
    assert trimmed.size == samples.size


def test_shift_words_offsets_and_drops_out_of_range():
    words = [
        WordTiming(word="a", start=0.1, end=0.3),
        WordTiming(word="b", start=0.5, end=0.7),
        WordTiming(word="c", start=1.5, end=1.6),
    ]

    shifted = audio.shift_words(words, offset=0.2, max_duration=1.0)

    assert [w.word for w in shifted] == ["a", "b"]
    assert shifted[0].start == pytest.approx(0.0)  # clipped, was negative
    assert shifted[0].end == pytest.approx(0.1)
    assert shifted[1].start == pytest.approx(0.3)
    assert shifted[1].end == pytest.approx(0.5)


def test_pcm16_to_float32_round_trip():
    original = np.array([-1.0, -0.5, 0.0, 0.5], dtype=np.float32)
    pcm = (original * 32767).astype("<i2").tobytes()
    decoded = audio.pcm16_to_float32(pcm)
    assert decoded == pytest.approx(original, abs=1e-3)


def test_apply_speed_changes_duration(tmp_path: Path):
    sr = 48_000
    tone = np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32) * 0.3
    path = tmp_path / "tone.wav"
    audio.write_wav(tone, sr, path)

    audio.apply_speed(path, 2.5)  # exercises atempo chaining (2.5 > 2.0 cap)

    info = sf.info(str(path))
    assert info.samplerate == sr
    assert info.channels == 1
    assert info.frames / sr == pytest.approx(1.0 / 2.5, rel=0.05)


# --- elevenlabs.py ------------------------------------------------------------------------


def test_alignment_to_words_splits_on_whitespace():
    text = "hi there"
    chars = list(text)
    starts = [i * 0.1 for i in range(len(chars))]
    ends = [s + 0.1 for s in starts]
    alignment = {
        "characters": chars,
        "character_start_times_seconds": starts,
        "character_end_times_seconds": ends,
    }

    words = alignment_to_words(alignment)

    assert [w.word for w in words] == ["hi", "there"]
    assert words[0].start == pytest.approx(0.0)
    assert words[0].end == pytest.approx(0.2)
    assert words[1].start == pytest.approx(0.3)
    assert words[1].end == pytest.approx(0.8)


def _fake_alignment_response(
    text: str, sr: int = audio.TARGET_SAMPLE_RATE
) -> tuple[np.ndarray, dict]:
    """A response shaped like ElevenLabs' with-timestamps endpoint: 0.3s silence,
    one "word" of tone per non-space run of `text`, 0.3s trailing silence."""
    n_words = max(1, len(text.split()))
    word_dur = 0.4
    silence = np.zeros(int(0.3 * sr), dtype=np.float32)
    tone = np.tile(
        (np.sin(2 * np.pi * 220 * np.arange(int(word_dur * sr)) / sr) * 0.3).astype(np.float32),
        n_words,
    )
    samples = np.concatenate([silence, tone, silence])

    chars = list(text)
    t = 0.3

    starts, ends = [], []
    per_char = word_dur * n_words / max(1, len(chars))
    for _ch in chars:
        starts.append(t)
        t += per_char
        ends.append(t)
    alignment = {
        "characters": chars,
        "character_start_times_seconds": starts,
        "character_end_times_seconds": ends,
    }
    return samples, alignment


def _mock_transport(status_codes: list[int], text: str) -> httpx.MockTransport:
    samples, alignment = _fake_alignment_response(text)
    pcm_bytes = (samples * 32767).astype("<i2").tobytes()
    body = {
        "audio_base64": base64.b64encode(pcm_bytes).decode("ascii"),
        "alignment": alignment,
        "normalized_alignment": alignment,
    }
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        code = status_codes[min(calls["n"], len(status_codes) - 1)]
        calls["n"] += 1
        if code >= 400:
            return httpx.Response(code, json={"detail": "error"}, request=request)
        return httpx.Response(code, json=body, request=request)

    transport = httpx.MockTransport(handler)
    transport.calls = calls  # type: ignore[attr-defined]
    return transport


def _client(transport: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(base_url="https://api.elevenlabs.io", transport=transport)


def test_elevenlabs_synthesize_writes_48k_mono_wav(tmp_path: Path):
    text = "xin chao ban"
    transport = _mock_transport([200], text)
    provider = ElevenLabsProvider(
        {"voice_id": "voice123", "max_retries": 0},
        VoiceSpec(provider="elevenlabs"),
        client=_client(transport),
    )

    output_path = tmp_path / "scene_01.wav"
    result = provider.synthesize(text, output_path)

    assert output_path.exists()
    info = sf.info(str(output_path))
    assert info.samplerate == audio.TARGET_SAMPLE_RATE
    assert info.channels == 1
    assert isinstance(result, AudioResult)
    assert result.duration > 0
    assert len(result.words) == len(text.split())
    assert all(0.0 <= w.start < w.end <= result.duration + 1e-6 for w in result.words)


def test_elevenlabs_retries_on_429_then_succeeds(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("anime_short.tts.elevenlabs.time.sleep", lambda _s: None)
    text = "xin chao"
    transport = _mock_transport([429, 200], text)
    provider = ElevenLabsProvider(
        {"voice_id": "voice123", "max_retries": 3},
        VoiceSpec(provider="elevenlabs"),
        client=_client(transport),
    )

    result = provider.synthesize(text, tmp_path / "out.wav")

    assert transport.calls["n"] == 2  # type: ignore[attr-defined]
    assert result.duration > 0


def test_elevenlabs_does_not_retry_on_4xx(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("anime_short.tts.elevenlabs.time.sleep", lambda _s: None)
    transport = _mock_transport([401], "xin chao")
    provider = ElevenLabsProvider(
        {"voice_id": "voice123", "max_retries": 3},
        VoiceSpec(provider="elevenlabs"),
        client=_client(transport),
    )

    with pytest.raises(httpx.HTTPStatusError):
        provider.synthesize("xin chao", tmp_path / "out.wav")

    assert transport.calls["n"] == 1  # type: ignore[attr-defined]


def test_elevenlabs_requires_voice_id():
    with pytest.raises(ValueError):
        ElevenLabsProvider({}, VoiceSpec(provider="elevenlabs"), client=httpx.Client())


def test_elevenlabs_requires_api_key_without_injected_client(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        ElevenLabsProvider({"voice_id": "v1"}, VoiceSpec(provider="elevenlabs"))


# --- factory --------------------------------------------------------------------------


def test_get_tts_selects_vieneu_by_default():
    config = load_config()
    provider = get_tts(config, VoiceSpec(provider="vieneu"))
    assert isinstance(provider, VieNeuProvider)
    assert provider.name == "vieneu"


def test_get_tts_selects_elevenlabs(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    config = load_config()
    provider = get_tts(config, VoiceSpec(provider="elevenlabs", voice="voice123"))
    assert isinstance(provider, ElevenLabsProvider)
    assert provider.name == "elevenlabs"


def test_get_tts_unknown_provider_raises():
    config = load_config()
    with pytest.raises(ValueError):
        get_tts(config, VoiceSpec(provider="not-a-provider"))


# --- stage.py -------------------------------------------------------------------------


class _DummyProvider:
    name = "dummy"
    min_free_vram_gb = 0.0

    def __init__(self) -> None:
        self.calls = 0
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def synthesize(self, text: str, output_path: Path) -> AudioResult:
        self.calls += 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"RIFF____WAVEfmt fake")
        return AudioResult(path=output_path, duration=1.23, words=[])


def _make_project() -> ProjectSpec:
    return ProjectSpec(
        id="proj",
        title="Test",
        content=ContentSpec(topic="t"),
        characters={"a": Character(name="A", tags="1girl", description="a girl")},
        scenes=[
            Scene(id="scene_01", narration="Xin chao."),
            Scene(id="scene_02", narration="Tam biet."),
        ],
    )


def test_run_tts_skips_existing_then_respects_force(tmp_path: Path, monkeypatch):
    provider = _DummyProvider()
    monkeypatch.setattr("anime_short.tts.stage.get_tts", lambda config, voice: provider)

    config = load_config()
    project = _make_project()
    project_dir = tmp_path / "proj"

    run_tts(project, project_dir, config)
    assert provider.calls == 2
    assert all(s.audio is not None and s.audio.duration == 1.23 for s in project.scenes)
    saved = json.loads((project_dir / "project.json").read_text())
    assert saved["scenes"][0]["audio"]["path"] == "audio/scene_01.wav"

    run_tts(project, project_dir, config)  # nothing pending, no new synthesis
    assert provider.calls == 2

    run_tts(project, project_dir, config, force=True)
    assert provider.calls == 4


def test_run_tts_regenerates_wav_without_state(tmp_path: Path, monkeypatch):
    """A wav on disk that project.json does not record (crash before save) must be redone."""
    provider = _DummyProvider()
    monkeypatch.setattr("anime_short.tts.stage.get_tts", lambda config, voice: provider)
    project = _make_project()
    project_dir = tmp_path / "proj"
    (project_dir / "audio").mkdir(parents=True)
    (project_dir / "audio" / "scene_01.wav").write_bytes(b"stale")

    run_tts(project, project_dir, load_config())

    assert provider.calls == 2
    assert all(s.audio is not None for s in project.scenes)
