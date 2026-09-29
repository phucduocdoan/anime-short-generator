"""ElevenLabs TTS provider (SPEC §8): `eleven_v3` via the `with-timestamps` endpoint,
converting its character-level `alignment` into word timings.

No `ELEVENLABS_API_KEY` is set on this development machine, so the real HTTP call is
only covered by `tests/test_tts.py` via `httpx.MockTransport`; anything hitting the
network is marked `@pytest.mark.network`.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

from anime_short.schemas import AudioResult, VoiceSpec, WordTiming
from anime_short.tts import audio
from anime_short.utils.media import probe_duration

log = logging.getLogger(__name__)

API_BASE_URL = "https://api.elevenlabs.io"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def alignment_to_words(alignment: dict[str, Any]) -> list[WordTiming]:
    """Character-level `alignment` (from the with-timestamps response) -> word timings.

    Words are runs of non-whitespace characters in the alignment's own character
    stream. Deriving boundaries this way (rather than re-splitting the original
    `text`) keeps this correct even if ElevenLabs normalises the text before
    aligning it, since normalisation can change string length/content.
    """
    characters = alignment.get("characters", [])
    starts = alignment.get("character_start_times_seconds", [])
    ends = alignment.get("character_end_times_seconds", [])

    words: list[WordTiming] = []
    chars: list[str] = []
    word_start = 0.0
    word_end = 0.0
    for ch, start, end in zip(characters, starts, ends, strict=False):
        if ch.strip():
            if not chars:
                word_start = start
            chars.append(ch)
            word_end = end
        elif chars:
            words.append(WordTiming(word="".join(chars), start=word_start, end=word_end))
            chars = []
    if chars:
        words.append(WordTiming(word="".join(chars), start=word_start, end=word_end))
    return words


class ElevenLabsProvider:
    name = "elevenlabs"
    min_free_vram_gb = 0.0  # network provider, no GPU/model to load

    def __init__(
        self,
        options: dict[str, Any],
        voice: VoiceSpec,
        client: httpx.Client | None = None,
    ) -> None:
        self._model_id = options.get("model_id", "eleven_v3")
        self._voice_id = voice.voice or options.get("voice_id")
        if not self._voice_id:
            raise ValueError("elevenlabs: no voice_id (set voice.voice or tts.elevenlabs.voice_id)")
        self._speed = voice.speed
        self._max_retries = int(options.get("max_retries", 3))
        self._retry_backoff = float(options.get("retry_backoff_seconds", 1.0))

        if client is not None:
            self._client = client
        else:
            api_key = os.environ.get("ELEVENLABS_API_KEY")
            if not api_key:
                raise RuntimeError("ELEVENLABS_API_KEY is not set")
            self._client = httpx.Client(
                base_url=API_BASE_URL,
                headers={"xi-api-key": api_key},
                timeout=options.get("timeout_seconds", 60),
            )

    def load(self) -> None:
        pass  # network provider, nothing to load

    def unload(self) -> None:
        self._client.close()

    def synthesize(self, text: str, output_path: Path) -> AudioResult:
        response = self._request(text)
        data = response.json()
        samples = audio.pcm16_to_float32(base64.b64decode(data["audio_base64"]))
        words = alignment_to_words(data["alignment"])

        trimmed, offset = audio.trim_silence(samples, audio.TARGET_SAMPLE_RATE)
        trimmed_duration = trimmed.size / audio.TARGET_SAMPLE_RATE
        words = audio.shift_words(words, offset, trimmed_duration)

        audio.write_wav(trimmed, audio.TARGET_SAMPLE_RATE, output_path)
        return AudioResult(path=output_path, duration=probe_duration(output_path), words=words)

    def _request(self, text: str) -> httpx.Response:
        payload = {
            "text": text,
            "model_id": self._model_id,
            "voice_settings": {"speed": self._speed},
        }
        params = {"output_format": "pcm_48000"}

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(
                    f"/v1/text-to-speech/{self._voice_id}/with-timestamps",
                    json=payload,
                    params=params,
                )
            except httpx.HTTPError as exc:
                last_error = exc
            else:
                if response.status_code < 400:
                    return response
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    response.raise_for_status()
                last_error = httpx.HTTPStatusError(
                    f"elevenlabs {response.status_code}",
                    request=response.request,
                    response=response,
                )
            if attempt < self._max_retries:
                delay = self._retry_backoff * (2**attempt)
                log.warning(
                    "elevenlabs request failed (attempt %d), retrying in %.1fs", attempt + 1, delay
                )
                time.sleep(delay)
        assert last_error is not None
        raise last_error
