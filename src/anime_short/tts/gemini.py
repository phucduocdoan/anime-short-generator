"""Gemini TTS provider (SPEC §8): Google's `generateContent` API with a TTS model (default
`gemini-3.8-flash-tts`) and a prebuilt voice (`voice.voice`, e.g. "Enceladus").

The model takes no system instruction; the delivery is steered by a structured prompt (audio
profile + director's notes + transcript) built from `voice.style`. A plain instruction in front
of the text would be read aloud. The API returns a 24 kHz mono WAV and no timestamps, so word
timings are empty. `synthesize_script` reads all lines in one take and cuts the audio at the
longest pauses, one between each pair of lines.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import soundfile as sf

from anime_short.schemas import AudioResult, VoiceSpec
from anime_short.tts import audio
from anime_short.utils.media import probe_duration

log = logging.getLogger(__name__)

API_BASE_URL = "https://generativelanguage.googleapis.com"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
LINE_SEPARATOR = "\n\n"
FRAME_SECONDS = 0.01
SILENCE_DB = -40.0  # frame RMS below this (dBFS) counts as a pause
# A line's share of the take may differ from its share of the characters by at most this factor;
# beyond that the longest pauses were not the ones between lines (e.g. a pause inside a line).
MAX_SHARE_RATIO = 2.5


class GeminiTTSProvider:
    name = "gemini"
    min_free_vram_gb = 0.0  # network provider, no GPU/model to load

    def __init__(
        self,
        options: dict[str, Any],
        voice: VoiceSpec,
        client: httpx.Client | None = None,
    ) -> None:
        self._model = options.get("model", "gemini-3.8-flash-tts")
        self._voice_name = voice.voice or options.get("voice") or "Enceladus"
        self._style = voice.style
        self._speed = voice.speed
        self._max_retries = int(options.get("max_retries", 3))
        self._retry_backoff = float(options.get("retry_backoff_seconds", 5.0))

        if client is not None:
            self._client = client
        else:
            api_key_env = options.get("api_key_env", "GEMINI_API_KEY")
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise RuntimeError(f"{api_key_env} is not set")
            self._client = httpx.Client(
                base_url=API_BASE_URL,
                headers={"x-goog-api-key": api_key},
                timeout=options.get("timeout_seconds", 300),
            )

    def load(self) -> None:
        pass  # network provider, nothing to load

    def unload(self) -> None:
        self._client.close()

    def synthesize(self, text: str, output_path: Path) -> AudioResult:
        samples, sample_rate = self._read(text)
        return self._write(samples, sample_rate, output_path)

    def synthesize_script(self, texts: list[str], output_paths: list[Path]) -> list[AudioResult]:
        """Read all lines in ONE take, so intonation carries from line to line, then cut the
        audio in the middle of the longest pauses, one between each pair of lines. The model can
        pause inside a line too, so a split whose pieces do not fit the lines' lengths is
        rejected and the script read again (up to `max_retries` times)."""
        for attempt in range(self._max_retries + 1):
            samples, sample_rate = self._read(LINE_SEPARATOR.join(texts))
            try:
                cuts = split_at_pauses(samples, sample_rate, [len(t) for t in texts])
                break
            except ValueError as exc:
                if attempt == self._max_retries:
                    raise RuntimeError(f"gemini tts: {exc}") from exc
                log.warning("gemini tts: %s; reading the script again", exc)
        return [
            self._write(samples[lo:hi], sample_rate, path)
            for lo, hi, path in zip(cuts, cuts[1:], output_paths, strict=False)
        ]

    def _write(self, samples: np.ndarray, sample_rate: int, output_path: Path) -> AudioResult:
        trimmed, _offset = audio.trim_silence(samples, sample_rate)
        audio.write_wav(audio.fade_edges(trimmed, sample_rate), sample_rate, output_path)
        audio.apply_speed(output_path, self._speed)
        return AudioResult(path=output_path, duration=probe_duration(output_path), words=[])

    def _prompt(self, text: str) -> str:
        if not self._style:
            return text
        return (
            "# AUDIO PROFILE: narrator\n"
            f"### DIRECTOR'S NOTES\nStyle: {self._style}\n"
            f"#### TRANSCRIPT\n{text}"
        )

    def _read(self, text: str) -> tuple[np.ndarray, int]:
        payload = {
            "contents": [{"parts": [{"text": self._prompt(text)}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self._voice_name}}
                },
            },
        }
        body = self._request(payload).json()
        candidates = body.get("candidates") or []
        for part in candidates[0].get("content", {}).get("parts", []) if candidates else []:
            if "inlineData" in part:
                data = base64.b64decode(part["inlineData"]["data"])
                samples, sample_rate = sf.read(io.BytesIO(data), dtype="float32")
                return samples, sample_rate
        reason = candidates[0].get("finishReason") if candidates else None
        raise RuntimeError(f"gemini tts returned no audio (reason: {reason or 'unknown'})")

    def _request(self, payload: dict[str, Any]) -> httpx.Response:
        url = f"/v1beta/models/{self._model}:generateContent"
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(url, json=payload)
            except httpx.HTTPError as exc:
                last_error = exc
            else:
                if response.status_code < 400:
                    return response
                last_error = RuntimeError(
                    f"gemini tts {response.status_code}: {response.text[:500]}"
                )
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    raise last_error
            if attempt < self._max_retries:
                delay = self._retry_backoff * (2**attempt)
                log.warning(
                    "gemini tts request failed (attempt %d), retrying in %.1fs", attempt + 1, delay
                )
                time.sleep(delay)
        assert last_error is not None
        raise last_error


def split_at_pauses(samples: np.ndarray, sample_rate: int, line_chars: list[int]) -> list[int]:
    """Sample indices `[0, cut_1, ..., cut_{n-1}, len]` splitting one take of `n` lines in the
    middle of its `n - 1` longest inner pauses. Raises ValueError when there are too few pauses
    or a piece's share of the take is far from its line's share of the characters."""
    frame = int(FRAME_SECONDS * sample_rate)
    n_frames = samples.size // frame
    frames = samples[: n_frames * frame].reshape(n_frames, frame)
    silent = 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-12) < SILENCE_DB

    pauses = []  # (length, first frame, end frame) of each silent run between sounds
    voiced = np.flatnonzero(~silent)
    for a, b in zip(voiced, voiced[1:], strict=False):
        if b - a > 1:
            pauses.append((b - a - 1, a + 1, b))
    needed = len(line_chars) - 1
    if len(pauses) < needed:
        raise ValueError(f"found {len(pauses)} pause(s), need {needed}")
    chosen = sorted(sorted(pauses, reverse=True)[:needed], key=lambda p: p[1])
    cuts = [0, *(((start + end) // 2) * frame for _, start, end in chosen), samples.size]

    total_chars = sum(line_chars)
    for i, chars in enumerate(line_chars):
        share = (cuts[i + 1] - cuts[i]) / samples.size
        ratio = share / (chars / total_chars)
        if not 1 / MAX_SHARE_RATIO <= ratio <= MAX_SHARE_RATIO:
            raise ValueError(f"line {i + 1} got {ratio:.2f}x its share of the take")
    return cuts
