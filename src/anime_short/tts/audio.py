"""Shared audio post-processing for TTS providers (SPEC §8): decode to float32,
trim silence, fade edges, adjust speed, write 48 kHz mono WAV, shift word timings. Pure
functions (no model, no network) so they are cheap to unit test directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from anime_short.schemas import WordTiming
from anime_short.utils.media import run_ffmpeg

TARGET_SAMPLE_RATE = 48_000


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """Decode little-endian 16-bit PCM bytes into float32 samples in [-1, 1]."""
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def trim_silence(
    samples: np.ndarray,
    sample_rate: int,
    max_edge_silence: float = 0.15,
    threshold_db: float = -40.0,
) -> tuple[np.ndarray, float]:
    """Trim leading/trailing silence down to at most `max_edge_silence` seconds.

    Returns `(trimmed_samples, seconds_removed_from_the_start)`; callers with word
    timings relative to the original (untrimmed) clip must subtract that offset.
    Audio that never rises above the threshold (or is empty) is returned unchanged.
    """
    if samples.size == 0:
        return samples, 0.0
    threshold = 10 ** (threshold_db / 20)
    nonsilent = np.flatnonzero(np.abs(samples) > threshold)
    if nonsilent.size == 0:
        return samples, 0.0
    pad = int(max_edge_silence * sample_rate)
    start = max(0, int(nonsilent[0]) - pad)
    end = min(samples.size, int(nonsilent[-1]) + pad + 1)
    return samples[start:end], start / sample_rate


def fade_edges(samples: np.ndarray, sample_rate: int, seconds: float = 0.01) -> np.ndarray:
    """Short linear fade-in/out, so a clip that starts or stops while sound is still decaying
    (a cut between lines, or a provider that stops on the last word) does not click."""
    n = min(int(seconds * sample_rate), samples.size // 2)
    if n == 0:
        return samples
    faded = samples.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=samples.dtype)
    faded[:n] *= ramp
    faded[-n:] *= ramp[::-1]
    return faded


def write_wav(samples: np.ndarray, sample_rate: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), samples, sample_rate, subtype="PCM_16")


def shift_words(words: list[WordTiming], offset: float, max_duration: float) -> list[WordTiming]:
    """Shift word timings by `-offset` (silence trimmed off the start) and clip to
    `[0, max_duration]`, dropping words that land entirely outside the clip."""
    shifted = []
    for w in words:
        start, end = w.start - offset, w.end - offset
        if end <= 0 or start >= max_duration:
            continue
        shifted.append(WordTiming(word=w.word, start=max(0.0, start), end=min(max_duration, end)))
    return shifted


def apply_speed(path: Path, speed: float) -> None:
    """Time-stretch the WAV at `path` in place by `speed` (>1 = faster) via ffmpeg's
    `atempo`, which only accepts factors in [0.5, 2.0] per instance — chained here
    for values outside that range."""
    if speed == 1.0:
        return
    factors = []
    remaining = speed
    while remaining < 0.5 or remaining > 2.0:
        step = 2.0 if remaining > 2.0 else 0.5
        factors.append(step)
        remaining /= step
    factors.append(remaining)
    filter_chain = ",".join(f"atempo={f:.6f}" for f in factors)
    tmp_path = path.with_suffix(".tmp.wav")
    run_ffmpeg(["-i", str(path), "-filter:a", filter_chain, str(tmp_path)])
    tmp_path.replace(path)
