"""Local VieNeu-TTS provider (SPEC §8): https://github.com/pnnbao97/VieNeu-TTS
(Apache-2.0), used for the local end-to-end run since no ElevenLabs key is present
on this machine.

`Vieneu(mode="v3turbo")` auto-selects its backend: PyTorch on GPU, torch-free ONNX
Runtime on CPU. On GPU it reuses whatever torch/transformers are already installed
in the environment (no version pin), which is why the package is installed here
without its `[cuda]` extra — that extra pins `torch==2.8.0`, which would fight the
project's shared torch 2.14 install.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from anime_short.schemas import AudioResult, VoiceSpec
from anime_short.tts import audio
from anime_short.utils.media import probe_duration

log = logging.getLogger(__name__)

# Male, Northern accent, "kể chuyện" (storytelling) preset — a calm narrator voice
# and one of the editors'-pick presets. Override via `voice.voice` / `tts.vieneu.voice`.
DEFAULT_VOICE = "Thiện Minh"


class VieNeuProvider:
    name = "vieneu"

    def __init__(self, options: dict[str, Any], voice: VoiceSpec) -> None:
        self._device = options.get("device", "cuda")
        self._voice_name = voice.voice or options.get("voice") or DEFAULT_VOICE
        self._speed = voice.speed
        self.min_free_vram_gb = float(options.get("min_free_vram_gb", 1.0))
        self._tts: Any = None

    def load(self) -> None:
        from vieneu import Vieneu

        self._tts = Vieneu(mode="v3turbo", device=self._device)

    def unload(self) -> None:
        self._tts = None

    def synthesize(self, text: str, output_path: Path) -> AudioResult:
        if self._tts is None:
            raise RuntimeError("VieNeuProvider.load() must run before synthesize()")

        wav = self._tts.infer(text, voice=self._voice_name)
        sample_rate = self._tts.sample_rate
        trimmed, _offset = audio.trim_silence(wav, sample_rate)
        audio.write_wav(trimmed, sample_rate, output_path)
        if self._speed != 1.0:
            audio.apply_speed(output_path, self._speed)
        return AudioResult(path=output_path, duration=probe_duration(output_path), words=[])
