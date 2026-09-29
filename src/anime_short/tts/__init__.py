"""TTS provider factory (SPEC §8)."""

from __future__ import annotations

from anime_short.config import AppConfig
from anime_short.schemas import VoiceSpec
from anime_short.tts.base import TTSProvider


def get_tts(config: AppConfig, voice: VoiceSpec) -> TTSProvider:
    """Build the provider named by `voice.provider` (overrides `config.tts.provider`,
    though the two agree unless the project spec picked a different one)."""
    provider = voice.provider or config.tts.provider
    if provider == "elevenlabs":
        from anime_short.tts.elevenlabs import ElevenLabsProvider

        return ElevenLabsProvider(config.tts.elevenlabs, voice)
    if provider == "vieneu":
        from anime_short.tts.vieneu import VieNeuProvider

        return VieNeuProvider(config.tts.vieneu, voice)
    raise ValueError(f"unknown tts provider: {provider!r}")
