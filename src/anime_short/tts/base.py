from __future__ import annotations

from pathlib import Path
from typing import Protocol

from anime_short.schemas import AudioResult


class TTSProvider(Protocol):
    name: str

    def synthesize(self, text: str, output_path: Path) -> AudioResult:
        """Write 48 kHz mono WAV to output_path; return real duration and word timings (may be [])."""
        ...
