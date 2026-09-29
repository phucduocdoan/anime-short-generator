from __future__ import annotations

from pathlib import Path
from typing import Protocol

from anime_short.schemas import CameraMove, VideoResult


class VideoGenerator(Protocol):
    """Output must already be normalised: project width x height, project fps, exact duration,
    no audio (see video/postprocess.py)."""

    name: str

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def generate(
        self,
        image: Path,
        prompt: str,
        camera: CameraMove,
        duration: float,
        output_path: Path,
    ) -> VideoResult: ...
