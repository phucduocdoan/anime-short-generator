from __future__ import annotations

from pathlib import Path
from typing import Protocol

from anime_short.schemas import ImageResult, PromptStyle


class ImageGenerator(Protocol):
    name: str
    prompt_style: PromptStyle
    supports_reference: bool  # whether `generate` uses `reference_image` (a style image)

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None,
        width: int,
        height: int,
        seed: int,
        output_path: Path,
        reference_image: Path | None = None,
    ) -> ImageResult: ...
