"""NetaYume Lumina Image 2.0 generator (SPEC §6): `duongve/NetaYume-Lumina-Image-2.0-Diffusers-v40`
via `diffusers.Lumina2Pipeline`. Apache-2.0, tag-style prompts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from anime_short.schemas import ImageResult, PromptStyle


class NetaYumeGenerator:
    name = "netayume"
    prompt_style = PromptStyle.tags

    def __init__(self, options: dict[str, Any]) -> None:
        self.repo = options.get("repo", "duongve/NetaYume-Lumina-Image-2.0-Diffusers-v40")
        self.steps = options.get("steps", 40)
        self.cfg = options.get("cfg", 5.0)
        self._pipe: Any = None

    def load(self) -> None:
        import torch
        from diffusers import Lumina2Pipeline

        self._pipe = Lumina2Pipeline.from_pretrained(self.repo, torch_dtype=torch.bfloat16)
        self._pipe.to("cuda")

    def unload(self) -> None:
        self._pipe = None

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None,
        width: int,
        height: int,
        seed: int,
        output_path: Path,
    ) -> ImageResult:
        import torch

        assert self._pipe is not None, "call load() first"
        generator = torch.Generator(device="cuda").manual_seed(seed)
        image = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            height=height,
            width=width,
            num_inference_steps=self.steps,
            guidance_scale=self.cfg,
            generator=generator,
        ).images[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return ImageResult(path=output_path, seed=seed, width=width, height=height)
