"""Anima Turbo image generator (SPEC §6): circlestone-labs Anima, Turbo weights, CFG 1, ~10 steps.

`circlestone-labs/Anima` ships the Turbo weights only as a raw ComfyUI-style safetensors file
inside a repo with every other variant too (~39 GB). Instead we load
`vladmandic/Anima-1.0-Turbo`, a community Diffusers conversion of exactly that checkpoint (Turbo
LoRA pre-merged, ~5.6 GB) that bundles a custom pipeline (`AnimaTextToImagePipeline`: Cosmos
Predict2 backbone + Qwen3 text encoder + LLM adapter cross-attending T5 token embeddings). This
avoids both downloading the other Anima variants and re-implementing the LLM-adapter plumbing.
Prompting is tag-style (Danbooru convention); the model card recommends the
"masterpiece, best quality, score_7, safe, " prefix already used in the style bibles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from anime_short.schemas import ImageResult, PromptStyle


class AnimaGenerator:
    name = "anima"
    prompt_style = PromptStyle.tags
    supports_reference = False

    def __init__(self, options: dict[str, Any]) -> None:
        self.repo = options.get("repo", "vladmandic/Anima-1.0-Turbo")
        self.steps = options.get("steps", 10)
        self.cfg = options.get("cfg", 1.0)
        self._pipe: Any = None

    def load(self) -> None:
        import torch
        from diffusers import DiffusionPipeline

        self._pipe = DiffusionPipeline.from_pretrained(
            self.repo,
            custom_pipeline="pipeline",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
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
        reference_image: Path | None = None,  # unsupported, ignored
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
