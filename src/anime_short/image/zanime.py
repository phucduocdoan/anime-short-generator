"""Z-Anime distill-8-step generator (SPEC §6): `SeeSee21/Z-Anime` via `diffusers.ZImagePipeline`.
Apache-2.0, natural-language prompts.

The repo's `diffusers/` subfolder is drop-in `from_pretrained`-able but only contains the **Base**
transformer (that's what its own README example loads). The distilled 8-step weights only exist as
a ComfyUI-style single file (`diffusion_models/z-anime-distill-8step-bf16.safetensors`) with
original (non-Diffusers) key names. Diffusers ships an official conversion for exactly this
architecture (`convert_z_image_transformer_checkpoint_to_diffusers`, registered for
`ZImageTransformer2DModel.from_single_file`), so we load the transformer from that single file and
the rest of the pipeline (scheduler/tokenizer/text encoder/vae) from the repo's `diffusers/`
subfolder, passing the transformer in to skip downloading the (unwanted) Base transformer weights.

The repo's `diffusers/text_encoder/model.safetensors` is itself broken: it's FP8-quantized
(per-layer `*.scale` tensors), but a plain `Qwen3ForCausalLM.from_pretrained(..., torch_dtype=bf16)`
silently drops those scale tensors instead of dequantizing, which corrupts the weights and produces
prompt-unrelated garbage images. The repo separately ships a clean bf16 checkpoint of the same text
encoder as a ComfyUI-style single file (`text_encoder/qwen_3_4b-bf16.safetensors`, verified
key-for-key against `Qwen3ForCausalLM`'s own `state_dict()`, modulo the tied `lm_head.weight`), so
we build the text encoder from the `diffusers/text_encoder` config and load that file's weights
into it instead of using `from_pretrained` on the broken one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from anime_short.schemas import ImageResult, PromptStyle


class ZAnimeGenerator:
    name = "zanime"
    prompt_style = PromptStyle.natural

    def __init__(self, options: dict[str, Any]) -> None:
        self.repo = options.get("repo", "SeeSee21/Z-Anime")
        self.transformer_file = options.get(
            "transformer_file", "diffusion_models/z-anime-distill-8step-bf16.safetensors"
        )
        self.steps = options.get("steps", 8)
        self.cfg = options.get("cfg", 1.0)
        self._pipe: Any = None

    def load(self) -> None:
        import torch
        from diffusers import ZImagePipeline, ZImageTransformer2DModel
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import AutoConfig, Qwen3ForCausalLM

        transformer = ZImageTransformer2DModel.from_single_file(
            f"https://huggingface.co/{self.repo}/blob/main/{self.transformer_file}",
            config=self.repo,
            subfolder="diffusers/transformer",
            torch_dtype=torch.bfloat16,
        )
        # `DiffusionPipeline.from_pretrained` doesn't honor a top-level `subfolder` (unlike
        # model-level `from_pretrained`), so resolve the "diffusers/" subfolder on disk
        # ourselves. The transformer is passed in directly (see above), and the (broken, FP8)
        # `diffusers/text_encoder/model.safetensors` is skipped in favor of the clean bf16 file
        # loaded below, so neither is downloaded here.
        local_dir = snapshot_download(
            self.repo,
            allow_patterns=[
                "diffusers/model_index.json",
                "diffusers/scheduler/*",
                "diffusers/text_encoder/config.json",
                "diffusers/tokenizer/*",
                "diffusers/vae/*",
                "text_encoder/qwen_3_4b-bf16.safetensors",
            ],
        )
        text_encoder_config = AutoConfig.from_pretrained(f"{local_dir}/diffusers/text_encoder")
        text_encoder = Qwen3ForCausalLM(text_encoder_config)
        state_dict = load_file(f"{local_dir}/text_encoder/qwen_3_4b-bf16.safetensors")
        missing, unexpected = text_encoder.load_state_dict(state_dict, strict=False, assign=True)
        assert not unexpected, f"unexpected text encoder keys: {unexpected}"
        assert missing == ["lm_head.weight"], f"missing text encoder keys: {missing}"
        text_encoder.tie_weights()  # re-link lm_head.weight to the just-loaded embed_tokens.weight

        self._pipe = ZImagePipeline.from_pretrained(
            f"{local_dir}/diffusers",
            transformer=transformer,
            text_encoder=text_encoder,
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
