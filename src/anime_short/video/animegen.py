"""GPU video generator (SPEC §7.1): `aidealab/AnimeGen-I2V`, a Wan2.2-I2V-A14B MoE fine-tune,
driven through `diffusers.WanImageToVideoPipeline` with the Wan2.2 Lightning 4-step LoRA, exactly
as shown on the model card (https://huggingface.co/aidealab/AnimeGen-I2V).

VRAM is shared with other services (SPEC §1/§12): fp8 layerwise casting on both transformer
experts plus `enable_model_cpu_offload()` keep the resident footprint small; `video/stage.py` is
still responsible for the `min_free_vram_gb` gate before `load()` is ever called.
"""

from __future__ import annotations

import math
import tempfile
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from anime_short.schemas import CameraMove, VideoResult
from anime_short.video import postprocess

WAN_BASE_REPO = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
LIGHTNING_REPO = "lightx2v/Wan2.2-Lightning"
LIGHTNING_SUBDIR = "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1"
NEGATIVE_PROMPT = "3d, cg, photo, stop, wait"
# cfg=1.0 ignores the negative prompt; without this the model invents gibberish text or images on
# screens and graphics that the motion prompt asks to change (checked on real clips).
STABLE_CONTENT = "any screens, writing and graphics stay exactly the same and do not change"

_CAMERA_PHRASE = {
    CameraMove.static: "camera holds steady",
    CameraMove.push_in: "camera slowly pushes in",
    CameraMove.pull_out: "camera slowly pulls out",
    CameraMove.pan_left: "camera pans left",
    CameraMove.pan_right: "camera pans right",
    CameraMove.tilt_up: "camera tilts up",
    CameraMove.tilt_down: "camera tilts down",
}


class AnimeGenGenerator:
    """`VideoGenerator` for `aidealab/AnimeGen-I2V`."""

    name = "animegen"

    def __init__(
        self,
        animegen_config: dict[str, Any] | None = None,
        width: int = 1080,
        height: int = 1920,
        fps: int = 24,
    ) -> None:
        config = animegen_config or {}
        self.repo = config.get("repo", "aidealab/AnimeGen-I2V")
        self.gen_width = int(config.get("width", 480))
        self.gen_height = int(config.get("height", 832))
        self.steps = int(config.get("steps", 4))
        self.cfg = float(config.get("cfg", 1.0))
        self.native_fps = int(config.get("native_fps", 16))
        self.max_frames = int(config.get("max_frames", 81))
        # Output (project) dims; set by video/stage.py from the ProjectSpec before generating.
        self.width = width
        self.height = height
        self.fps = fps
        self.pipe: Any = None

    def load(self) -> None:
        import torch
        from diffusers import (
            AutoencoderKLWan,
            FlowMatchEulerDiscreteScheduler,
            WanImageToVideoPipeline,
            WanTransformer3DModel,
        )

        transformer = WanTransformer3DModel.from_pretrained(
            self.repo, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        transformer_2 = WanTransformer3DModel.from_pretrained(
            self.repo, subfolder="transformer_2", torch_dtype=torch.bfloat16
        )
        vae = AutoencoderKLWan.from_pretrained(
            WAN_BASE_REPO, subfolder="vae", torch_dtype=torch.float32
        )
        scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)

        pipe = WanImageToVideoPipeline.from_pretrained(
            WAN_BASE_REPO,
            transformer=transformer,
            transformer_2=transformer_2,
            scheduler=scheduler,
            vae=vae,
            torch_dtype=torch.bfloat16,
        )
        pipe.load_lora_weights(
            LIGHTNING_REPO,
            weight_name=f"{LIGHTNING_SUBDIR}/high_noise_model.safetensors",
            adapter_name="high",
        )
        pipe.load_lora_weights(
            LIGHTNING_REPO,
            weight_name=f"{LIGHTNING_SUBDIR}/low_noise_model.safetensors",
            adapter_name="low",
            load_into_transformer_2=True,
        )
        pipe.set_adapters(["high", "low"], adapter_weights=[1.0, 1.0])

        transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
        )
        transformer_2.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()
        self.pipe = pipe

    def unload(self) -> None:
        self.pipe = None

    def _num_frames(self, duration: float) -> int:
        needed = duration * self.native_fps
        n = max(1, math.ceil((needed - 1) / 4))
        frames = 4 * n + 1
        return min(frames, self.max_frames)

    def _resolution(self, image: Image.Image) -> tuple[int, int]:
        """Model-card recipe: snap to the pipeline's mod value, keeping the input aspect ratio,
        within the configured max_area."""
        max_area = self.gen_width * self.gen_height
        aspect_ratio = image.height / image.width
        mod_value = self.pipe.vae_scale_factor_spatial * self.pipe.transformer.config.patch_size[1]
        height = round(math.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(math.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        return max(mod_value, width), max(mod_value, height)

    def generate(
        self,
        image: Path,
        prompt: str,
        camera: CameraMove,
        duration: float,
        output_path: Path,
    ) -> VideoResult:
        import torch

        assert self.pipe is not None, "call load() before generate()"

        source = Image.open(image).convert("RGB")
        width, height = self._resolution(source)
        source = source.resize((width, height))

        text_prompt = ", ".join(
            part
            for part in ("Japanese anime style", _CAMERA_PHRASE[camera], prompt, STABLE_CONTENT)
            if part
        )
        seed = zlib.crc32(str(output_path).encode()) & 0xFFFFFFFF
        rng = torch.Generator(device="cuda").manual_seed(seed)

        result = self.pipe(
            image=source,
            prompt=text_prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=height,
            width=width,
            num_frames=self._num_frames(duration),
            guidance_scale=self.cfg,
            num_inference_steps=self.steps,
            generator=rng,
        )
        frames = result.frames[0]  # list of per-frame arrays (float32 RGB in [0, 1])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".mp4", dir=output_path.parent, delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            # Encode the raw pipeline output ourselves (no OpenCV/imageio dependency needed):
            # `postprocess.encode_raw_frames` already pipes uint8 RGB frames straight to the
            # project's configured ffmpeg, same as still_motion does.
            uint8_frames = (np.clip(f * 255.0, 0, 255).astype(np.uint8) for f in frames)
            postprocess.encode_raw_frames(uint8_frames, tmp_path, width, height, self.native_fps)
            return postprocess.normalize_clip(
                tmp_path, output_path, self.width, self.height, self.fps, duration
            )
        finally:
            tmp_path.unlink(missing_ok=True)
