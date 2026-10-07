"""GPU video generator (SPEC §7.1): `aidealab/AnimeGen-I2V`, a Wan2.2-I2V-A14B MoE fine-tune,
driven through `diffusers.WanImageToVideoPipeline` with the Wan2.2 Lightning 4-step LoRA, exactly
as shown on the model card (https://huggingface.co/aidealab/AnimeGen-I2V).

VRAM is shared with other services (SPEC §1/§12): fp8 layerwise casting on both transformer
experts plus `enable_model_cpu_offload()` keep the resident footprint small; `video/stage.py` is
still responsible for the `min_free_vram_gb` gate before `load()` is ever called.

The model is always asked for a static camera. Asked to move the camera, it redraws the scene and
invents events that are not in the keyframe; held steady, it only adds the ambient motion from the
prompt. The shot's camera move is then applied in post, with the same path as still_motion.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from anime_short.schemas import CameraMove, VideoResult
from anime_short.video import postprocess
from anime_short.video.still_motion import camera_boxes

WAN_BASE_REPO = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
LIGHTNING_REPO = "lightx2v/Wan2.2-Lightning"
LIGHTNING_SUBDIR = "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1"
NEGATIVE_PROMPT = "3d, cg, photo, stop, wait"
# cfg=1.0 ignores the negative prompt; without this the model invents gibberish text or images on
# screens and graphics that the motion prompt asks to change (checked on real clips).
STABLE_CONTENT = "any screens, writing and graphics stay exactly the same and do not change"

STATIC_CAMERA = "camera holds steady"


def camera_move_frames(
    frames: list[np.ndarray],
    camera: CameraMove,
    width: int,
    height: int,
    fps: int,
    native_fps: int,
    n_frames: int,
) -> Iterator[np.ndarray]:
    """Yield `n_frames` `width`x`height` frames at `fps` from the generated `frames` (uint8 RGB at
    `native_fps`), cover-cropped to the output aspect with the shot's camera move. Each output
    frame is one Lanczos resample of a float crop box, so the move is sub-pixel smooth. A
    generated clip shorter than the shot is slowed down to fill it."""
    src_h, src_w = frames[0].shape[:2]
    scale = max(width / src_w, height / src_h)
    crop_w, crop_h = width / scale, height / scale
    off_x, off_y = (src_w - crop_w) / 2, (src_h - crop_h) / 2
    boxes = camera_boxes(crop_w, crop_h, camera, n_frames)

    src_span = (len(frames) - 1) / native_fps
    out_span = (n_frames - 1) / fps
    speed = min(1.0, src_span / out_span) if out_span > 0 else 1.0
    images = [Image.fromarray(frame) for frame in frames]
    for i, (left, top, right, bottom) in enumerate(boxes):
        j = min(len(images) - 1, round(i / fps * native_fps * speed))
        box = (left + off_x, top + off_y, right + off_x, bottom + off_y)
        yield np.asarray(images[j].resize((width, height), Image.LANCZOS, box=box))


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
        self.offload_strategy = str(config.get("offload_strategy", "none"))
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
        if self.offload_strategy == "model_cpu":
            pipe.enable_model_cpu_offload()
        elif self.offload_strategy == "none":
            pipe.to("cuda")
        else:
            raise ValueError(f"unknown animegen offload_strategy: {self.offload_strategy}")
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
            part for part in ("Japanese anime style", STATIC_CAMERA, prompt, STABLE_CONTENT) if part
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

        uint8_frames = [np.clip(f * 255.0, 0, 255).astype(np.uint8) for f in frames]
        n_frames = max(1, postprocess.target_frame_count(duration, self.fps))
        out_frames = camera_move_frames(
            uint8_frames, camera, self.width, self.height, self.fps, self.native_fps, n_frames
        )
        return postprocess.encode_raw_frames(
            out_frames, output_path, self.width, self.height, self.fps
        )
