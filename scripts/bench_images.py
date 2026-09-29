#!/usr/bin/env python
"""Benchmark the 3 image providers on the same prompts/seeds (SPEC §6, §14).

Loads one provider at a time (GPU is shared with other services), times each of 4 fixed shots
(same seed across providers), records peak VRAM, and builds a contact sheet for visual comparison.

Usage (inside the project venv, on the GPU host):
    flock <gpu.lock> python scripts/bench_images.py

Output: `workspace/bench_images/<provider>/<shot_id>.png` and
`workspace/bench_images/contact_sheet.png`.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from anime_short.config import REPO_ROOT, load_config, load_style
from anime_short.image import get_image_generator
from anime_short.image.prompt_builder import build as build_prompt
from anime_short.schemas import CameraMove, Character, Shot, VisualSpec
from anime_short.utils.gpu import free_vram_gb, release_cuda_memory

# Character bible entry, copied from examples/tin_nhan_10_nam.yaml.
MINH = Character(
    name="Minh",
    age=22,
    tags="1boy, solo, young man, short black hair, messy hair, brown eyes, slim",
    description=(
        "a slim 22-year-old Vietnamese young man with short, slightly messy black hair"
        " and brown eyes"
    ),
    outfit_tags="dark navy oversized hoodie, black pants",
    outfit_description="wearing a dark navy oversized hoodie and black trousers",
)

SHOTS = [
    Shot(
        id="closeup_night_phone",
        visual=VisualSpec(
            subject="Minh's face lit only by his phone screen",
            action="staring at a text message, eyes widening",
            setting="a dark bedroom at night",
            lighting="cold blue phone screen glow, otherwise near-black shadows",
            shot_type="extreme close-up",
            mood="shocked, mysterious",
        ),
        characters=["minh"],
        camera=CameraMove.push_in,
    ),
    Shot(
        id="fullbody_train_platform",
        visual=VisualSpec(
            subject="Minh standing alone",
            action="waiting, hands in hoodie pockets",
            setting="an empty train platform late at night",
            lighting="sparse overhead fluorescent lights, long shadows",
            shot_type="full body shot, wide",
            mood="lonely, contemplative",
        ),
        characters=["minh"],
        camera=CameraMove.static,
    ),
    Shot(
        id="scenery_no_humans",
        visual=VisualSpec(
            subject="a quiet Hanoi alley with hanging string lights and parked bicycles",
            setting="early morning, mist over the street",
            lighting="soft golden sunrise light",
            shot_type="establishing shot",
            mood="calm, nostalgic",
        ),
        characters=[],
        camera=CameraMove.pan_right,
    ),
    Shot(
        id="closeup_crying",
        visual=VisualSpec(
            subject="Minh's face",
            action="tears rolling down his cheeks, holding back a smile",
            setting="sitting on the floor of his room",
            lighting="warm lamp light from one side, soft shadows",
            shot_type="close-up",
            mood="emotional, bittersweet",
        ),
        characters=["minh"],
        camera=CameraMove.push_in,
    ),
]

PROVIDERS = ["anima", "netayume", "zanime"]
SEED = 12345


def main() -> None:
    config = load_config()
    style = load_style("cinematic_anime")
    out_dir = REPO_ROOT / "workspace" / "bench_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    characters = {"minh": MINH}
    results: dict[str, dict[str, float]] = {}
    thumbnails: dict[tuple[str, str], Path] = {}

    for provider_name in PROVIDERS:
        generator = get_image_generator(config, provider_name)
        provider_dir = out_dir / provider_name
        provider_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {provider_name}: loading ===")
        free_before = free_vram_gb()
        torch.cuda.reset_peak_memory_stats()
        generator.load()

        times = []
        for shot in SHOTS:
            prompt, negative = build_prompt(shot, characters, style, generator.prompt_style)
            output_path = provider_dir / f"{shot.id}.png"
            start = time.perf_counter()
            generator.generate(
                prompt=prompt,
                negative_prompt=negative,
                width=config.image.width,
                height=config.image.height,
                seed=SEED,
                output_path=output_path,
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            thumbnails[(provider_name, shot.id)] = output_path
            print(f"  {shot.id}: {elapsed:.1f}s -> {output_path}")

        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        generator.unload()
        release_cuda_memory()

        results[provider_name] = {
            "sec_per_image": sum(times) / len(times),
            "peak_vram_gb": peak_gb,
            "free_vram_before_gb": free_before,
        }

    print("\n=== results ===")
    for provider_name, stats in results.items():
        print(
            f"{provider_name}: {stats['sec_per_image']:.1f} s/image, "
            f"peak VRAM {stats['peak_vram_gb']:.1f} GB "
            f"(free before load: {stats['free_vram_before_gb']:.1f} GB)"
        )

    contact_sheet_path = out_dir / "contact_sheet.png"
    _build_contact_sheet(thumbnails, PROVIDERS, SHOTS, contact_sheet_path)
    print(f"\ncontact sheet: {contact_sheet_path}")


def _build_contact_sheet(
    thumbnails: dict[tuple[str, str], Path],
    providers: list[str],
    shots: list[Shot],
    output_path: Path,
) -> None:
    thumb_w, thumb_h = 216, 384  # 864x1536 downscaled 4x
    label_h = 20
    cell_w, cell_h = thumb_w, thumb_h + label_h
    sheet = Image.new("RGB", (cell_w * len(providers), cell_h * len(shots)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for row, shot in enumerate(shots):
        for col, provider_name in enumerate(providers):
            path = thumbnails.get((provider_name, shot.id))
            x, y = col * cell_w, row * cell_h
            if path is not None and path.exists():
                thumb = Image.open(path).convert("RGB").resize((thumb_w, thumb_h))
                sheet.paste(thumb, (x, y))
            draw.text(
                (x + 4, y + thumb_h + 2), f"{provider_name}/{shot.id}", fill="black", font=font
            )

    sheet.save(output_path)


if __name__ == "__main__":
    main()
