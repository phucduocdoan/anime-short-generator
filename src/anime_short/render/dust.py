"""Old-film dust (SPEC §11): black specks of random size at random spots, new ones every frame,
like CapCut's "dust & scratches" overlay. Written as a gray video (white background) that the
final encode multiplies onto the picture's luma."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from anime_short.utils.media import FFmpegError, check_ffmpeg


def dust_frame(rng: np.random.Generator, width: int, height: int, density: float) -> Image.Image:
    """One gray frame: on average `density` dark specks, mostly small, some up to ~10 px (at a
    1080 px wide frame), slightly soft-edged."""
    scale = width / 1080
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    for _ in range(rng.poisson(density)):
        x, y = rng.uniform(0, width), rng.uniform(0, height)
        r = scale * (rng.uniform(1.5, 4) if rng.random() < 0.75 else rng.uniform(4, 10))
        rx, ry = r * rng.uniform(0.6, 1.4), r * rng.uniform(0.6, 1.4)
        draw.ellipse([x - rx, y - ry, x + rx, y + ry], fill=int(rng.uniform(0, 90)))
    return img.filter(ImageFilter.GaussianBlur(0.9 * scale))


def write_dust_video(
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    duration: float,
    density: float,
    seed: int = 0,
) -> Path:
    """Render `duration` seconds of dust frames to a lossless gray video."""
    cmd = [
        check_ffmpeg(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "ffv1",
        str(output_path),
    ]
    rng = np.random.default_rng(seed)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for _ in range(int(np.ceil(duration * fps)) + 1):
            proc.stdin.write(dust_frame(rng, width, height, density).tobytes())
    except BrokenPipeError:
        pass  # ffmpeg exited early; its stderr says why
    _, stderr = proc.communicate()
    if proc.returncode != 0:
        raise FFmpegError(f"{' '.join(cmd)}\n{stderr.decode().strip()}")
    return output_path
