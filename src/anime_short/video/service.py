from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from anime_short.schemas import CameraMove, VideoResult


class AnimeGenServiceGenerator:
    """VideoGenerator-compatible client for a resident local AnimeGen service."""

    name = "animegen_service"

    def __init__(
        self,
        service_url: str,
        timeout_seconds: float,
        width: int = 1080,
        height: int = 1920,
        fps: int = 24,
    ) -> None:
        self.service_url = service_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.width = width
        self.height = height
        self.fps = fps

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def generate(
        self,
        image: Path,
        prompt: str,
        camera: CameraMove,
        duration: float,
        output_path: Path,
    ) -> VideoResult:
        payload: dict[str, Any] = {
            "image": str(image),
            "prompt": prompt,
            "camera": camera.value,
            "duration": duration,
            "output_path": str(output_path),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }
        timeout = httpx.Timeout(self.timeout_seconds, connect=10.0)
        response = httpx.post(f"{self.service_url}/generate", json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return VideoResult(
            path=Path(data["path"]),
            duration=float(data["duration"]),
            fps=int(data["fps"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )


def health(service_url: str, timeout_seconds: float = 5.0) -> dict[str, Any]:
    response = httpx.get(f"{service_url.rstrip('/')}/health", timeout=timeout_seconds)
    response.raise_for_status()
    return dict(response.json())
