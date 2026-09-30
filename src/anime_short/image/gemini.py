"""Gemini image generator (SPEC §6): Google's `generateContent` API with an image model
(default `gemini-3-pro-image`), natural-language prompts.

This is the only provider that uses a style's `reference_image`: the image is attached to every
request and the prompt tells the model to copy only its drawing style. Gemini has no negative
prompt, so `negative_prompt` is ignored. The API returns (JPEG) images at its own resolution
(2K 9:16 is 1536×2752, not exactly 9:16), so the result is center-cropped to the requested
aspect ratio, resized to `width`×`height` and saved as PNG.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageOps

from anime_short.schemas import ImageResult, PromptStyle

log = logging.getLogger(__name__)

API_BASE_URL = "https://generativelanguage.googleapis.com"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

REFERENCE_INSTRUCTION = (
    "Use the attached image only as a style reference: match its drawing style, line work, "
    "color palette and texture. Do not copy its characters, composition or any text from it. "
    "Draw this frame: "
)


class GeminiGenerator:
    name = "gemini"
    prompt_style = PromptStyle.natural
    supports_reference = True

    def __init__(self, options: dict[str, Any], client: httpx.Client | None = None) -> None:
        self.model = options.get("model", "gemini-3-pro-image")
        self.image_size = options.get("image_size", "2K")
        self._max_retries = int(options.get("max_retries", 3))
        self._retry_backoff = float(options.get("retry_backoff_seconds", 5.0))

        if client is not None:
            self._client = client
        else:
            api_key_env = options.get("api_key_env", "GEMINI_API_KEY")
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise RuntimeError(f"{api_key_env} is not set")
            self._client = httpx.Client(
                base_url=API_BASE_URL,
                headers={"x-goog-api-key": api_key},
                timeout=options.get("timeout_seconds", 300),
            )

    def load(self) -> None:
        pass  # network provider, nothing to load

    def unload(self) -> None:
        self._client.close()

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None,
        width: int,
        height: int,
        seed: int,
        output_path: Path,
        reference_image: Path | None = None,
    ) -> ImageResult:
        parts: list[dict[str, Any]] = []
        if reference_image is not None:
            with Image.open(reference_image) as ref:
                mime = Image.MIME[ref.format]
            data = base64.b64encode(reference_image.read_bytes()).decode()
            parts.append({"inline_data": {"mime_type": mime, "data": data}})
            prompt = REFERENCE_INSTRUCTION + prompt
        parts.append({"text": prompt})

        divisor = math.gcd(width, height)
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "seed": seed % 2**31,  # the API field is int32; deterministic seeds are uint32
                "imageConfig": {
                    "aspectRatio": f"{width // divisor}:{height // divisor}",
                    "imageSize": self.image_size,
                },
            },
        }
        body = self._request(payload).json()

        image_bytes = _extract_image(body)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = ImageOps.fit(image, (width, height), Image.Resampling.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="PNG")
        return ImageResult(path=output_path, seed=seed, width=width, height=height)

    def _request(self, payload: dict[str, Any]) -> httpx.Response:
        url = f"/v1beta/models/{self.model}:generateContent"
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(url, json=payload)
            except httpx.HTTPError as exc:
                last_error = exc
            else:
                if response.status_code < 400:
                    return response
                last_error = RuntimeError(f"gemini {response.status_code}: {response.text[:500]}")
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    raise last_error
            if attempt < self._max_retries:
                delay = self._retry_backoff * (2**attempt)
                log.warning(
                    "gemini request failed (attempt %d), retrying in %.1fs", attempt + 1, delay
                )
                time.sleep(delay)
        assert last_error is not None
        raise last_error


def _extract_image(body: dict[str, Any]) -> bytes:
    """The first inline image of the first candidate; a clear error when there is none
    (e.g. the prompt or the output was blocked)."""
    candidates = body.get("candidates") or []
    if candidates:
        for part in candidates[0].get("content", {}).get("parts", []):
            if "inlineData" in part:
                return base64.b64decode(part["inlineData"]["data"])
    reason = (
        candidates[0].get("finishReason")
        if candidates
        else body.get("promptFeedback", {}).get("blockReason")
    )
    raise RuntimeError(f"gemini returned no image (reason: {reason or 'unknown'})")
