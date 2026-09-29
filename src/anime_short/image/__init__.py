"""Image generator factory (SPEC §6)."""

from __future__ import annotations

from anime_short.config import AppConfig
from anime_short.image.anima import AnimaGenerator
from anime_short.image.base import ImageGenerator
from anime_short.image.netayume import NetaYumeGenerator
from anime_short.image.zanime import ZAnimeGenerator

_GENERATORS = {
    "anima": AnimaGenerator,
    "netayume": NetaYumeGenerator,
    "zanime": ZAnimeGenerator,
}


def get_image_generator(config: AppConfig, name: str | None = None) -> ImageGenerator:
    name = name or config.image.provider
    try:
        cls = _GENERATORS[name]
    except KeyError:
        raise ValueError(f"unknown image provider: {name!r}") from None
    options = config.image.providers.get(name, {})
    return cls(options)
