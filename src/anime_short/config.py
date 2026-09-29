"""Load configs/default.yaml + optional spec.yaml overrides into a typed AppConfig (SPEC §4).

Provider sections stay plain dicts: each provider validates its own options.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
TEMPLATES_DIR = REPO_ROOT / "templates"
ASSETS_DIR = REPO_ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"


class LLMConfig(BaseModel):
    base_url: str
    model: str
    api_key_env: str = "LLM_API_KEY"
    temperature: float = 0.8
    max_retries: int = 3
    timeout_seconds: float = 180

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env) or "EMPTY"


class ScriptConfig(BaseModel):
    words_per_second: float = 2.7
    hook_seconds: float = 3
    max_shots_per_scene: int = 3


class TTSConfig(BaseModel):
    provider: str
    elevenlabs: dict[str, Any] = {}
    vieneu: dict[str, Any] = {}


class TimelineConfig(BaseModel):
    scene_gap_seconds: float = 0.15
    min_shot_seconds: float = 1.5
    max_shot_seconds: float = 5.0


class ImageConfig(BaseModel):
    provider: str
    width: int = 864
    height: int = 1536
    providers: dict[str, dict[str, Any]] = {}


class VideoConfig(BaseModel):
    provider: str
    animegen: dict[str, Any] = {}
    still_motion: dict[str, Any] = {}


class SubtitleConfig(BaseModel):
    enabled: bool = True
    style: str = "shorts"
    font_name: str = "Be Vietnam Pro"
    max_words: int = 5
    max_chars: int = 28


class RenderConfig(BaseModel):
    crf: int = 18
    audio_bitrate: str = "192k"
    loudness_lufs: float = -14
    true_peak_db: float = -1
    fade_seconds: float = 0.3


class MusicConfig(BaseModel):
    enabled: bool = False
    path: str | None = None
    volume_db: float = -22


class GPUConfig(BaseModel):
    device: str = "cuda"


class AppConfig(BaseModel):
    workspace_dir: Path = Path("workspace")
    llm: LLMConfig
    script: ScriptConfig = ScriptConfig()
    tts: TTSConfig
    timeline: TimelineConfig = TimelineConfig()
    image: ImageConfig
    video: VideoConfig
    subtitle: SubtitleConfig = SubtitleConfig()
    render: RenderConfig = RenderConfig()
    music: MusicConfig = MusicConfig()
    gpu: GPUConfig = GPUConfig()

    def project_dir(self, project_id: str) -> Path:
        base = self.workspace_dir
        if not base.is_absolute():
            base = REPO_ROOT / base
        return base / project_id


class StyleBible(BaseModel):
    name: str
    positive_tags: str
    positive_natural: str
    negative_tags: str
    negative_natural: str
    camera_preferences: list[str] = []


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(overrides: dict[str, Any] | None = None) -> AppConfig:
    load_dotenv(REPO_ROOT / ".env")
    data = load_yaml(CONFIG_DIR / "default.yaml")
    if overrides:
        data = deep_merge(data, overrides)
    return AppConfig.model_validate(data)


def load_style(name: str) -> StyleBible:
    return StyleBible.model_validate(load_yaml(CONFIG_DIR / "styles" / f"{name}.yaml"))
