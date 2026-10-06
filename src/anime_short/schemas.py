"""Data model shared by every stage. `project.json` is a serialised `ProjectSpec` (SPEC §3).

All artifact paths are stored relative to the project directory.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class StageStatus(StrEnum):
    pending = "pending"
    done = "done"
    failed = "failed"


STAGES: tuple[str, ...] = (
    "script",
    "plan",
    "tts",
    "timeline",
    "images",
    "videos",
    "subtitles",
    "music",
    "render",
)


class CameraMove(StrEnum):
    static = "static"
    push_in = "push_in"
    pull_out = "pull_out"
    pan_left = "pan_left"
    pan_right = "pan_right"
    tilt_up = "tilt_up"
    tilt_down = "tilt_down"


class VideoMode(StrEnum):
    i2v = "i2v"
    still_motion = "still_motion"


class PromptStyle(StrEnum):
    tags = "tags"
    natural = "natural"


# --- inputs ---------------------------------------------------------------------------------


class Character(BaseModel):
    """Character bible entry. The prompt builder injects this; the LLM never writes appearance."""

    name: str
    age: int | None = None
    tags: str = Field(description="Danbooru-style tags, e.g. '1boy, short black hair, brown eyes'")
    description: str = Field(description="Natural English description of appearance")
    outfit_tags: str = ""
    outfit_description: str = ""


class ContentSpec(BaseModel):
    topic: str
    # story: a narrated plot with recurring characters; quotes: a spoken monologue of short
    # lines about love/life, one line per scene over scenery (no character bible).
    format: Literal["story", "quotes"] = "story"
    duration_seconds: float = 45
    tone: list[str] = Field(default_factory=list)
    audience: str | None = None


class VoiceSpec(BaseModel):
    provider: str = "vieneu"
    voice: str | None = None
    speed: float = 1.0
    style: str | None = None  # delivery notes for providers steered by a prompt (gemini)


# --- enrichment -----------------------------------------------------------------------------


class WordTiming(BaseModel):
    word: str
    start: float  # seconds, relative to the scene audio start
    end: float


class AudioInfo(BaseModel):
    path: str
    duration: float
    words: list[WordTiming] = Field(default_factory=list)
    provider: str | None = None


class VisualSpec(BaseModel):
    """One frame, in English. No character appearance here — only ids in `Shot.characters`."""

    subject: str
    action: str = ""
    setting: str = ""
    lighting: str = ""
    shot_type: str = ""  # e.g. "close-up", "medium shot", "establishing shot"
    mood: str = ""
    extra: str = ""
    text: str = ""  # exact English words visible in the frame (a message, a sign); usually empty


class ImageInfo(BaseModel):
    path: str
    seed: int
    provider: str
    prompt: str
    negative_prompt: str | None = None


class VideoInfo(BaseModel):
    path: str
    mode: VideoMode
    provider: str


class Shot(BaseModel):
    id: str  # "scene_01_a"
    visual: VisualSpec
    motion_prompt: str = ""
    camera: CameraMove = CameraMove.push_in
    characters: list[str] = Field(default_factory=list)  # ids into ProjectSpec.characters
    weight: float = 1.0  # relative duration inside the scene
    start: float | None = None  # absolute seconds in the final video (timeline)
    duration: float | None = None  # seconds (timeline)
    seed: int | None = None
    image: ImageInfo | None = None
    video: VideoInfo | None = None


class Scene(BaseModel):
    id: str  # "scene_01"
    narration: str
    visual_description: str = ""
    emotion: str = ""
    importance: str = "normal"
    audio: AudioInfo | None = None
    start: float | None = None  # absolute seconds in the final video (timeline)
    shots: list[Shot] = Field(default_factory=list)


class ProjectSpec(BaseModel):
    id: str
    title: str
    language: str = "vi"
    width: int = 1080
    height: int = 1920
    fps: int = 24
    style: str = "cinematic_anime"
    content: ContentSpec
    voice: VoiceSpec = Field(default_factory=VoiceSpec)
    characters: dict[str, Character] = Field(default_factory=dict)
    hook: str | None = None
    ending: str | None = None
    scenes: list[Scene] = Field(default_factory=list)
    stages: dict[str, StageStatus] = Field(
        default_factory=lambda: {s: StageStatus.pending for s in STAGES}
    )
    music: str | None = None  # relative path to the generated background music (music stage)
    output: str | None = None  # relative path to final.mp4
    youtube_uploads: list[dict[str, str | None]] = Field(default_factory=list)

    def all_shots(self) -> list[Shot]:
        return [shot for scene in self.scenes for shot in scene.shots]


# --- provider results -----------------------------------------------------------------------


class AudioResult(BaseModel):
    path: Path
    duration: float
    words: list[WordTiming] = Field(default_factory=list)


class ImageResult(BaseModel):
    path: Path
    seed: int
    width: int
    height: int


class VideoResult(BaseModel):
    path: Path
    duration: float
    fps: int
    width: int
    height: int
