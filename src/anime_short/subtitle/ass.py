"""Build the burned-subtitle ASS file (SPEC §10). Two styles (`subtitle.style`):

- "shorts": Be Vietnam Pro, large, white with a thick dark outline, centred around ~70% of the
  frame height. Word timings (when present) drive per-word karaoke highlighting.
- "quote" (quotes videos): white italic handwriting mid-frame over the sky with a soft
  blurred halo, whole lines faded in and held until the next one (no karaoke).

Without word timings, chunks are timed proportionally by character count within the scene's
audio duration.
"""

from __future__ import annotations

from anime_short.config import AppConfig
from anime_short.schemas import ProjectSpec, Scene, WordTiming
from anime_short.subtitle.chunks import chunk_text, chunk_words

# Style constants, sized for the project's nominal 1080x1920 frame and scaled to the actual
# project resolution. Not exposed as config: these are rendering details, not pipeline knobs.
FONT_SIZE_AT_1920 = 80
OUTLINE_WIDTH_AT_1920 = 5.0
SHADOW_AT_1920 = 2.0
MARGIN_H_AT_1080 = 60
VERTICAL_FRACTION = 0.70  # subtitle centre, as a fraction of height from the top (safe area)

PRIMARY_COLOR = "&H0000D7FF"  # ASS is &HAABBGGRR: gold - the already-highlighted word
SECONDARY_COLOR = "&H00FFFFFF"  # white - not yet highlighted
OUTLINE_COLOR = "&H00101010"
BACK_COLOR = "&H80000000"
MIN_KARAOKE_CS = 3  # minimum \k duration (centiseconds) so very short words stay visible

QUOTE_FONT_SIZE_AT_1920 = 96
QUOTE_HALO_AT_1920 = 4.0  # outline width, blurred into a soft halo by QUOTE_BLUR
QUOTE_BLUR = 4
QUOTE_MARGIN_H_AT_1080 = 120
QUOTE_VERTICAL_FRACTION = 0.46
QUOTE_COLOR = "&H00FFFFFF"
QUOTE_HALO_COLOR = "&H80000000"  # half-transparent black
QUOTE_FADE_MS = 250


def escape_ass_text(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", r"\N")


def format_ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    cs_total = round(seconds * 100)
    hours, rem = divmod(cs_total, 360000)
    minutes, rem = divmod(rem, 6000)
    secs, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def build_header(project: ProjectSpec, config: AppConfig) -> tuple[str, int]:
    """Returns (header text, subtitle centre y in pixels)."""
    scale = project.height / 1920
    margin_v = round(project.height * 0.06)
    font = config.subtitle.font_name
    if config.subtitle.style == "quote":
        font_size = round(QUOTE_FONT_SIZE_AT_1920 * scale)
        halo = round(QUOTE_HALO_AT_1920 * scale, 1)
        margin_h = round(QUOTE_MARGIN_H_AT_1080 * (project.width / 1080))
        pos_y = round(project.height * QUOTE_VERTICAL_FRACTION)
        style = (
            f"Style: quote,{font},{font_size},{QUOTE_COLOR},{QUOTE_COLOR},{QUOTE_HALO_COLOR},"
            f"{QUOTE_HALO_COLOR},0,-1,0,0,100,100,0,0,1,{halo},0,5,"
            f"{margin_h},{margin_h},{margin_v},1"
        )
        wrap_style = 0  # a whole quote line is shown at once: wrap it into balanced lines
    else:
        font_size = round(FONT_SIZE_AT_1920 * scale)
        outline = round(OUTLINE_WIDTH_AT_1920 * scale, 1)
        shadow = round(SHADOW_AT_1920 * scale, 1)
        margin_h = round(MARGIN_H_AT_1080 * (project.width / 1080))
        pos_y = round(project.height * VERTICAL_FRACTION)
        style = (
            f"Style: shorts,{font},{font_size},"
            f"{PRIMARY_COLOR},{SECONDARY_COLOR},{OUTLINE_COLOR},{BACK_COLOR},"
            f"-1,0,0,0,100,100,0,0,1,{outline},{shadow},5,{margin_h},{margin_h},{margin_v},1"
        )
        wrap_style = 2
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"WrapStyle: {wrap_style}\n"
        "ScaledBorderAndShadow: yes\n"
        f"PlayResX: {project.width}\n"
        f"PlayResY: {project.height}\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"{style}\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    return header, pos_y


def build_karaoke_text(words: list[WordTiming]) -> str:
    """One dialogue line with a `\\k` tag per word (plus gap fillers), so libass sweeps the
    highlight from SecondaryColour to PrimaryColour in sync with each word."""
    parts = []
    prev_end = words[0].start
    for w in words:
        gap_cs = round((w.start - prev_end) * 100)
        if gap_cs > 0:
            parts.append(f"{{\\k{gap_cs}}}")
        dur_cs = max(MIN_KARAOKE_CS, round((w.end - w.start) * 100))
        parts.append(f"{{\\k{dur_cs}}}{escape_ass_text(w.word)} ")
        prev_end = w.end
    return "".join(parts).strip()


def _dialogue(start: float, end: float, x: int, y: int, text: str, style: str = "shorts") -> str:
    return (
        f"Dialogue: 0,{format_ass_time(start)},{format_ass_time(end)},{style},,0,0,0,,"
        f"{{\\an5\\pos({x},{y})}}{text}\n"
    )


def build_scene_events(scene: Scene, config: AppConfig, center_x: int, center_y: int) -> str:
    assert scene.audio is not None
    max_words = config.subtitle.max_words
    max_chars = config.subtitle.max_chars
    events = []
    if scene.audio.words:
        for group in chunk_words(scene.audio.words, max_words, max_chars):
            start = scene.start + group[0].start
            end = scene.start + group[-1].end
            events.append(_dialogue(start, end, center_x, center_y, build_karaoke_text(group)))
    else:
        chunks = chunk_text(scene.narration, max_words, max_chars)
        total_chars = sum(len(c) for c in chunks) or 1
        cursor = scene.start
        for chunk in chunks:
            duration = scene.audio.duration * (len(chunk) / total_chars)
            events.append(
                _dialogue(cursor, cursor + duration, center_x, center_y, escape_ass_text(chunk))
            )
            cursor += duration
    return "".join(events)


def build_quote_events(
    scene: Scene, config: AppConfig, center_x: int, center_y: int, scene_end: float
) -> str:
    """Each chunk (normally the whole line) fades in when it starts being spoken and stays on
    screen until the next chunk starts, or until `scene_end` for the last one."""
    assert scene.audio is not None and scene.start is not None
    max_words = config.subtitle.max_words
    max_chars = config.subtitle.max_chars
    timed: list[tuple[float, str]] = []
    if scene.audio.words:
        for group in chunk_words(scene.audio.words, max_words, max_chars):
            timed.append((scene.start + group[0].start, " ".join(w.word for w in group)))
    else:
        chunks = chunk_text(scene.narration, max_words, max_chars)
        total_chars = sum(len(c) for c in chunks) or 1
        cursor = scene.start
        for chunk in chunks:
            timed.append((cursor, chunk))
            cursor += scene.audio.duration * (len(chunk) / total_chars)
    ends = [start for start, _ in timed[1:]] + [scene_end]
    fx = f"{{\\fad({QUOTE_FADE_MS},{QUOTE_FADE_MS})\\blur{QUOTE_BLUR}}}"
    return "".join(
        _dialogue(start, end, center_x, center_y, fx + escape_ass_text(text), "quote")
        for (start, text), end in zip(timed, ends, strict=True)
    )


def build_ass_document(project: ProjectSpec, config: AppConfig) -> str:
    header, pos_y = build_header(project, config)
    center_x = project.width // 2
    scenes = [scene for scene in project.scenes if scene.audio is not None]
    if config.subtitle.style == "quote" and scenes:
        # A line stays up through the gap after it, until the next line starts.
        last = scenes[-1]
        assert last.start is not None and last.audio is not None
        ends = [scene.start for scene in scenes[1:]]
        ends.append(last.start + last.audio.duration + config.timeline.tail_seconds)
        body = [
            build_quote_events(scene, config, center_x, pos_y, end)
            for scene, end in zip(scenes, ends, strict=True)
        ]
    else:
        body = [build_scene_events(scene, config, center_x, pos_y) for scene in scenes]
    return header + "".join(body)
