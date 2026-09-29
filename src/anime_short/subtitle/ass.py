"""Build the burned-subtitle ASS file (SPEC §10): a "shorts" style — Be Vietnam Pro, large,
white with a thick dark outline, centred around ~70% of the frame height. Word timings (when
present) drive per-word karaoke highlighting; otherwise chunks are timed proportionally by
character count within the scene's audio duration.
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
    font_size = round(FONT_SIZE_AT_1920 * scale)
    outline = round(OUTLINE_WIDTH_AT_1920 * scale, 1)
    shadow = round(SHADOW_AT_1920 * scale, 1)
    margin_h = round(MARGIN_H_AT_1080 * (project.width / 1080))
    margin_v = round(project.height * 0.06)
    pos_y = round(project.height * VERTICAL_FRACTION)
    style = (
        f"Style: shorts,{config.subtitle.font_name},{font_size},"
        f"{PRIMARY_COLOR},{SECONDARY_COLOR},{OUTLINE_COLOR},{BACK_COLOR},"
        f"-1,0,0,0,100,100,0,0,1,{outline},{shadow},5,{margin_h},{margin_h},{margin_v},1"
    )
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "WrapStyle: 2\n"
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


def _dialogue(start: float, end: float, x: int, y: int, text: str) -> str:
    return (
        f"Dialogue: 0,{format_ass_time(start)},{format_ass_time(end)},shorts,,0,0,0,,"
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


def build_ass_document(project: ProjectSpec, config: AppConfig) -> str:
    header, pos_y = build_header(project, config)
    center_x = project.width // 2
    body = [
        build_scene_events(scene, config, center_x, pos_y)
        for scene in project.scenes
        if scene.audio is not None
    ]
    return header + "".join(body)
