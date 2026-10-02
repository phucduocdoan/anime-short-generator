from __future__ import annotations

import re

import pytest

from anime_short.config import AppConfig, load_config
from anime_short.schemas import (
    AudioInfo,
    ContentSpec,
    ProjectSpec,
    Scene,
    Shot,
    VisualSpec,
    WordTiming,
)
from anime_short.subtitle.ass import (
    build_ass_document,
    build_header,
    build_karaoke_text,
    escape_ass_text,
    format_ass_time,
)
from anime_short.subtitle.chunks import chunk_text, chunk_words
from anime_short.subtitle.stage import run_subtitles


@pytest.fixture
def config() -> AppConfig:
    return load_config(
        {
            "llm": {"base_url": "http://x", "model": "m"},
            "tts": {"provider": "vieneu"},
            "image": {"provider": "anima"},
            "video": {"provider": "animegen"},
        }
    )


# --- chunks ---------------------------------------------------------------------------------


def test_chunk_text_respects_max_words():
    text = "mot hai ba bon nam sau bay tam"
    chunks = chunk_text(text, max_words=3, max_chars=100)
    assert all(len(c.split()) <= 3 for c in chunks)
    assert " ".join(chunks).replace("  ", " ") == text


def test_chunk_text_respects_max_chars():
    text = "nhung hom ay chuyen tau cuoi cung khong bao gio den"
    chunks = chunk_text(text, max_words=10, max_chars=20)
    assert all(len(c) <= 20 for c in chunks)


def test_chunk_text_prefers_punctuation_break():
    text = "Xin chao, the gioi hom nay dep troi"
    chunks = chunk_text(text, max_words=4, max_chars=100)
    # "Xin chao," ends with punctuation and fits within max_words -> should be its own chunk
    assert chunks[0] == "Xin chao,"


def test_chunk_text_empty():
    assert chunk_text("", max_words=5, max_chars=28) == []


def test_chunk_words_matches_chunk_text_grouping():
    letters = ["a", "b", "c", "d", "e"]
    words = [WordTiming(word=w, start=i, end=i + 1) for i, w in enumerate(letters)]
    groups = chunk_words(words, max_words=2, max_chars=100)
    assert [len(g) for g in groups] == [2, 2, 1]
    assert groups[0][0].word == "a"


# --- ass primitives ---------------------------------------------------------------------------


def test_escape_ass_text():
    assert escape_ass_text("a{b}c\\d\ne") == r"a\{b\}c\\d\Ne"


@pytest.mark.parametrize(
    "seconds,expected",
    [(0, "0:00:00.00"), (1.5, "0:00:01.50"), (61.25, "0:01:01.25"), (3661.0, "1:01:01.00")],
)
def test_format_ass_time(seconds, expected):
    assert format_ass_time(seconds) == expected


def test_build_header_scales_with_resolution(config):
    project = ProjectSpec(id="p", title="t", content=ContentSpec(topic="t"))
    header, pos_y = build_header(project, config)
    assert "PlayResX: 1080" in header
    assert "PlayResY: 1920" in header
    assert "Be Vietnam Pro" in header
    assert 0.68 * 1920 <= pos_y <= 0.72 * 1920


def test_build_karaoke_text_has_k_tags_and_diacritics():
    words = [
        WordTiming(word="Nhưng", start=0.0, end=0.3),
        WordTiming(word="hôm", start=0.3, end=0.6),
        WordTiming(word="ấy,", start=0.6, end=1.0),
    ]
    text = build_karaoke_text(words)
    assert text.count(r"\k") == 3
    assert "Nhưng" in text
    assert "ấy," in text


# --- full document ---------------------------------------------------------------------------


def make_scene_with_words(scene_id: str, start: float) -> Scene:
    words = [
        WordTiming(word="Nhưng", start=0.0, end=0.3),
        WordTiming(word="hôm", start=0.3, end=0.6),
        WordTiming(word="ấy,", start=0.6, end=1.0),
        WordTiming(word="chuyến", start=1.0, end=1.4),
        WordTiming(word="tàu", start=1.4, end=1.7),
        WordTiming(word="cuối", start=1.7, end=2.0),
        WordTiming(word="cùng", start=2.0, end=2.3),
    ]
    scene = Scene(
        id=scene_id,
        narration="Nhưng hôm ấy, chuyến tàu cuối cùng không bao giờ đến.",
        audio=AudioInfo(path="audio/x.wav", duration=2.3, words=words),
        shots=[Shot(id=f"{scene_id}_a", visual=VisualSpec(subject="train"))],
    )
    scene.start = start
    return scene


def make_scene_without_words(scene_id: str, start: float) -> Scene:
    scene = Scene(
        id=scene_id,
        narration="2 giờ sáng, điện thoại của Minh bỗng sáng lên.",
        audio=AudioInfo(path="audio/y.wav", duration=3.0, words=[]),
        shots=[Shot(id=f"{scene_id}_a", visual=VisualSpec(subject="phone"))],
    )
    scene.start = start
    return scene


def test_build_ass_document_word_timed_scene(config):
    scene = make_scene_with_words("scene_01", start=5.0)
    project = ProjectSpec(id="p", title="t", content=ContentSpec(topic="t"), scenes=[scene])
    doc = build_ass_document(project, config)
    assert "[Events]" in doc
    assert doc.count("Dialogue:") >= 1
    # absolute time offset by scene.start
    assert re.search(r"Dialogue: 0,0:00:05\.00,", doc)
    assert "Nhưng" in doc  # diacritics preserved


def test_build_ass_document_proportional_timing_without_words(config):
    scene = make_scene_without_words("scene_01", start=0.0)
    project = ProjectSpec(id="p", title="t", content=ContentSpec(topic="t"), scenes=[scene])
    doc = build_ass_document(project, config)
    dialogue_lines = [line for line in doc.splitlines() if line.startswith("Dialogue:")]
    assert len(dialogue_lines) >= 1
    # last event should end at (or very near) the scene duration
    last_end = dialogue_lines[-1].split(",")[2]
    hh, mm, ss = last_end.split(":")
    total = int(hh) * 3600 + int(mm) * 60 + float(ss)
    assert total == pytest.approx(3.0, abs=0.05)


def test_run_subtitles_writes_file_and_marks_done(config, tmp_path):
    scene = make_scene_with_words("scene_01", start=0.0)
    project = ProjectSpec(id="p", title="t", content=ContentSpec(topic="t"), scenes=[scene])
    run_subtitles(project, tmp_path, config)
    out = tmp_path / "subtitles" / "final.ass"
    assert out.exists()
    assert "Nhưng" in out.read_text(encoding="utf-8")
    assert project.stages["subtitles"] == "done"
    assert (tmp_path / "project.json").exists()


# --- quote style -----------------------------------------------------------------------------


def _dialogue_times(doc: str) -> list[tuple[str, str]]:
    lines = [line for line in doc.splitlines() if line.startswith("Dialogue:")]
    return [tuple(line.split(",")[1:3]) for line in lines]


def test_quote_style_header_is_italic_mid_frame(config):
    config.subtitle.style = "quote"
    config.subtitle.font_name = "Mali"
    project = ProjectSpec(id="p", title="t", content=ContentSpec(topic="t"))
    header, pos_y = build_header(project, config)
    style = next(line for line in header.splitlines() if line.startswith("Style:"))
    fields = style.split(",")
    assert fields[0] == "Style: quote" and fields[1] == "Mali"
    assert (fields[7], fields[8]) == ("0", "-1")  # not bold, italic
    assert "WrapStyle: 0" in header
    assert 0.4 * 1920 <= pos_y <= 0.5 * 1920


def test_quote_lines_stay_up_until_the_next_line(config):
    config.subtitle.style = "quote"
    config.subtitle.max_words = 16
    config.subtitle.max_chars = 90
    first = make_scene_with_words("scene_01", start=0.0)
    second = make_scene_without_words("scene_02", start=3.0)
    project = ProjectSpec(id="p", title="t", content=ContentSpec(topic="t"), scenes=[first, second])
    doc = build_ass_document(project, config)

    # one event per line: no karaoke, the whole line shown at once, held through the gap
    assert r"\k" not in doc
    assert r"\fad(" in doc
    assert _dialogue_times(doc) == [
        ("0:00:00.00", "0:00:03.00"),
        ("0:00:03.00", format_ass_time(3.0 + 3.0 + config.timeline.tail_seconds)),
    ]
    assert "Nhưng hôm ấy, chuyến tàu cuối cùng" in doc
