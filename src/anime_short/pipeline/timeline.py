"""Distribute scene duration over shots and assign absolute start times (SPEC §9).

Pure, unit-testable helper functions plus the `run_timeline` stage entry point. A scene's
"span" is its audio duration plus the gap that follows it (or a short tail after the last
scene), so shot durations always sum exactly to `audio.duration + gap`, and the concatenated
video track stays in sync with the voice timeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

from anime_short.config import AppConfig
from anime_short.pipeline.state import save_project
from anime_short.schemas import ProjectSpec, Scene, StageStatus

log = logging.getLogger(__name__)


def compute_cut_points(total_duration: float, weights: list[float]) -> list[float]:
    """Internal cut points (n-1 of them) splitting `total_duration` proportional to `weights`."""
    if len(weights) <= 1:
        return []
    total_weight = sum(weights)
    if total_weight <= 0:
        weights = [1.0] * len(weights)
        total_weight = float(len(weights))
    cum = 0.0
    points = []
    for w in weights[:-1]:
        cum += w
        points.append(total_duration * cum / total_weight)
    return points


def snap_to_words(
    points: list[float], word_ends: list[float], total_duration: float
) -> list[float]:
    """Snap each cut point to the nearest word-end boundary, keeping points strictly increasing."""
    if not word_ends:
        return list(points)
    snapped = []
    prev = 0.0
    for point in points:
        candidates = [w for w in word_ends if prev < w < total_duration]
        best = min(candidates, key=lambda w: abs(w - point)) if candidates else point
        best = max(best, prev + 0.01)
        best = min(best, total_duration - 0.01)
        snapped.append(best)
        prev = best
    return snapped


def durations_from_points(points: list[float], total_duration: float) -> list[float]:
    bounds = [0.0, *points, total_duration]
    return [b - a for a, b in zip(bounds, bounds[1:], strict=False)]


def merge_short_durations(durations: list[float], min_shot_seconds: float) -> list[list[int]]:
    """Group original indices, merging any run shorter than `min_shot_seconds` into a neighbour
    (the following shot, or the previous one when it's the last shot)."""
    groups: list[list[int]] = [[i] for i in range(len(durations))]
    dur = list(durations)
    while len(dur) > 1:
        short_idx = next((j for j, d in enumerate(dur) if d < min_shot_seconds), None)
        if short_idx is None:
            break
        target = short_idx - 1 if short_idx == len(dur) - 1 else short_idx + 1
        lo, hi = sorted((short_idx, target))
        dur[lo] += dur[hi]
        del dur[hi]
        groups[lo] = groups[lo] + groups[hi]
        del groups[hi]
    return groups


def plan_scene_shots(scene: Scene, min_shot_seconds: float) -> tuple[list[float], list[list[int]]]:
    """Split `scene.audio.duration` over `scene.shots` by weight, snapped to word boundaries
    when available, then merge shots shorter than `min_shot_seconds`.

    Returns (merged_durations, groups) where `groups[i]` lists the original shot indices merged
    into merged shot `i` (its duration is `merged_durations[i]`).
    """
    assert scene.audio is not None
    total = scene.audio.duration
    weights = [shot.weight for shot in scene.shots]
    word_ends = [w.end for w in scene.audio.words]
    points = compute_cut_points(total, weights)
    points = snap_to_words(points, word_ends, total)
    durations = durations_from_points(points, total)
    groups = merge_short_durations(durations, min_shot_seconds)
    merged_durations = [sum(durations[i] for i in group) for group in groups]
    return merged_durations, groups


def resolve_scene(
    project: ProjectSpec, scene: Scene, config: AppConfig, *, allow_replan: bool = True
) -> list[float]:
    """Compute (and apply to `scene.shots`) shot durations for one scene, re-planning the scene
    once via `script.stage.replan_scene` if any shot would exceed `max_shot_seconds`."""
    durations, groups = plan_scene_shots(scene, config.timeline.min_shot_seconds)
    if allow_replan and any(d > config.timeline.max_shot_seconds for d in durations):
        from anime_short.script.stage import replan_scene  # lazy: avoids an import cycle

        replan_scene(project, scene, config)
        durations, groups = plan_scene_shots(scene, config.timeline.min_shot_seconds)
        if any(d > config.timeline.max_shot_seconds for d in durations):
            log.warning(
                "%s: still has a shot longer than max_shot_seconds (%.1fs) after replan",
                scene.id,
                config.timeline.max_shot_seconds,
            )

    new_shots = []
    for group, duration in zip(groups, durations, strict=True):
        shot = scene.shots[group[0]]
        shot.duration = duration
        new_shots.append(shot)
    scene.shots = new_shots
    return durations


def run_timeline(
    project: ProjectSpec, project_dir: Path, config: AppConfig, force: bool = False
) -> None:
    if project.stages.get("timeline") == StageStatus.done and not force:
        return

    tail_seconds = config.timeline.tail_seconds
    gap_seconds = config.timeline.scene_gap_seconds
    cursor = 0.0
    for idx, scene in enumerate(project.scenes):
        if scene.audio is None:
            raise ValueError(f"{scene.id}: run the tts stage before timeline")
        durations = resolve_scene(project, scene, config)
        is_last = idx == len(project.scenes) - 1
        durations[-1] += tail_seconds if is_last else gap_seconds
        scene.shots[-1].duration = durations[-1]

        scene.start = cursor
        shot_cursor = cursor
        for shot, duration in zip(scene.shots, durations, strict=True):
            shot.start = shot_cursor
            shot.duration = duration
            shot_cursor += duration
        cursor += sum(durations)

    project.stages["timeline"] = StageStatus.done
    save_project(project, project_dir)
