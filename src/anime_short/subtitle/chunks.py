"""Split narration into subtitle chunks of <= max_words / max_chars, preferring punctuation
breaks (SPEC §10). Works on plain text or on word-timed tokens (`WordTiming`).
"""

from __future__ import annotations

from anime_short.schemas import WordTiming

_BREAK_CHARS = ".,!?;:…"


def _last_break_index(words: list[str]) -> int | None:
    for i in range(len(words) - 1, -1, -1):
        if words[i][-1:] in _BREAK_CHARS:
            return i
    return None


def _group_indices(tokens: list[str], max_words: int, max_chars: int) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    for i, _tok in enumerate(tokens):
        candidate = current + [i]
        text_len = len(" ".join(tokens[j] for j in candidate))
        if current and (len(candidate) > max_words or text_len > max_chars):
            break_at = _last_break_index([tokens[j] for j in current])
            if break_at is not None and break_at < len(current) - 1:
                groups.append(current[: break_at + 1])
                current = current[break_at + 1 :] + [i]
            else:
                groups.append(current)
                current = [i]
        else:
            current.append(i)
    if current:
        groups.append(current)
    return groups


def chunk_text(text: str, max_words: int, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    groups = _group_indices(words, max_words, max_chars)
    return [" ".join(words[i] for i in g) for g in groups]


def chunk_words(words: list[WordTiming], max_words: int, max_chars: int) -> list[list[WordTiming]]:
    tokens = [w.word for w in words]
    if not tokens:
        return []
    groups = _group_indices(tokens, max_words, max_chars)
    return [[words[i] for i in g] for g in groups]
