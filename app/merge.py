from __future__ import annotations

import re
from dataclasses import dataclass

from .chunking import AudioChunk
from .models import Segment, Word


@dataclass(frozen=True)
class ChunkResult:
    chunk: AudioChunk
    segments: list[Segment]


_WS_RE = re.compile(r"\s+")


def _norm_word(w: str) -> str:
    return _WS_RE.sub(" ", w.strip().lower())


def _word_midpoint(w: Word) -> float | None:
    if w.start is None or w.end is None:
        return None
    return (w.start + w.end) / 2.0


def shift_to_global(segments: list[Segment], offset_sec: float) -> list[Segment]:
    out: list[Segment] = []
    for seg in segments:
        words: list[Word] = []
        for w in seg.words:
            words.append(
                Word(
                    word=w.word,
                    start=(w.start + offset_sec) if w.start is not None else None,
                    end=(w.end + offset_sec) if w.end is not None else None,
                    score=w.score,
                    speaker=w.speaker,
                )
            )
        out.append(
            Segment(
                id=seg.id,
                start=seg.start + offset_sec,
                end=seg.end + offset_sec,
                text=seg.text,
                words=words,
                speaker=seg.speaker,
            )
        )
    return out


def filter_to_core_region(
    segments: list[Segment],
    *,
    core_start_sec: float,
    core_end_sec: float,
) -> list[Segment]:
    """
    Keep only words whose midpoint lies in [core_start_sec, core_end_sec).
    Drop segments with no remaining words.
    """
    out: list[Segment] = []
    for seg in segments:
        kept_words: list[Word] = []
        for w in seg.words:
            mid = _word_midpoint(w)
            if mid is None:
                continue
            if core_start_sec <= mid < core_end_sec:
                kept_words.append(w)
        if kept_words:
            out.append(
                Segment(
                    id=seg.id,
                    start=kept_words[0].start if kept_words[0].start is not None else seg.start,
                    end=kept_words[-1].end if kept_words[-1].end is not None else seg.end,
                    text=seg.text,
                    words=kept_words,
                    speaker=seg.speaker,
                )
            )
    return out


def dedupe_adjacent_words(
    segments: list[Segment],
    *,
    max_time_delta_sec: float = 0.35,
) -> list[Segment]:
    """
    Deterministic de-duplication by comparing each word with the immediately
    previous kept word. If text matches and start times are very close, drop it.
    """
    flat: list[Word] = []
    for seg in segments:
        flat.extend(seg.words)

    deduped: list[Word] = []
    prev: Word | None = None
    for w in flat:
        if prev is not None and prev.start is not None and w.start is not None:
            if _norm_word(prev.word) == _norm_word(w.word) and abs(prev.start - w.start) <= max_time_delta_sec:
                continue
        deduped.append(w)
        prev = w

    # Rebuild segments as a simple single segment list-of-words container (v1).
    # Keeps schema stable and deterministic; can be improved later.
    if not deduped:
        return []
    start = deduped[0].start or 0.0
    end = deduped[-1].end or start
    text = " ".join(w.word for w in deduped).strip()
    return [Segment(id=0, start=float(start), end=float(end), text=text, words=deduped, speaker=None)]


def merge_chunk_results(results: list[ChunkResult]) -> list[Segment]:
    """
    v1 merge strategy:
    - shift each chunk's timestamps to global
    - keep only the trusted core region words
    - sort by time
    - dedupe adjacent words in overlaps
    - rebuild into a stable segment list
    """
    all_segments: list[Segment] = []
    for r in results:
        global_segs = shift_to_global(r.segments, r.chunk.start_sec)
        core_segs = filter_to_core_region(
            global_segs,
            core_start_sec=r.chunk.core_start_sec,
            core_end_sec=r.chunk.core_end_sec,
        )
        all_segments.extend(core_segs)

    # Sort everything by time to make merge deterministic.
    all_segments.sort(key=lambda s: (s.start, s.end))
    return dedupe_adjacent_words(all_segments)

