from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Word:
    word: str
    start: float | None = None
    end: float | None = None
    score: float | None = None
    speaker: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"word": self.word}
        if self.start is not None:
            out["start"] = self.start
        if self.end is not None:
            out["end"] = self.end
        if self.score is not None:
            out["score"] = self.score
        if self.speaker is not None:
            out["speaker"] = self.speaker
        return out


@dataclass(frozen=True)
class Segment:
    id: int
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    speaker: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
        }
        if self.speaker is not None:
            out["speaker"] = self.speaker
        return out


@dataclass(frozen=True)
class TranscriptionResult:
    source_file: str
    normalized_audio_file: str
    model: str
    language: str | None
    duration_seconds: float | None
    full_text: str
    segments: list[Segment]
    # Reserved for later extensions (e.g., diarization) without breaking the schema.
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "full_text": self.full_text,
            "source_file": self.source_file,
            "normalized_audio_file": self.normalized_audio_file,
            "model": self.model,
            "language": self.language,
            "duration_seconds": self.duration_seconds,
            "segments": [s.to_dict() for s in self.segments],
        }
        if self.extras:
            out["extras"] = self.extras
        return out

