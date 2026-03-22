from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .chunking import AudioChunk
from .merge import ChunkResult, merge_chunk_results
from .models import Segment, TranscriptionResult, Word

logger = logging.getLogger(__name__)


class TranscriptionError(RuntimeError):
    pass


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _segment_from_dict(seg: dict[str, Any], idx: int) -> Segment:
    words: list[Word] = []
    for w in (seg.get("words") or []):
        word_text = str(w.get("word", "")).strip()
        if not word_text:
            continue
        words.append(
            Word(
                word=word_text,
                start=w.get("start"),
                end=w.get("end"),
                score=w.get("score"),
            )
        )
    return Segment(
        id=int(seg.get("id", idx)),
        start=float(seg.get("start", 0.0)),
        end=float(seg.get("end", 0.0)),
        text=str(seg.get("text", "")),
        words=words,
    )


class WhisperXRunner:
    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        compute_type: str,
        language: str | None,
        batch_size: int,
    ) -> None:
        try:
            import whisperx  # type: ignore
        except Exception as e:
            raise TranscriptionError(
                "Failed to import whisperx. Install dependencies from requirements.txt."
            ) from e

        self._whisperx = whisperx
        self._device = device
        self._batch_size = batch_size
        self._language = language
        self._model_name = model_name
        self._compute_type = compute_type

        self._model = None
        self._align_model = None
        self._align_metadata = None
        self._detected_language: str | None = None

    def load_model(self) -> float:
        t0 = time.perf_counter()
        self._model = self._whisperx.load_model(
            self._model_name, self._device, compute_type=self._compute_type
        )
        return time.perf_counter() - t0

    @property
    def detected_language(self) -> str | None:
        return self._detected_language or self._language

    def _ensure_align_loaded(self) -> float:
        if self._align_model is not None and self._align_metadata is not None:
            return 0.0
        lang = self.detected_language
        if not lang:
            raise TranscriptionError("Language could not be detected for alignment.")
        t0 = time.perf_counter()
        self._align_model, self._align_metadata = self._whisperx.load_align_model(
            language_code=lang,
            device=self._device,
        )
        return time.perf_counter() - t0

    def transcribe_align(self, audio_path: Path) -> tuple[list[Segment], dict[str, float]]:
        if self._model is None:
            raise TranscriptionError("Model not loaded.")

        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        result: dict[str, Any] = self._model.transcribe(
            str(audio_path),
            batch_size=self._batch_size,
            language=self._language,
        )
        timings["whisperx_transcribe_seconds"] = time.perf_counter() - t0

        self._detected_language = result.get("language") or self._language
        t_align_load = self._ensure_align_loaded()
        if t_align_load:
            timings["whisperx_load_align_model_seconds"] = t_align_load

        t0 = time.perf_counter()
        aligned: dict[str, Any] = self._whisperx.align(
            result["segments"],
            self._align_model,
            self._align_metadata,
            str(audio_path),
            self._device,
            return_char_alignments=False,
        )
        timings["whisperx_align_seconds"] = time.perf_counter() - t0

        segments = [_segment_from_dict(seg, idx) for idx, seg in enumerate(aligned.get("segments", []))]
        return segments, timings


def transcribe_and_align(
    *,
    audio_path: Path,
    source_path: Path,
    model_name: str,
    device: str,
    compute_type: str,
    language: str | None,
    batch_size: int,
) -> TranscriptionResult:
    """
    Run WhisperX transcription + alignment to produce word-level timestamps.
    """
    resolved_device = _resolve_device(device)
    timings: dict[str, float] = {}

    try:
        logger.info(
            "Loading WhisperX model. model=%s device=%s compute_type=%s",
            model_name,
            resolved_device,
            compute_type,
        )
        runner = WhisperXRunner(
            model_name=model_name,
            device=resolved_device,
            compute_type=compute_type,
            language=language,
            batch_size=batch_size,
        )
        timings["whisperx_load_model_seconds"] = runner.load_model()
        segments, step_timings = runner.transcribe_align(audio_path)
        timings.update(step_timings)
        detected_lang = runner.detected_language
    except Exception as e:
        raise TranscriptionError(f"Transcription failed: {e}") from e

    full_text = " ".join(s.text.strip() for s in segments if s.text and s.text.strip()).strip()

    return TranscriptionResult(
        source_file=str(source_path),
        normalized_audio_file=str(audio_path),
        model=model_name,
        language=detected_lang,
        duration_seconds=None,  # caller fills if desired
        full_text=full_text,
        segments=segments,
        extras={
            "runtime_stats": timings,
            "device": resolved_device,
            "compute_type": compute_type,
        },
    )


def transcribe_and_align_chunked(
    *,
    chunks: list[AudioChunk],
    source_path: Path,
    normalized_audio_path: Path,
    model_name: str,
    device: str,
    compute_type: str,
    language: str | None,
    batch_size: int,
) -> TranscriptionResult:
    if not chunks:
        raise TranscriptionError("No chunks were produced for chunked processing.")

    resolved_device = _resolve_device(device)
    timings: dict[str, float] = {}

    runner = WhisperXRunner(
        model_name=model_name,
        device=resolved_device,
        compute_type=compute_type,
        language=language,
        batch_size=batch_size,
    )
    timings["whisperx_load_model_seconds"] = runner.load_model()

    results: list[ChunkResult] = []
    chunks_meta: list[dict[str, Any]] = []

    for ch in chunks:
        logger.info("Processing chunk %s (%0.2fs-%0.2fs).", ch.index, ch.start_sec, ch.end_sec)
        segs, t = runner.transcribe_align(ch.path)
        # keep per-chunk timings only as totals (v1)
        timings[f"chunk_{ch.index:04d}_transcribe_seconds"] = t.get("whisperx_transcribe_seconds", 0.0)
        timings[f"chunk_{ch.index:04d}_align_seconds"] = t.get("whisperx_align_seconds", 0.0)
        results.append(ChunkResult(chunk=ch, segments=segs))
        chunks_meta.append(
            {
                "index": ch.index,
                "start_sec": ch.start_sec,
                "end_sec": ch.end_sec,
                "core_start_sec": ch.core_start_sec,
                "core_end_sec": ch.core_end_sec,
                "path": str(ch.path),
            }
        )

    merged_segments = merge_chunk_results(results)
    # Re-id segments sequentially
    merged_segments = [
        Segment(
            id=i,
            start=s.start,
            end=s.end,
            text=s.text,
            words=s.words,
        )
        for i, s in enumerate(merged_segments)
    ]

    full_text = " ".join(s.text.strip() for s in merged_segments if s.text and s.text.strip()).strip()

    extras: dict[str, Any] = {
        "runtime_stats": timings,
        "device": resolved_device,
        "compute_type": compute_type,
        "chunking_enabled": True,
        "chunk_length_sec": chunks[0].end_sec - chunks[0].start_sec,
        "chunk_overlap_sec": (chunks[1].start_sec - chunks[0].start_sec) if len(chunks) > 1 else 0.0,
        "chunks_processed": len(chunks),
        "chunks": chunks_meta,
    }

    return TranscriptionResult(
        source_file=str(source_path),
        normalized_audio_file=str(normalized_audio_path),
        model=model_name,
        language=runner.detected_language,
        duration_seconds=None,
        full_text=full_text,
        segments=merged_segments,
        extras=extras,
    )

