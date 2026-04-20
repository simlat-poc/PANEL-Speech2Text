from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .diarization import SpeakerTurn
from .models import Segment, Word

logger = logging.getLogger(__name__)


class VoiceprintError(RuntimeError):
    pass


@dataclass(frozen=True)
class EnrollmentSpec:
    name: str
    audio_path: Path


@dataclass(frozen=True)
class LabelEvidence:
    diarization_label: str
    candidate_name: str | None
    confidence: float
    segments_used: int
    total_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "diarization_label": self.diarization_label,
            "candidate_name": self.candidate_name,
            "confidence": self.confidence,
            "segments_used": self.segments_used,
            "total_seconds": self.total_seconds,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_enrollment_specs(items: tuple[str, ...]) -> list[EnrollmentSpec]:
    specs: list[EnrollmentSpec] = []
    for raw in items:
        if "=" not in raw:
            raise VoiceprintError(
                f"Invalid --voiceprint-enroll value '{raw}'. Expected NAME=PATH format."
            )
        name, path_str = raw.split("=", 1)
        name = name.strip()
        path = Path(path_str.strip()).expanduser().resolve()
        if not name:
            raise VoiceprintError(f"Invalid --voiceprint-enroll value '{raw}': empty name.")
        if not path.exists():
            raise VoiceprintError(
                f"Invalid --voiceprint-enroll value '{raw}': audio file not found ({path})."
            )
        specs.append(EnrollmentSpec(name=name, audio_path=path))
    return specs


def _load_audio_mono(audio_path: Path) -> tuple[np.ndarray, int]:
    try:
        import torchaudio  # type: ignore
    except Exception as e:
        raise VoiceprintError("torchaudio is required for voiceprint processing.") from e

    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.ndim != 2:
        raise VoiceprintError(f"Unexpected waveform shape for {audio_path}.")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    mono = waveform.squeeze(0).numpy().astype(np.float32, copy=False)
    return mono, int(sr)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class SpeakerEmbedder:
    def __init__(self, *, model_name: str, use_auth_token: str | None = None) -> None:
        try:
            from pyannote.audio import Inference, Model  # type: ignore
        except Exception as e:
            raise VoiceprintError(
                "pyannote.audio is required for voiceprint embedding extraction."
            ) from e

        try:
            model = Model.from_pretrained(model_name, use_auth_token=use_auth_token)
            self._inference = Inference(model, window="whole")
        except Exception as e:
            raise VoiceprintError(f"Failed to load embedding model '{model_name}': {e}") from e

    def embedding_from_waveform(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        if waveform.size == 0:
            raise VoiceprintError("Cannot extract embedding from empty waveform.")
        try:
            import torch
        except Exception as e:
            raise VoiceprintError("PyTorch is required for voiceprint embedding extraction.") from e

        sample = {
            "waveform": torch.from_numpy(waveform).unsqueeze(0),
            "sample_rate": int(sample_rate),
        }
        emb = self._inference(sample)
        vec = np.asarray(emb, dtype=np.float32).reshape(-1)
        nrm = float(np.linalg.norm(vec))
        if nrm > 0:
            vec = vec / nrm
        return vec


def _load_db_index(db_dir: Path) -> dict[str, Any]:
    index_path = db_dir / "index.json"
    if not index_path.exists():
        return {"version": 1, "embedding_model": None, "speakers": []}
    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise VoiceprintError(f"Failed to read voiceprint index at {index_path}: {e}") from e


def _write_db_index(db_dir: Path, index: dict[str, Any]) -> None:
    index_path = db_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=True), encoding="utf-8")


def load_enrolled_voiceprints(db_dir: Path) -> dict[str, np.ndarray]:
    index = _load_db_index(db_dir)
    enrolled: dict[str, np.ndarray] = {}
    for sp in index.get("speakers", []):
        name = str(sp.get("name", "")).strip()
        rel_file = str(sp.get("embedding_file", "")).strip()
        if not name or not rel_file:
            continue
        emb_path = db_dir / rel_file
        if not emb_path.exists():
            continue
        arr = np.load(str(emb_path)).astype(np.float32, copy=False).reshape(-1)
        nrm = float(np.linalg.norm(arr))
        if nrm > 0:
            arr = arr / nrm
        enrolled[name] = arr
    return enrolled


def enroll_voiceprints(
    *,
    db_dir: Path,
    specs: list[EnrollmentSpec],
    embedder: SpeakerEmbedder,
    embedding_model: str,
) -> dict[str, Any]:
    db_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = db_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    index = _load_db_index(db_dir)
    by_name: dict[str, dict[str, Any]] = {
        str(sp.get("name", "")).strip(): sp
        for sp in index.get("speakers", [])
        if str(sp.get("name", "")).strip()
    }

    enrolled_names: list[str] = []
    for spec in specs:
        waveform, sr = _load_audio_mono(spec.audio_path)
        emb = embedder.embedding_from_waveform(waveform, sr)
        emb_file = f"{spec.name}.npy"
        np.save(str(emb_dir / emb_file), emb)
        by_name[spec.name] = {
            "name": spec.name,
            "embedding_file": f"embeddings/{emb_file}",
            "source_audio": str(spec.audio_path),
            "updated_at": _utc_now(),
        }
        enrolled_names.append(spec.name)

    index["version"] = 1
    index["embedding_model"] = embedding_model
    index["speakers"] = sorted(by_name.values(), key=lambda x: x["name"])
    _write_db_index(db_dir, index)
    return {"count": len(enrolled_names), "names": sorted(enrolled_names)}


def _extract_segment_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
) -> np.ndarray:
    s = max(0, int(round(start * sample_rate)))
    e = max(s, int(round(end * sample_rate)))
    return waveform[s:e]


def _choose_label_mapping(
    speaker_vectors: dict[str, np.ndarray],
    enrolled: dict[str, np.ndarray],
    threshold: float,
) -> tuple[dict[str, str], list[LabelEvidence]]:
    candidates: list[tuple[str, str, float]] = []
    evidence: list[LabelEvidence] = []
    for diar_label, vec in speaker_vectors.items():
        best_name: str | None = None
        best_score = -1.0
        for name, ref_vec in enrolled.items():
            score = _cosine(vec, ref_vec)
            if score > best_score:
                best_score = score
                best_name = name
        confidence = max(0.0, best_score)
        evidence.append(
            LabelEvidence(
                diarization_label=diar_label,
                candidate_name=best_name if confidence >= threshold else None,
                confidence=confidence,
                segments_used=0,
                total_seconds=0.0,
            )
        )
        if best_name and confidence >= threshold:
            candidates.append((diar_label, best_name, confidence))

    mapping: dict[str, str] = {}
    used_names: set[str] = set()
    for diar_label, name, score in sorted(candidates, key=lambda x: x[2], reverse=True):
        if name in used_names:
            continue
        mapping[diar_label] = name
        used_names.add(name)
    return mapping, evidence


def relabel_segments(segments: list[Segment], mapping: dict[str, str]) -> list[Segment]:
    out: list[Segment] = []
    for seg in segments:
        seg_label = seg.speaker
        seg_speaker = mapping.get(seg_label, seg_label) if seg_label else None
        words: list[Word] = []
        for w in seg.words:
            w_label = w.speaker
            w_speaker = mapping.get(w_label, w_label) if w_label else None
            words.append(
                Word(
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    score=w.score,
                    speaker=w_speaker,
                )
            )
        out.append(
            Segment(
                id=seg.id,
                start=seg.start,
                end=seg.end,
                text=seg.text,
                words=words,
                speaker=seg_speaker,
            )
        )
    return out


def identify_speakers_from_turns(
    *,
    db_dir: Path,
    normalized_audio_path: Path,
    turns: list[SpeakerTurn],
    embedder: SpeakerEmbedder,
    threshold: float,
    min_segment_sec: float,
) -> tuple[dict[str, str], list[LabelEvidence]]:
    enrolled = load_enrolled_voiceprints(db_dir)
    if not enrolled:
        return {}, []

    waveform, sr = _load_audio_mono(normalized_audio_path)
    agg_sum: dict[str, np.ndarray] = {}
    agg_weight: dict[str, float] = {}
    seg_count: dict[str, int] = {}

    for t in turns:
        dur = float(t.end - t.start)
        if dur < min_segment_sec:
            continue
        chunk = _extract_segment_waveform(waveform, sr, t.start, t.end)
        if chunk.size == 0:
            continue
        emb = embedder.embedding_from_waveform(chunk, sr)
        if t.speaker not in agg_sum:
            agg_sum[t.speaker] = emb * dur
            agg_weight[t.speaker] = dur
            seg_count[t.speaker] = 1
        else:
            agg_sum[t.speaker] += emb * dur
            agg_weight[t.speaker] += dur
            seg_count[t.speaker] += 1

    speaker_vectors: dict[str, np.ndarray] = {}
    for label, vec_sum in agg_sum.items():
        w = agg_weight.get(label, 0.0)
        if w <= 0:
            continue
        avg = vec_sum / w
        nrm = float(np.linalg.norm(avg))
        if nrm > 0:
            avg = avg / nrm
        speaker_vectors[label] = avg

    mapping, evidence = _choose_label_mapping(speaker_vectors, enrolled, threshold)
    enriched: list[LabelEvidence] = []
    by_label = {ev.diarization_label: ev for ev in evidence}
    for label, ev in by_label.items():
        enriched.append(
            LabelEvidence(
                diarization_label=label,
                candidate_name=mapping.get(label),
                confidence=ev.confidence,
                segments_used=seg_count.get(label, 0),
                total_seconds=float(agg_weight.get(label, 0.0)),
            )
        )
    logger.info(
        "Voiceprint identification complete. assigned=%d total_diar_labels=%d",
        len(mapping),
        len(speaker_vectors),
    )
    return mapping, sorted(enriched, key=lambda x: x.diarization_label)
