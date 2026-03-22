from __future__ import annotations

import inspect
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Segment, Word

logger = logging.getLogger(__name__)


class DiarizationError(RuntimeError):
    pass


HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN")


def resolve_hf_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token.strip() or None
    for k in HF_TOKEN_ENV_VARS:
        v = os.environ.get(k)
        if v and v.strip():
            return v.strip()
    return None


@dataclass(frozen=True)
class SpeakerTurn:
    speaker: str
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        return {"speaker": self.speaker, "start": self.start, "end": self.end}


def run_whisperx_diarization(
    *,
    audio_path: Path,
    device: str,
    hf_token: str,
    min_speakers: int | None,
    max_speakers: int | None,
    num_speakers: int | None = None,
) -> list[SpeakerTurn]:
    try:
        import whisperx  # type: ignore
    except Exception as e:
        raise DiarizationError("Failed to import whisperx for diarization.") from e

    logger.info("Running diarization (WhisperX/pyannote).")
    try:
        diarize_ns = getattr(whisperx, "diarize", None)
        if diarize_ns is not None and hasattr(diarize_ns, "DiarizationPipeline"):
            pipeline_cls = diarize_ns.DiarizationPipeline
        elif hasattr(whisperx, "DiarizationPipeline"):
            pipeline_cls = whisperx.DiarizationPipeline
        else:
            raise DiarizationError(
                "WhisperX diarization API not found. Expected `whisperx.diarize.DiarizationPipeline`."
            )

        # WhisperX API changed from `use_auth_token` to `token` in newer versions.
        init_sig = inspect.signature(pipeline_cls.__init__)

        # Match WhisperX default: community-1 (lighter / older pipeline than 3.1).
        # Hugging Face repo id:
        #   pyannote/speaker-diarization-community-1
        desired_model_name = "pyannote/speaker-diarization-community-1"
        if "model_name" in init_sig.parameters:
            model_name_kw: dict[str, Any] = {"model_name": desired_model_name}
        else:
            model_name_kw = {}

        if "token" in init_sig.parameters:
            pipeline = pipeline_cls(token=hf_token, device=device, **model_name_kw)
        elif "use_auth_token" in init_sig.parameters:
            pipeline = pipeline_cls(use_auth_token=hf_token, device=device, **model_name_kw)
        else:
            # Best-effort default for unknown signatures.
            pipeline = pipeline_cls(token=hf_token, device=device, **model_name_kw)
        # WhisperX forwards these to pyannote.SpeakerDiarization (see whisperx/diarize.py).
        #
        # Important: pyannote does *not* guarantee that the output has exactly N speaker
        # labels. `num_speakers` / min/max are clustering *targets*; the pipeline may
        # still emit fewer speakers (e.g. too few embeddings after filtering, short audio,
        # or some clusters never active). See pyannote speaker_diarization.py around
        # num_different_speakers vs min/max and clustering.set_num_clusters capping by
        # num_embeddings.
        diarize_kwargs: dict[str, Any] = {}
        ns = int(num_speakers) if num_speakers is not None else None
        mn = int(min_speakers) if min_speakers is not None else None
        mx = int(max_speakers) if max_speakers is not None else None

        if ns is not None:
            if mn is not None or mx is not None:
                logger.warning(
                    "Diarization: explicit num_speakers=%s set; min_speakers/max_speakers are not passed to pyannote.",
                    ns,
                )
            diarize_kwargs["num_speakers"] = ns
            logger.info(
                "Diarization: num_speakers=%s (target for clustering; output may still have fewer labels)",
                ns,
            )
        elif mn is not None and mx is not None and mn == mx:
            diarize_kwargs["num_speakers"] = mn
            logger.info(
                "Diarization: num_speakers=%s (from min==max; target only; output may still have fewer labels)",
                mn,
            )
        else:
            if mn is not None:
                diarize_kwargs["min_speakers"] = mn
            if mx is not None:
                diarize_kwargs["max_speakers"] = mx
            if diarize_kwargs:
                logger.info("Diarization: min_speakers=%s max_speakers=%s", mn, mx)
        diarize_segments = pipeline(str(audio_path), **diarize_kwargs)
    except Exception as e:
        raise DiarizationError(f"Diarization failed: {e}") from e

    turns: list[SpeakerTurn] = []
    # whisperx diarization returns a pandas DataFrame-like object with
    # columns: start, end, speaker
    for _, row in diarize_segments.iterrows():  # type: ignore[attr-defined]
        turns.append(
            SpeakerTurn(
                speaker=str(row["speaker"]),
                start=float(row["start"]),
                end=float(row["end"]),
            )
        )
    # Sort for deterministic assignment
    turns.sort(key=lambda t: (t.start, t.end, t.speaker))
    return turns


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _pick_speaker_for_range(
    *,
    start: float,
    end: float,
    turns: list[SpeakerTurn],
) -> str | None:
    if end <= start:
        return None
    best_speaker: str | None = None
    best_overlap = 0.0
    for t in turns:
        ov = _overlap(start, end, t.start, t.end)
        if ov > best_overlap:
            best_overlap = ov
            best_speaker = t.speaker
    return best_speaker


def assign_speakers(
    *,
    segments: list[Segment],
    turns: list[SpeakerTurn],
    assign_words: bool = True,
) -> list[Segment]:
    out: list[Segment] = []
    for seg in segments:
        seg_speaker = _pick_speaker_for_range(start=seg.start, end=seg.end, turns=turns)
        words: list[Word] = []
        if assign_words:
            for w in seg.words:
                if w.start is not None and w.end is not None:
                    w_spk = _pick_speaker_for_range(start=float(w.start), end=float(w.end), turns=turns)
                else:
                    w_spk = None
                words.append(
                    Word(
                        word=w.word,
                        start=w.start,
                        end=w.end,
                        score=w.score,
                        speaker=w_spk,
                    )
                )
        else:
            words = seg.words

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

