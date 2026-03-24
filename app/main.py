"""
Offline transcription pipeline (WhisperX + alignment).

Quickstart:
  1) Install ffmpeg and ensure `ffmpeg` is on PATH.
  2) Create a venv and install deps:
       python -m venv .venv
       .\\.venv\\Scripts\\activate
       pip install -r requirements.txt
  3) Run:
      python -m app.main --input "path\\to\\file.mp4" --output "out.json" --model-preset original --device auto
"""

from __future__ import annotations

import argparse
import logging
import shutil
import platform
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

from .config import RuntimeConfig
from .chunking import materialize_chunks
from .diarization import assign_speakers, resolve_hf_token, run_whisperx_diarization
from .ffmpeg_utils import normalize_to_wav_mono_16k, validate_ffmpeg_exists
from .serializers import write_json
from .transcription import transcribe_and_align, transcribe_and_align_chunked
from .utils import configure_logging, resolve_device, wav_duration_seconds

logger = logging.getLogger(__name__)

MODEL_PRESETS: dict[str, str] = {
    "original": "large-v3",
    "ivrit-he": "ivrit-ai/whisper-large-v3-ct2",
}


def _ensure_supported_python() -> None:
    # WhisperX (and core deps like ctranslate2) commonly lag newest Python versions.
    # Pin here so users get an actionable error instead of a long pip backtrack.
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 14):
        raise RuntimeError(
            "Python 3.14+ is not supported by WhisperX dependencies yet. "
            "Please install Python 3.12 (recommended) or 3.11, then recreate your venv. "
            f"Detected: Python {platform.python_version()}."
        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline transcription using WhisperX + alignment.")
    p.add_argument("--input", required=True, type=Path, help="Path to input audio/video file.")
    p.add_argument("--output", required=True, type=Path, help="Path to output JSON.")
    p.add_argument(
        "--model",
        default=None,
        help=(
            "Custom Whisper model id/name (e.g. tiny, base, small, medium, large-v3, "
            "ivrit-ai/whisper-large-v3-ct2). Overrides --model-preset when provided."
        ),
    )
    p.add_argument(
        "--model-preset",
        default="original",
        choices=sorted(MODEL_PRESETS.keys()),
        help=(
            "Convenience model preset. "
            "'original' -> large-v3, "
            "'ivrit-he' -> ivrit-ai/whisper-large-v3-ct2."
        ),
    )
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device selection.")
    p.add_argument("--compute-type", default=None, help="Compute type (e.g. float16, int8). Default depends on device.")
    p.add_argument("--language", default=None, help="Language code (e.g. en). If omitted, auto-detect.")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size for transcription.")
    p.add_argument("--chunked", action="store_true", help="Enable chunked processing for long recordings.")
    p.add_argument("--chunk-length-sec", type=float, default=120.0, help="Chunk length in seconds (default: 120).")
    p.add_argument("--chunk-overlap-sec", type=float, default=5.0, help="Chunk overlap in seconds (default: 5).")
    p.add_argument("--enable-diarization", action="store_true", help="Enable speaker diarization.")
    p.add_argument("--hf-token", default=None, help="Hugging Face token (or set HF_TOKEN env var).")
    p.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Minimum speakers hint for pyannote (optional; not a hard guarantee).",
    )
    p.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum speakers hint for pyannote (optional; not a hard guarantee).",
    )
    p.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Target exact speaker count for pyannote (WhisperX num_speakers). "
            "Overrides --min-speakers/--max-speakers for the diarization call if set. "
            "Not a hard guarantee of N labels in the output."
        ),
    )
    p.add_argument("--temp-dir", type=Path, default=None, help="Optional base temp/work directory.")
    p.add_argument("--keep-temp-files", action="store_true", help="Keep normalized audio and temp files.")
    p.add_argument("--verbose", action="store_true", help="Verbose logging.")
    p.add_argument(
        "--export-html",
        action="store_true",
        help="After the JSON is written, also generate the standalone interactive HTML timeline (same as app.timeline_exporter).",
    )
    p.add_argument(
        "--html-output",
        type=Path,
        default=None,
        help="Path for the HTML file (default: <json-stem>_timeline.html next to --output JSON).",
    )
    p.add_argument(
        "--html-direction",
        choices=["auto", "ltr", "rtl"],
        default="auto",
        help="HTML timeline direction mode (auto: RTL timeline + RTL text when JSON language is Hebrew).",
    )
    p.add_argument(
        "--html-px-per-sec",
        type=float,
        default=8.0,
        help="Initial zoom baseline (pixels per second) embedded in the HTML.",
    )
    return p


def _default_compute_type(device: str) -> str:
    # WhisperX / faster-whisper commonly uses float16 on cuda, int8 on cpu.
    return "float16" if device == "cuda" else "int8"


@contextmanager
def _work_dir(base_temp_dir: Path | None, keep: bool) -> Iterator[Path]:
    if base_temp_dir is None:
        with TemporaryDirectory(prefix="whisperx_work_") as td:
            yield Path(td)
        return

    base_temp_dir.mkdir(parents=True, exist_ok=True)
    run_dir = base_temp_dir / f"whisperx_run_{uuid.uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield run_dir
    finally:
        if not keep:
            shutil.rmtree(run_dir, ignore_errors=True)


def _validate_inputs(cfg: RuntimeConfig) -> None:
    if not cfg.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {cfg.input_path}")
    if cfg.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if cfg.chunk_length_sec <= 0:
        raise ValueError("--chunk-length-sec must be > 0")
    if cfg.chunk_overlap_sec < 0:
        raise ValueError("--chunk-overlap-sec must be >= 0")
    if cfg.chunk_overlap_sec >= cfg.chunk_length_sec:
        raise ValueError("--chunk-overlap-sec must be < --chunk-length-sec")
    if cfg.export_html and cfg.html_px_per_sec <= 0:
        raise ValueError("--html-px-per-sec must be > 0")
    if cfg.num_speakers is not None and cfg.num_speakers < 1:
        raise ValueError("--num-speakers must be >= 1 when set")


def run_pipeline(cfg: RuntimeConfig) -> int:
    pipeline_start = time.perf_counter()
    _validate_inputs(cfg)

    pipeline_timings: dict[str, float] = {}
    final_result = None

    t0 = time.perf_counter()
    ffmpeg_path = validate_ffmpeg_exists()
    pipeline_timings["validate_ffmpeg_seconds"] = time.perf_counter() - t0
    logger.debug("Using ffmpeg at %s", ffmpeg_path)

    with _work_dir(cfg.temp_dir, cfg.keep_temp_files) as work_dir:
        normalized_path = work_dir / "normalized.wav"
        t0 = time.perf_counter()
        normalize_to_wav_mono_16k(
            input_path=cfg.input_path,
            output_path=normalized_path,
            ffmpeg_path=ffmpeg_path,
        )
        pipeline_timings["normalize_audio_seconds"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if cfg.chunked:
            duration = wav_duration_seconds(normalized_path) or 0.0
            chunks = materialize_chunks(
                ffmpeg_path=ffmpeg_path,
                normalized_wav_path=normalized_path,
                duration_sec=float(duration),
                chunk_length_sec=cfg.chunk_length_sec,
                chunk_overlap_sec=cfg.chunk_overlap_sec,
                chunks_dir=work_dir / "chunks",
            )
            result = transcribe_and_align_chunked(
                chunks=chunks,
                source_path=cfg.input_path,
                normalized_audio_path=normalized_path,
                model_name=cfg.model_name,
                device=cfg.device,
                compute_type=cfg.compute_type,
                language=cfg.language,
                batch_size=cfg.batch_size,
            )
        else:
            result = transcribe_and_align(
                audio_path=normalized_path,
                source_path=cfg.input_path,
                model_name=cfg.model_name,
                device=cfg.device,
                compute_type=cfg.compute_type,
                language=cfg.language,
                batch_size=cfg.batch_size,
            )
        _ = time.perf_counter() - t0  # covered by leaf timings from transcription module

        t0 = time.perf_counter()
        duration = wav_duration_seconds(normalized_path)
        pipeline_timings["measure_duration_seconds"] = time.perf_counter() - t0
        # Merge pipeline-level timings into the result extras.
        merged_extras = dict(result.extras or {})
        runtime_stats = dict(merged_extras.get("runtime_stats") or {})
        runtime_stats.update(pipeline_timings)
        merged_extras["runtime_stats"] = runtime_stats

        # Optional diarization (post-process on global timeline).
        if cfg.enable_diarization:
            resolved = resolve_device(cfg.device)
            token = resolve_hf_token(cfg.hf_token)
            if not token:
                raise RuntimeError(
                    "Diarization enabled but no Hugging Face token provided. "
                    "Use --hf-token or set HF_TOKEN/HUGGINGFACE_TOKEN env var."
                )
            t0 = time.perf_counter()
            turns = run_whisperx_diarization(
                audio_path=normalized_path,
                device=resolved,
                hf_token=token,
                min_speakers=cfg.min_speakers,
                max_speakers=cfg.max_speakers,
                num_speakers=cfg.num_speakers,
            )
            pipeline_timings["run_diarization_seconds"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            diarized_segments = assign_speakers(segments=result.segments, turns=turns, assign_words=True)
            pipeline_timings["assign_speakers_seconds"] = time.perf_counter() - t0
            merged_extras["diarization_enabled"] = True
            merged_extras["diarization_method"] = "whisperx"
            merged_extras["speakers"] = [t.to_dict() for t in turns]
            result = type(result)(
                source_file=result.source_file,
                normalized_audio_file=result.normalized_audio_file,
                model=result.model,
                language=result.language,
                duration_seconds=result.duration_seconds,
                full_text=result.full_text,
                segments=diarized_segments,
                extras=merged_extras,
            )
        else:
            merged_extras["diarization_enabled"] = False
            merged_extras["diarization_method"] = None

        result = type(result)(
            source_file=result.source_file,
            normalized_audio_file=result.normalized_audio_file,
            model=result.model,
            language=result.language,
            duration_seconds=duration,
            full_text=result.full_text,
            segments=result.segments,
            extras=merged_extras,
        )

        # First write to measure elapsed time, then write again with the timing
        # embedded in the output for easy auditing.
        pipeline_timings["write_json_seconds"] = write_json(result, cfg.output_path)
        merged_extras = dict(result.extras or {})
        runtime_stats = dict(merged_extras.get("runtime_stats") or {})
        runtime_stats["write_json_seconds"] = pipeline_timings["write_json_seconds"]
        merged_extras["runtime_stats"] = runtime_stats
        result = type(result)(
            source_file=result.source_file,
            normalized_audio_file=result.normalized_audio_file,
            model=result.model,
            language=result.language,
            duration_seconds=result.duration_seconds,
            full_text=result.full_text,
            segments=result.segments,
            extras=merged_extras,
        )
        write_json(result, cfg.output_path)
        logger.info("Wrote JSON to %s", cfg.output_path)

        if cfg.export_html:
            from .timeline_exporter import export_timeline_from_json_file

            html_path = cfg.html_output_path
            if html_path is None:
                html_path = cfg.output_path.parent / f"{cfg.output_path.stem}_timeline.html"
            html_path = html_path.resolve()
            t_html = time.perf_counter()
            export_timeline_from_json_file(
                cfg.output_path.resolve(),
                html_path,
                px_per_sec=cfg.html_px_per_sec,
                direction=cfg.html_direction,
            )
            pipeline_timings["export_html_seconds"] = time.perf_counter() - t_html
            logger.info(
                "Wrote HTML timeline to %s (%.3fs)",
                html_path,
                pipeline_timings["export_html_seconds"],
            )

        final_result = result

        if cfg.keep_temp_files:
            logger.info("Temp files kept in %s", work_dir)

    if final_result is None:
        raise RuntimeError("Pipeline produced no result.")

    pipeline_timings["total_pipeline_seconds"] = time.perf_counter() - pipeline_start
    merged_extras = dict(final_result.extras or {})
    runtime_stats = dict(merged_extras.get("runtime_stats") or {})
    runtime_stats.update(pipeline_timings)
    merged_extras["runtime_stats"] = runtime_stats
    final_result = type(final_result)(
        source_file=final_result.source_file,
        normalized_audio_file=final_result.normalized_audio_file,
        model=final_result.model,
        language=final_result.language,
        duration_seconds=final_result.duration_seconds,
        full_text=final_result.full_text,
        segments=final_result.segments,
        extras=merged_extras,
    )
    write_json(final_result, cfg.output_path)

    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.verbose)
    _ensure_supported_python()

    compute_type = args.compute_type
    if compute_type is None:
        resolved_for_default = args.device
        if resolved_for_default == "auto":
            try:
                import torch

                resolved_for_default = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                resolved_for_default = "cpu"
        compute_type = _default_compute_type("cuda" if resolved_for_default == "cuda" else "cpu")

    model_name = args.model or MODEL_PRESETS.get(args.model_preset, MODEL_PRESETS["original"])
    logger.info(
        "Selected model: %s (preset=%s, custom_override=%s)",
        model_name,
        args.model_preset,
        bool(args.model),
    )

    cfg = RuntimeConfig(
        input_path=args.input.resolve(),
        output_path=args.output.resolve(),
        model_name=model_name,
        device=args.device,
        compute_type=compute_type,
        language=args.language,
        batch_size=args.batch_size,
        chunked=bool(args.chunked),
        chunk_length_sec=float(args.chunk_length_sec),
        chunk_overlap_sec=float(args.chunk_overlap_sec),
        enable_diarization=bool(args.enable_diarization),
        hf_token=args.hf_token,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        num_speakers=args.num_speakers,
        temp_dir=args.temp_dir.resolve() if args.temp_dir else None,
        keep_temp_files=bool(args.keep_temp_files),
        verbose=bool(args.verbose),
        export_html=bool(args.export_html),
        html_output_path=args.html_output.resolve() if args.html_output else None,
        html_direction=str(args.html_direction),
        html_px_per_sec=float(args.html_px_per_sec),
    )

    try:
        return run_pipeline(cfg)
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

