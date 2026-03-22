from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .ffmpeg_utils import FfmpegError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioChunk:
    index: int
    start_sec: float
    end_sec: float
    core_start_sec: float
    core_end_sec: float
    path: Path


def plan_chunks(
    *,
    duration_sec: float,
    chunk_length_sec: float,
    chunk_overlap_sec: float,
) -> list[tuple[int, float, float]]:
    if duration_sec <= 0:
        return []
    if chunk_length_sec <= 0:
        raise ValueError("chunk_length_sec must be > 0")
    if chunk_overlap_sec < 0:
        raise ValueError("chunk_overlap_sec must be >= 0")
    if chunk_overlap_sec >= chunk_length_sec:
        raise ValueError("chunk_overlap_sec must be < chunk_length_sec")

    step = chunk_length_sec - chunk_overlap_sec
    chunks: list[tuple[int, float, float]] = []
    idx = 0
    start = 0.0
    while start < duration_sec:
        end = min(duration_sec, start + chunk_length_sec)
        chunks.append((idx, start, end))
        if end >= duration_sec:
            break
        idx += 1
        start += step
    return chunks


def extract_chunk_wav(
    *,
    ffmpeg_path: Path,
    input_wav_path: Path,
    output_wav_path: Path,
    start_sec: float,
    end_sec: float,
) -> Path:
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ffmpeg_path),
        "-hide_banner",
        "-nostdin",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-to",
        f"{end_sec:.3f}",
        "-i",
        str(input_wav_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_wav_path),
    ]
    logger.debug("Extracting chunk %0.3f-%0.3f to %s", start_sec, end_sec, output_wav_path)
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise FfmpegError(
            f"ffmpeg chunk extract failed ({start_sec:.3f}-{end_sec:.3f}s). stderr:\n{completed.stderr}"
        )
    return output_wav_path


def materialize_chunks(
    *,
    ffmpeg_path: Path,
    normalized_wav_path: Path,
    duration_sec: float,
    chunk_length_sec: float,
    chunk_overlap_sec: float,
    chunks_dir: Path,
) -> list[AudioChunk]:
    plan = plan_chunks(
        duration_sec=duration_sec,
        chunk_length_sec=chunk_length_sec,
        chunk_overlap_sec=chunk_overlap_sec,
    )
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Trusted "core" region: split overlap in half between adjacent chunks.
    half_overlap = chunk_overlap_sec / 2.0
    out: list[AudioChunk] = []
    for i, start, end in plan:
        core_start = start + (0.0 if i == 0 else half_overlap)
        core_end = end - (0.0 if end >= duration_sec else half_overlap)
        path = chunks_dir / f"chunk_{i:04d}.wav"
        extract_chunk_wav(
            ffmpeg_path=ffmpeg_path,
            input_wav_path=normalized_wav_path,
            output_wav_path=path,
            start_sec=start,
            end_sec=end,
        )
        out.append(
            AudioChunk(
                index=i,
                start_sec=start,
                end_sec=end,
                core_start_sec=core_start,
                core_end_sec=core_end,
                path=path,
            )
        )
    return out

