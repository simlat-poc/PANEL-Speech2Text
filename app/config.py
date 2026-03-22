from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    input_path: Path
    output_path: Path
    model_name: str
    device: str  # "auto" | "cpu" | "cuda"
    compute_type: str  # e.g. "float16", "int8"
    language: str | None
    batch_size: int
    chunked: bool
    chunk_length_sec: float
    chunk_overlap_sec: float
    enable_diarization: bool
    hf_token: str | None
    min_speakers: int | None
    max_speakers: int | None
    num_speakers: int | None  # forwarded as pyannote num_speakers when set (takes precedence over min/max)
    temp_dir: Path | None
    keep_temp_files: bool
    verbose: bool
    # Optional: write interactive HTML timeline after JSON (via app.timeline_exporter).
    export_html: bool
    html_output_path: Path | None
    html_direction: str
    html_px_per_sec: float

