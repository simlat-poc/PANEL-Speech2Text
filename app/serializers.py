from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .models import TranscriptionResult


def write_json(result: TranscriptionResult, output_path: Path) -> float:
    """
    Write JSON to disk and return elapsed seconds.
    """
    t0 = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = result.to_dict()
    output_path.write_text(
        # Keep insertion order so "full_text" stays near the top.
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return time.perf_counter() - t0

