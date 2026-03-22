from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class FfmpegNotFoundError(RuntimeError):
    pass


class FfmpegError(RuntimeError):
    pass


def validate_ffmpeg_exists() -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FfmpegNotFoundError(
            "ffmpeg not found on PATH. Install ffmpeg and ensure `ffmpeg` is available."
        )
    return Path(ffmpeg)


def normalize_to_wav_mono_16k(
    *,
    input_path: Path,
    output_path: Path,
    ffmpeg_path: Path | None = None,
) -> Path:
    """
    Convert any audio/video file to 16kHz mono PCM WAV, leaving the input untouched.
    """
    if ffmpeg_path is None:
        ffmpeg_path = validate_ffmpeg_exists()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(ffmpeg_path),
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]

    logger.info("Normalizing media with ffmpeg.")
    logger.debug("ffmpeg command: %s", " ".join(cmd))

    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
    except OSError as e:
        raise FfmpegError(f"Failed to execute ffmpeg: {e}") from e

    if completed.returncode != 0:
        raise FfmpegError(
            "ffmpeg failed with exit code "
            f"{completed.returncode}. stderr:\n{completed.stderr}"
        )

    if not output_path.exists():
        raise FfmpegError("ffmpeg reported success but output file was not created.")

    return output_path

