# WhisperX Offline Transcription (file-based)

This project is an **offline / post-processing** transcription pipeline:

- Accepts audio/video files (`wav`, `mp3`, `m4a`, `mp4`, `mov`, `mkv`, etc.)
- Normalizes input to **mono 16 kHz PCM WAV** using **ffmpeg**
- Runs **WhisperX** transcription + **alignment** for **word-level timestamps**
- Optional **speaker diarization** via **PyAnnote** (WhisperX `DiarizationPipeline`), assigning speakers to segments and words in the JSON
- Writes a stable JSON output

## Requirements

- **Python 3.11 or 3.12** (recommended: 3.12)
  - Python **3.14 is not supported** by WhisperX dependencies yet (you’ll see install errors like `ctranslate2` incompatibilities).
- **ffmpeg** installed and available on `PATH`
- **Diarization:** a [Hugging Face](https://huggingface.co) account and **access token**, and you must **accept the model license** for the diarization model you use (see below)

Check ffmpeg:

```powershell
ffmpeg -version
```

## Install (Windows / PowerShell)

From the project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python --version
pip install -U pip
pip install -r requirements.txt
```

## Run

```powershell
python -m app.main --input "C:\path\to\file.mp4" --output "C:\path\to\out.json" --device auto --model-preset original
```

Compare Hebrew model preset:

```powershell
python -m app.main --input "C:\path\to\file.mp4" --output "C:\path\to\out_ivrit.json" --device auto --model-preset ivrit-he
```

Use any custom model id directly (overrides preset):

```powershell
python -m app.main --input "C:\path\to\file.mp4" --output "C:\path\to\out_custom.json" --model "ivrit-ai/whisper-large-v3-ct2"
```

Write **JSON and the interactive HTML timeline** in one command:

```powershell
python -m app.main --input "C:\path\to\file.mp4" --output "C:\path\to\out.json" --export-html
```

Optional HTML paths and exporter options:

```powershell
python -m app.main --input "file.mp3" --output "out.json" --export-html --html-output "C:\path\to\custom.html"
python -m app.main --input "file.mp3" --output "out.json" --export-html --html-direction auto --html-px-per-sec 10
```

See all flags:

```powershell
python -m app.main --help
```

## Speaker diarization (PyAnnote)

When you pass **`--enable-diarization`**, the pipeline runs WhisperX’s **`DiarizationPipeline`** (PyAnnote) on the normalized audio and merges speaker labels into each segment and word in the JSON.

- **Model (this repo):** [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) (see `app/diarization.py`). On Hugging Face, open the model page and **accept the terms** so your token can download it.
- **Token:** use **`--hf-token`** or set one of **`HF_TOKEN`**, **`HUGGINGFACE_TOKEN`**, or **`HUGGINGFACEHUB_API_TOKEN`** in the environment. **Do not commit tokens** to git.
- **Speaker count hints (optional):** `--min-speakers`, `--max-speakers`, and **`--num-speakers`** are passed through to PyAnnote; they are **hints**, not a strict guarantee of how many speaker labels appear in the output.

Example:

```powershell
python -m app.main --input "file.mp4" --output "out.json" --enable-diarization --hf-token "hf_..." --export-html
```

Or with env var (PowerShell):

```powershell
$env:HF_TOKEN = "hf_..."
python -m app.main --input "file.mp4" --output "out.json" --enable-diarization
```

## Voiceprint identification (optional)

You can add identity matching on top of diarization:

- Enroll known speakers from reference files (`--voiceprint-enroll NAME=PATH`)
- Store voiceprints in a local DB (`--voiceprint-db-dir`, default: `.voiceprints`)
- Match diarized speakers to enrolled voiceprints using cosine similarity
- Keep generic labels when confidence is below threshold

Example:

```powershell
python -m app.main --input "file.mp3" --output "out.json" --enable-diarization --enable-voiceprint --voiceprint-enroll "Alice=C:\refs\alice.wav" --voiceprint-enroll "Bob=C:\refs\bob.wav" --voiceprint-threshold 0.72
```

## Export HTML Timeline (demo artifact)

This exporter converts your pipeline JSON output into a single self-contained HTML file with:
- A horizontal timeline with one lane per speaker
- Segment blocks positioned by timestamps
- Clickable blocks that show word-level timestamps in a details panel
- RTL text support for Hebrew
- **Zoom** (wider range, up to **200 px/s**) and a **Text** slider (about **50%–165%**) to scale transcript size
- **Drag the bar** under the timeline to resize its height (saved in `localStorage` for that browser/file origin)
- **Play / Stop**: auto-scroll the timeline in real time (1s of session = 1s wall clock at the current zoom); zoom/text sliders are locked while playing

Export:

```powershell
python -m app.timeline_exporter --input "Outputs\a3en-2speakers-small-diar.json" --output "Outputs\output_timeline.html"
```

Common options:

```powershell
python -m app.timeline_exporter --input out.json --output output_timeline.html --direction auto
python -m app.timeline_exporter --input out.json --output output_timeline.html --rtl-text-only
python -m app.timeline_exporter --input out.json --output output_timeline.html --rtl-timeline
python -m app.timeline_exporter --input out.json --output output_timeline.html --px-per-sec 10
```

