# WhisperX Offline Transcription (file-based)

This project is an **offline / post-processing** transcription pipeline:

- Accepts audio/video files (`wav`, `mp3`, `m4a`, `mp4`, `mov`, `mkv`, etc.)
- Normalizes input to **mono 16 kHz PCM WAV** using **ffmpeg**
- Runs **WhisperX** transcription + **alignment** for **word-level timestamps**
- Writes a stable JSON output (easy to extend later with diarization fields)

## Requirements

- **Python 3.11 or 3.12** (recommended: 3.12)
  - Python **3.14 is not supported** by WhisperX dependencies yet (you’ll see install errors like `ctranslate2` incompatibilities).
- **ffmpeg** installed and available on `PATH`

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
python -m app.main --input "C:\path\to\file.mp4" --output "C:\path\to\out.json" --device auto --model large-v3
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

