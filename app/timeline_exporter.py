from __future__ import annotations

import argparse
import html as html_lib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict


class TimelineWord(TypedDict, total=False):
    word: str
    start: float
    end: float
    speaker: str


class TimelineSegment(TypedDict, total=False):
    id: int
    lane_id: int
    speaker: str
    start_sec: float
    end_sec: float
    left_px_ltr: int
    width_px: int
    display_text: str
    full_text: str
    words: list[TimelineWord]


class TimelineModel(TypedDict):
    session_title: str
    language: str
    duration_seconds: float
    speakers_count: int
    segments_count: int
    overlap_events_count: int
    lanes: list[dict[str, Any]]
    segments: list[TimelineSegment]
    px_per_sec: float
    lane_height_px: int
    timeline_padding_left_px: int
    timeline_padding_right_px: int
    axis_height_px: int
    rtl_timeline: bool
    rtl_text_only: bool
    major_tick_step_seconds: float


def load_result_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _detect_language(result: dict[str, Any]) -> str:
    # Common fields in WhisperX output.
    language = result.get("language")
    if isinstance(language, str) and language.strip():
        return language.strip()

    # Fallback to nested metadata.
    meta = result.get("metadata") or {}
    if isinstance(meta, dict):
        lang = meta.get("language")
        if isinstance(lang, str) and lang.strip():
            return lang.strip()
    return "unknown"


def _language_is_hebrew(language: str) -> bool:
    l = (language or "").lower().strip()
    if not l or l == "unknown":
        return False
    if l in {"he", "iw", "he-il", "he_il", "hebrew", "עברית"}:
        return True
    # Whisper / locale-style codes: he_IL, he-IL, etc.
    return l.startswith("he-") or l.startswith("he_") or l.startswith("iw-") or l.startswith("iw_")


def _get_segments(result: dict[str, Any]) -> list[dict[str, Any]]:
    # Your older requirement mentioned transcript_segments / diarization_segments,
    # but the current pipeline emits top-level "segments".
    if isinstance(result.get("transcript_segments"), list):
        return [s for s in result["transcript_segments"] if isinstance(s, dict)]
    if isinstance(result.get("diarization_segments"), list):
        return [s for s in result["diarization_segments"] if isinstance(s, dict)]
    if isinstance(result.get("segments"), list):
        return [s for s in result["segments"] if isinstance(s, dict)]
    return []


def _transcript_suggests_hebrew(result: dict[str, Any]) -> bool:
    """When JSON ``language`` is missing, infer Hebrew from Unicode letters in the transcript."""
    parts: list[str] = []
    ft = result.get("full_text")
    if isinstance(ft, str) and ft.strip():
        parts.append(ft)
    for s in _get_segments(result)[:80]:
        t = s.get("text")
        if isinstance(t, str) and t.strip():
            parts.append(t)
    blob = "\n".join(parts)
    if not blob.strip():
        return False
    hebrew = sum(1 for c in blob if "\u0590" <= c <= "\u05FF")
    if hebrew < 10:
        return False
    letters = sum(1 for c in blob if c.isalpha() or ("\u0590" <= c <= "\u05FF"))
    if letters == 0:
        return hebrew >= 10
    return (hebrew / letters) >= 0.3


def _extract_speaker_name(segment: dict[str, Any]) -> str:
    # Prefer resolved speaker names when present.
    for key in ("resolved_speaker", "diarized_speaker", "speaker"):
        v = segment.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "UNKNOWN"


def _extract_words(segment: dict[str, Any]) -> list[TimelineWord]:
    words_raw = segment.get("words")
    if not isinstance(words_raw, list):
        return []

    out: list[TimelineWord] = []
    for w in words_raw:
        if not isinstance(w, dict):
            continue
        word_text = w.get("word")
        start = w.get("start")
        end = w.get("end")
        if not isinstance(word_text, str) or word_text.strip() == "":
            continue

        # words can be missing timing in some formats; keep only what we have.
        word: TimelineWord = {"word": word_text.strip()}
        if isinstance(start, (int, float)):
            word["start"] = float(start)
        if isinstance(end, (int, float)):
            word["end"] = float(end)

        speaker = w.get("speaker")
        if isinstance(speaker, str) and speaker.strip():
            word["speaker"] = speaker.strip()
        out.append(word)
    return out


def _segment_overlap_hint(segment: dict[str, Any]) -> Any:
    # Your JSON might include an "overlap" field; if absent, overlap events will be 0.
    return segment.get("overlap")


def build_timeline_model(result: dict[str, Any], *, rtl_timeline: bool, rtl_text_only: bool, px_per_sec: float) -> TimelineModel:
    language = _detect_language(result)
    segments_raw = _get_segments(result)

    # Duration: prefer top-level; else max segment end.
    duration = result.get("duration_seconds")
    if not isinstance(duration, (int, float)):
        duration = None
    if duration is None:
        ends: list[float] = []
        for s in segments_raw:
            end = s.get("end")
            if isinstance(end, (int, float)):
                ends.append(float(end))
        duration = max(ends) if ends else 0.0

    duration_sec = float(duration or 0.0)
    duration_sec = max(0.0, duration_sec)

    # Speaker lanes: keep stable order by first appearance.
    speaker_to_lane: dict[str, int] = {}
    lanes: list[dict[str, Any]] = []
    segments: list[TimelineSegment] = []

    for idx, seg in enumerate(segments_raw):
        if not isinstance(seg, dict):
            continue
        start = seg.get("start")
        end = seg.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue

        start_sec = float(start)
        end_sec = float(end)
        if end_sec <= start_sec:
            continue

        speaker = _extract_speaker_name(seg)
        if speaker not in speaker_to_lane:
            lane_id = len(speaker_to_lane)
            speaker_to_lane[speaker] = lane_id
            lanes.append({"lane_id": lane_id, "speaker": speaker})
        lane_id = speaker_to_lane[speaker]

        full_text = seg.get("text")
        if not isinstance(full_text, str):
            full_text = ""
        full_text = full_text.strip()
        display_text = full_text
        if len(display_text) > 220:
            display_text = display_text[:217] + "..."

        words = _extract_words(seg)
        # If the segment has speaker but words are missing speaker labels, fill them in.
        if speaker and words:
            for w in words:
                if "speaker" not in w:
                    w["speaker"] = speaker

        left_px_ltr = int(math.floor(start_sec * px_per_sec))
        # Keep blocks at least 2px wide so they remain visible.
        width_px = int(max(2, math.floor((end_sec - start_sec) * px_per_sec)))

        segments.append(
            {
                "id": idx,
                "lane_id": lane_id,
                "speaker": speaker,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "left_px_ltr": left_px_ltr,
                "width_px": width_px,
                "display_text": display_text,
                "full_text": full_text,
                "words": words,
            }
        )

    speakers_count = len(lanes) if lanes else 0
    segments_count = len(segments)

    overlap_events_count = 0
    for seg in segments_raw:
        if not isinstance(seg, dict):
            continue
        hint = _segment_overlap_hint(seg)
        if hint is None:
            continue
        # Treat truthy values as an overlap event marker.
        if isinstance(hint, bool) and hint:
            overlap_events_count += 1
        elif isinstance(hint, (int, float)) and float(hint) != 0.0:
            overlap_events_count += 1
        elif isinstance(hint, str) and hint.strip():
            overlap_events_count += 1

    source_file = result.get("source_file") or result.get("input") or result.get("file") or "session"
    if isinstance(source_file, str) and source_file.strip():
        session_title = Path(source_file).name
    else:
        session_title = "session"

    duration_seconds = float(duration_sec or 0.0)

    # Major tick spacing on the time ruler (seconds between labels).
    # Dense 5s grid up to 30 min; coarser beyond to limit DOM size.
    if duration_seconds <= 1800:
        major_tick_step_seconds = 5
    elif duration_seconds <= 7200:
        major_tick_step_seconds = 30
    else:
        major_tick_step_seconds = 60

    return {
        "session_title": session_title,
        "language": language,
        "duration_seconds": duration_seconds,
        "speakers_count": speakers_count,
        "segments_count": segments_count,
        "overlap_events_count": overlap_events_count,
        "lanes": lanes,
        "segments": segments,
        "px_per_sec": float(px_per_sec),
        "lane_height_px": 144,
        "timeline_padding_left_px": 92,
        "timeline_padding_right_px": 22,
        "axis_height_px": 42,
        "rtl_timeline": bool(rtl_timeline),
        "rtl_text_only": bool(rtl_text_only),
        "major_tick_step_seconds": float(major_tick_step_seconds),
    }


def render_timeline_html(model: TimelineModel, *, rtl: bool = False) -> str:
    # rtl argument is kept for compatibility with the suggested signature.
    # We prefer model["rtl_timeline"] and model["rtl_text_only"].
    rtl_timeline = bool(model.get("rtl_timeline", rtl))
    rtl_text_only = bool(model.get("rtl_text_only", False))

    # A restrained demo palette; cycle if speakers exceed palette length.
    palette = [
        "#4F46E5",  # indigo
        "#059669",  # teal
        "#DC2626",  # red
        "#D97706",  # amber
        "#2563EB",  # blue
        "#7C3AED",  # violet
        "#0891B2",  # cyan
        "#16A34A",  # green
        "#EA580C",  # orange
        "#0F172A",  # slate
    ]

    speakers = model["lanes"]
    speaker_colors: dict[str, str] = {}
    for i, lane in enumerate(speakers):
        speaker_colors[str(lane["speaker"])] = palette[i % len(palette)]

    # Embed model data as JSON for JS to render.
    # Important: model["segments"] can be large; keep it to what is needed for rendering + details panel.
    model_for_js = {
        "session_title": model["session_title"],
        "language": model["language"],
        "duration_seconds": model["duration_seconds"],
        "speakers_count": model["speakers_count"],
        "segments_count": model["segments_count"],
        "overlap_events_count": model["overlap_events_count"],
        "rtl_timeline": rtl_timeline,
        "rtl_text_only": rtl_text_only,
        "lanes": model["lanes"],
        "segments": model["segments"],
        "px_per_sec": model["px_per_sec"],
        "lane_height_px": model["lane_height_px"],
        "timeline_padding_left_px": model["timeline_padding_left_px"],
        "timeline_padding_right_px": model["timeline_padding_right_px"],
        "axis_height_px": model["axis_height_px"],
        "major_tick_step_seconds": model["major_tick_step_seconds"],
        "speaker_colors": speaker_colors,
    }

    duration_width_px_ltr = int(math.ceil(model["duration_seconds"] * float(model["px_per_sec"])))
    timeline_total_width_px = max(320, duration_width_px_ltr) + int(model["timeline_padding_left_px"]) + int(model["timeline_padding_right_px"])

    # Note: we keep everything inline (no external assets).
    escaped_css_dir = "rtl" if rtl_text_only else "ltr"
    html_dir = "rtl" if rtl_timeline else "ltr"

    # Major ticks are generated in JS, but we precompute the step.
    js_model_json = json.dumps(model_for_js, ensure_ascii=False)

    # We escape only in places we interpolate into HTML directly (e.g. aria labels); the rest is JS-rendered.
    rtl_text_preview_css_direction = "rtl" if rtl_text_only else "ltr"

    return f"""<!doctype html>
<html lang="{html_lib.escape(model['language'] or 'en')}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Transcript Timeline - {html_lib.escape(model['session_title'])}</title>
  <style>
    :root {{
      --bg: #0b1220;
      --panel: rgba(255,255,255,0.06);
      --panel2: rgba(255,255,255,0.08);
      --text: rgba(255,255,255,0.92);
      --muted: rgba(255,255,255,0.70);
      --border: rgba(255,255,255,0.14);
      --shadow: rgba(0,0,0,0.35);
      --font: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Noto Sans", "Liberation Sans", sans-serif;
    }}

    html, body {{
      height: 100%;
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      margin: 0;
    }}

    .app {{
      padding: 14px 16px 26px 16px;
      direction: {html_dir};
    }}

    .topbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: stretch;
    }}

    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: 0 10px 24px var(--shadow);
      padding: 12px 14px;
    }}

    .summary {{
      flex: 1 1 420px;
      min-width: 320px;
    }}
    .summary h1 {{
      font-size: 16px;
      margin: 0 0 8px 0;
      font-weight: 700;
      letter-spacing: 0.2px;
    }}
    .summary .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
    }}
    .kv {{
      padding: 10px 10px;
      background: var(--panel2);
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 12px;
    }}
    .kv .k {{
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .kv .v {{
      font-size: 14px;
      font-weight: 700;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}

    .legend {{
      flex: 0 0 320px;
      min-width: 280px;
    }}
    .legend h2 {{
      margin: 0 0 10px 0;
      font-size: 14px;
      letter-spacing: 0.2px;
    }}
    .legend .items {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
      align-items: center;
    }}
    .legend .item {{
      display: flex;
      gap: 8px;
      align-items: center;
      font-size: 13px;
      color: var(--muted);
    }}
    .dot {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      background: #999;
      border: 1px solid rgba(255,255,255,0.22);
      box-shadow: 0 0 0 2px rgba(255,255,255,0.06);
    }}

    .controls {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }}

    .btn {{
      padding: 8px 10px;
      background: rgba(255,255,255,0.10);
      border: 1px solid rgba(255,255,255,0.16);
      color: var(--text);
      border-radius: 12px;
      cursor: pointer;
      user-select: none;
      font-weight: 600;
      font-size: 13px;
    }}
    .btn:active {{ transform: translateY(1px); }}
    .btn:disabled {{
      opacity: 0.45;
      cursor: not-allowed;
    }}
    input[type="range"]:disabled {{
      opacity: 0.5;
      cursor: not-allowed;
    }}

    .zoom {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 8px 10px;
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 12px;
      font-size: 13px;
      color: var(--muted);
    }}
    input[type="range"] {{
      width: 220px;
    }}
    input[type="range"].text-scale {{
      width: 140px;
    }}
    .zoom .val {{
      font-weight: 800;
      color: var(--text);
    }}

    .main {{
      display: grid;
      grid-template-columns: 1fr 320px;
      gap: 12px;
      margin-top: 12px;
      align-items: start;
    }}

    @media (max-width: 980px) {{
      .main {{
        grid-template-columns: 1fr;
      }}
    }}

    .timeline-card {{
      grid-column: 1 / 2;
      padding: 0;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      min-height: 0;
    }}

    .timeline-header {{
      display: flex;
      justify-content: flex-end;
      align-items: center;
      padding: 12px 14px;
      border-bottom: 1px solid rgba(255,255,255,0.10);
      background: rgba(255,255,255,0.04);
    }}
    .timeline-frame {{
      display: flex;
      flex-direction: column;
      flex: 1;
      min-height: 0;
    }}

    .timeline-scroll {{
      overflow: auto;
      min-height: 220px;
      /* Horizontal scroll must stay LTR so scrollLeft 0..max works the same in all browsers
         (html may be dir=rtl for Hebrew; mirrored timeline layout is done via block positions). */
      direction: ltr;
      /* Keep scrollbars on the correct physical edge when page is RTL */
      unicode-bidi: isolate;
      /* max-height set by JS (resize handle) */
    }}

    .timeline-resize {{
      position: relative;
      flex: 0 0 auto;
      height: 16px;
      cursor: ns-resize;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(255,255,255,0.05);
      border-top: 1px solid rgba(255,255,255,0.12);
      user-select: none;
      touch-action: none;
    }}
    .timeline-resize:hover {{
      background: rgba(255,255,255,0.09);
    }}
    .timeline-resize:active {{
      background: rgba(99,102,241,0.25);
    }}
    .timeline-resize-grip {{
      width: 56px;
      height: 5px;
      border-radius: 999px;
      background: rgba(255,255,255,0.38);
      pointer-events: none;
    }}

    .timeline-inner {{
      position: relative;
      width: {timeline_total_width_px}px;
      min-height: {int(model["axis_height_px"] + model["lane_height_px"] * max(1, model["speakers_count"]))}px;
      background: linear-gradient(to bottom, rgba(255,255,255,0.03), rgba(255,255,255,0));
    }}

    .time-axis {{
      position: absolute;
      left: 0;
      right: 0;
      height: {model["axis_height_px"]}px;
      top: 0;
      border-bottom: 1px solid rgba(255,255,255,0.10);
      background: rgba(255,255,255,0.03);
    }}

    .ticks {{
      position: absolute;
      top: 0;
      left: {model['timeline_padding_left_px']}px;
      right: {model['timeline_padding_right_px']}px;
      height: {model['axis_height_px']}px;
      pointer-events: none;
    }}

    .tick {{
      position: absolute;
      top: 0;
      width: 1px;
      height: 100%;
      background: rgba(255,255,255,0.10);
    }}
    .tick .label {{
      position: absolute;
      top: 8px;
      transform: translateX(-50%);
      font-size: 12px;
      color: rgba(255,255,255,0.70);
      white-space: nowrap;
      font-weight: 700;
    }}

    .lanes {{
      position: absolute;
      left: 0;
      right: 0;
      top: {model["axis_height_px"]}px;
    }}

    .lane {{
      position: relative;
      /* height set in JS (Text scale slider) */
      border-bottom: 1px dashed rgba(255,255,255,0.10);
      display: flex;
      align-items: center;
      flex-direction: row;
    }}

    /* RTL timeline: speaker names / lane column on the RIGHT (blocks on the left). */
    .timeline-inner.timeline-rtl .lane-label {{
      order: 2;
      padding-left: 0;
      padding-right: 12px;
      text-align: right;
    }}
    .timeline-inner.timeline-rtl .lane-blocks {{
      order: 1;
      flex: 1;
      min-width: 0;
    }}

    .lane-label {{
      flex: 0 0 auto;
      width: {model["timeline_padding_left_px"]}px;
      padding-left: 12px;
      font-size: 15px;
      font-weight: 800;
      color: rgba(255,255,255,0.80);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}

    .lane-label .small {{
      display: block;
      font-weight: 700;
      color: rgba(255,255,255,0.60);
      font-size: 12px;
      margin-top: 2px;
    }}

    .block {{
      position: absolute;
      /* top/height set in JS (Text scale slider) */
      /* width from JS = timestamp span in px; border-box so padding/border don't extend past that */
      box-sizing: border-box;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.14);
      box-shadow: 0 8px 18px rgba(0,0,0,0.22);
      padding: 12px 14px;
      overflow: hidden;
      cursor: pointer;
      display: flex;
      align-items: flex-start;
      justify-content: flex-start;
    }}

    /* RTL timeline (page/hebrew): transcript starts from top-right of each segment box */
    .timeline-inner.timeline-rtl .block {{
      justify-content: flex-end;
    }}

    .segment-text {{
      font-size: 19px;
      font-weight: 600;
      line-height: 1.42;
      overflow: hidden;
      text-overflow: ellipsis;
      display: -webkit-box;
      -webkit-line-clamp: 7;
      -webkit-box-orient: vertical;
      white-space: normal;
      direction: {rtl_text_preview_css_direction};
      unicode-bidi: plaintext;
      text-align: left;
      width: 100%;
      min-width: 0;
      box-sizing: border-box;
    }}

    .timeline-inner.timeline-rtl .segment-text {{
      direction: rtl;
      text-align: right;
    }}

    /* RTL text with LTR timeline: right-align and RTL flow inside boxes */
    .segment-text.rtl-timeline {{
      text-align: right;
      direction: rtl;
    }}

    .details-card {{
      padding: 12px 14px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: 0 10px 24px var(--shadow);
    }}

    .details-card h2 {{
      margin: 0 0 10px 0;
      font-size: 14px;
      letter-spacing: 0.2px;
    }}

    .details-muted {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}

    .details-block {{
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 12px;
      padding: 10px 12px;
      margin-top: 10px;
    }}

    .details-title {{
      font-size: 13px;
      font-weight: 900;
      margin-bottom: 6px;
      color: rgba(255,255,255,0.92);
    }}

    .details-text {{
      font-size: 13px;
      color: rgba(255,255,255,0.86);
      line-height: 1.35;
      margin-bottom: 10px;
      direction: {rtl_text_preview_css_direction};
      unicode-bidi: plaintext;
    }}

    .word-list {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 12px;
      color: rgba(255,255,255,0.78);
      max-height: 220px;
      overflow: auto;
      padding-right: 6px;
      white-space: pre-wrap;
    }}
    .word-list .w {{
      margin-bottom: 4px;
    }}
  </style>
</head>
<body>
  <div class="app">
    <div class="topbar">
      <div class="card summary">
        <h1>{html_lib.escape(model['session_title'])} - Transcript Timeline</h1>
        <div class="grid">
          <div class="kv"><div class="k">Duration</div><div class="v">{model['duration_seconds']:.2f}s</div></div>
          <div class="kv"><div class="k">Language</div><div class="v">{html_lib.escape(model['language'] or '')}</div></div>
          <div class="kv"><div class="k">Speakers</div><div class="v">{model['speakers_count']}</div></div>
          <div class="kv"><div class="k">Segments</div><div class="v">{model['segments_count']}</div></div>
          <div class="kv"><div class="k">Overlap events</div><div class="v">{model['overlap_events_count']}</div></div>
          <div class="kv"><div class="k">View</div><div class="v">Segments on timeline</div></div>
          <div class="kv"><div class="k">Tip</div><div class="v">Click a block for word timestamps</div></div>
          <div class="kv"><div class="k">Mode</div><div class="v">{'RTL timeline' if rtl_timeline else ('RTL text' if rtl_text_only else 'LTR')}</div></div>
        </div>
      </div>

      <div class="card legend">
        <h2>Speaker Legend</h2>
        <div class="items" id="legendItems">
        </div>
      </div>
    </div>

    <div class="main">
      <div class="card timeline-card">
        <div class="timeline-header">
          <div class="controls">
            <div class="zoom">
              <span>Zoom</span>
              <input id="zoomRange" type="range" min="2" max="200" step="0.25" value="{model['px_per_sec']}"/>
              <span class="val" id="zoomVal">{model['px_per_sec']:.1f}px/s</span>
            </div>
            <div class="zoom">
              <span>Text</span>
              <input id="textScaleRange" class="text-scale" type="range" min="0.5" max="1.65" step="0.05" value="1.1"/>
              <span class="val" id="textScaleVal">110%</span>
            </div>
            <div class="btn" id="zoomReset" title="Reset zoom and text size to defaults">Reset</div>
            <div class="btn" id="playTimelineBtn" title="Scroll the timeline in real time (1s audio = 1s wall clock at current zoom)">Play</div>
            <div class="btn" id="stopTimelineBtn" title="Stop playback and return scroll to the start">Stop</div>
          </div>
        </div>

        <div class="timeline-frame">
          <div class="timeline-scroll" id="timelineScroll">
            <div class="timeline-inner{' timeline-rtl' if rtl_timeline else ''}" id="timelineInner">
              <div class="time-axis" id="timeAxis">
                <div class="ticks" id="ticks"></div>
              </div>
              <div class="lanes" id="lanes"></div>
            </div>
          </div>
          <div class="timeline-resize" id="timelineResize" title="Drag up or down to resize the timeline area (saved in this browser)">
            <div class="timeline-resize-grip" aria-hidden="true"></div>
          </div>
        </div>
      </div>

      <div class="details-card">
        <h2>Details</h2>
        <div class="details-muted" id="detailsMuted">
          Select a segment block on the timeline.
        </div>
        <div class="details-block" id="detailsBlock" style="display:none;">
          <div class="details-title" id="detailsTitle"></div>
          <div class="details-text" id="detailsText"></div>
          <div class="details-muted" id="detailsTimes"></div>
          <div class="word-list" id="wordList"></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const MODEL = {js_model_json};
    let TEXT_SCALE = 1.1;

    function formatZoomLabel(px) {{
      const n = parseFloat(px);
      if (!isFinite(n)) return '';
      const decimals = n < 10 ? 2 : 1;
      return n.toFixed(decimals) + ' px/s';
    }}

    function initTimelineResize() {{
      const scroll = document.getElementById('timelineScroll');
      const handle = document.getElementById('timelineResize');
      if (!scroll || !handle) return;

      const storageKey = 'whisperx_timeline_scroll_max_h';
      const minH = 200;

      function maxAllowed() {{
        return Math.max(minH + 120, window.innerHeight - 80);
      }}

      function clampH(h) {{
        return Math.round(Math.max(minH, Math.min(maxAllowed(), h)));
      }}

      function applyHeight(h) {{
        scroll.style.maxHeight = clampH(h) + 'px';
      }}

      function defaultHeight() {{
        return clampH(Math.round(window.innerHeight * 0.52));
      }}

      const saved = localStorage.getItem(storageKey);
      if (saved !== null) {{
        const n = parseInt(saved, 10);
        if (!isNaN(n)) applyHeight(n);
        else applyHeight(defaultHeight());
      }} else {{
        applyHeight(defaultHeight());
      }}

      window.addEventListener('resize', () => {{
        const cur = parseFloat(scroll.style.maxHeight);
        if (!isNaN(cur)) applyHeight(cur);
      }});

      let dragging = false;
      let startY = 0;
      let startH = 0;

      handle.addEventListener('pointerdown', (e) => {{
        if (e.pointerType === 'mouse' && e.button !== 0) return;
        dragging = true;
        startY = e.clientY;
        startH = scroll.getBoundingClientRect().height;
        try {{ handle.setPointerCapture(e.pointerId); }} catch (err) {{}}
        e.preventDefault();
      }});

      handle.addEventListener('pointermove', (e) => {{
        if (!dragging) return;
        const dy = e.clientY - startY;
        applyHeight(startH + dy);
      }});

      function endDrag(e) {{
        if (!dragging) return;
        dragging = false;
        try {{ handle.releasePointerCapture(e.pointerId); }} catch (err) {{}}
        const h = Math.round(scroll.getBoundingClientRect().height);
        localStorage.setItem(storageKey, String(h));
      }}

      handle.addEventListener('pointerup', endDrag);
      handle.addEventListener('pointercancel', endDrag);
    }}

    function formatTime(sec) {{
      if (typeof sec !== 'number' || !isFinite(sec)) return '';
      const s = sec;
      const mm = Math.floor(s / 60);
      const ss = s - mm * 60;
      // Keep 3 decimals for small-segment readability.
      return (mm > 0 ? String(mm).padStart(2,'0') + ':' : '') + ss.toFixed(3).padStart(mm > 0 ? 7 : 5, '0');
    }}

    function escapeHtml(str) {{
      if (str === null || str === undefined) return '';
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }}

    function majorTickStepSeconds(pxPerSec) {{
      // Step comes from the model (typically 5s for sessions up to 30 min).
      return MODEL.major_tick_step_seconds || 5;
    }}

    function truncate(str, n) {{
      const s = str || '';
      if (s.length <= n) return s;
      return s.slice(0, Math.max(0, n-3)) + '...';
    }}

    function pickSpeakerName(laneId) {{
      const lane = MODEL.lanes.find(l => l.lane_id === laneId);
      return lane ? lane.speaker : 'UNKNOWN';
    }}

    function buildLegend() {{
      const el = document.getElementById('legendItems');
      if (!el) return;
      el.innerHTML = '';
      for (const lane of MODEL.lanes) {{
        const speaker = lane.speaker;
        const color = (MODEL.speaker_colors && MODEL.speaker_colors[speaker]) || '#999';
        const item = document.createElement('div');
        item.className = 'item';
        const dot = document.createElement('div');
        dot.className = 'dot';
        dot.style.background = color;
        const name = document.createElement('div');
        name.textContent = speaker;
        name.title = speaker;
        item.appendChild(dot);
        item.appendChild(name);
        el.appendChild(item);
      }}
    }}

    function computeX(startSec, endSec, pxPerSec) {{
      const padL = MODEL.timeline_padding_left_px;
      const padR = MODEL.timeline_padding_right_px;
      const totalDataWidthPx = MODEL.duration_seconds * pxPerSec;
      const totalWidthPx = padL + totalDataWidthPx + padR;

      // LTR coordinate:
      const leftLtr = padL + startSec * pxPerSec;

      if (!MODEL.rtl_timeline) return leftLtr;

      // For RTL timeline, position blocks measured from the right edge of the inner timeline.
      // left = (padL + (totalDataWidthPx - endSec*pxPerSec))
      const leftRtl = padL + (totalDataWidthPx - endSec * pxPerSec);
      // Sanity clamp:
      if (leftRtl < padL) return padL;
      if (leftRtl > totalWidthPx) return totalWidthPx;
      return leftRtl;
    }}

    function computeWidth(startSec, endSec, pxPerSec) {{
      const w = (endSec - startSec) * pxPerSec;
      return Math.max(2, w);
    }}

    function clearChildren(el) {{
      while (el.firstChild) el.removeChild(el.firstChild);
    }}

    function buildTicks(pxPerSec) {{
      const ticksEl = document.getElementById('ticks');
      if (!ticksEl) return;
      clearChildren(ticksEl);

      const step = majorTickStepSeconds(pxPerSec);
      const duration = MODEL.duration_seconds || 0;
      const maxTicks = 2000;
      const tickCount = Math.min(maxTicks, Math.floor(duration / step) + 2);

      const dataW = duration * pxPerSec;
      for (let i = 0; i < tickCount; i++) {{
        const t = i * step;
        if (t > duration + 0.0001) break;
        const xData = MODEL.rtl_timeline ? (dataW - t * pxPerSec) : (t * pxPerSec);
        const tick = document.createElement('div');
        tick.className = 'tick';
        tick.style.left = xData + 'px';

        const label = document.createElement('div');
        label.className = 'label';
        label.textContent = (t < 60 ? t.toFixed(0) + 's' : (t/60).toFixed(1) + 'm');
        tick.appendChild(label);
        ticksEl.appendChild(tick);
      }}
    }}

    function buildLanesAndBlocks(pxPerSec) {{
      const lanesEl = document.getElementById('lanes');
      if (!lanesEl) return;
      clearChildren(lanesEl);

      const textScale = (typeof TEXT_SCALE === 'number' && isFinite(TEXT_SCALE)) ? TEXT_SCALE : 1;
      const laneHeight = Math.round(MODEL.lane_height_px * textScale);
      const segFontPx = Math.round(19 * textScale);
      const lineClamp = Math.max(4, Math.min(11, Math.round(7 * textScale)));
      const blockTop = 6;
      const blockHeight = Math.max(44, laneHeight - 12);

      const axisHeight = MODEL.axis_height_px;
      const speakers = MODEL.lanes.length;
      const inner = document.getElementById('timelineInner');

      const padL = MODEL.timeline_padding_left_px;
      const padR = MODEL.timeline_padding_right_px;
      const totalDataWidthPx = MODEL.duration_seconds * pxPerSec;
      const totalWidthPx = padL + totalDataWidthPx + padR;
      inner.style.width = totalWidthPx + 'px';

      const lanesTotalHeight = axisHeight + speakers * laneHeight;
      inner.style.minHeight = lanesTotalHeight + 'px';

      const segmentByLane = new Map();
      for (const seg of MODEL.segments) {{
        const laneId = seg.lane_id;
        if (!segmentByLane.has(laneId)) segmentByLane.set(laneId, []);
        segmentByLane.get(laneId).push(seg);
      }}

      for (const lane of MODEL.lanes) {{
        const laneId = lane.lane_id;
        const laneRow = document.createElement('div');
        laneRow.className = 'lane';
        laneRow.style.top = '0px';

        const label = document.createElement('div');
        label.className = 'lane-label';
        label.textContent = lane.speaker || 'UNKNOWN';

        const small = document.createElement('span');
        small.className = 'small';
        small.textContent = 'lane ' + (laneId + 1);
        label.appendChild(small);
        laneRow.appendChild(label);

        // Blocks container for this lane (class used for RTL flex order).
        const blocks = document.createElement('div');
        blocks.className = 'lane-blocks';
        blocks.style.position = 'relative';
        blocks.style.flex = '1';
        blocks.style.height = laneHeight + 'px';

        const segs = segmentByLane.get(laneId) || [];
        for (const seg of segs) {{
          const block = document.createElement('div');
          block.className = 'block';
          const speakerColor = (MODEL.speaker_colors && MODEL.speaker_colors[seg.speaker]) || '#999';
          block.style.background = speakerColor + '1A'; // translucent
          block.style.borderColor = speakerColor + '66';
          const ax = computeX(seg.start_sec, seg.end_sec, pxPerSec);
          // LTR: blocks area starts after lane label (padL). RTL: label is on the right, blocks span from inner x=0.
          block.style.left = (MODEL.rtl_timeline ? ax : (ax - padL)) + 'px';
          block.style.width = computeWidth(seg.start_sec, seg.end_sec, pxPerSec) + 'px';

          const t = document.createElement('div');
          t.className = 'segment-text';
          t.textContent = seg.display_text || '';
          t.style.fontSize = segFontPx + 'px';
          t.style.webkitLineClamp = String(lineClamp);
          try {{ t.style.lineClamp = String(lineClamp); }} catch (e) {{}}
          if (MODEL.rtl_text_only || MODEL.rtl_timeline) {{
            t.classList.add('rtl-timeline');
          }}
          block.style.top = blockTop + 'px';
          block.style.height = blockHeight + 'px';
          block.appendChild(t);

          block.addEventListener('click', () => {{
            showDetails(seg);
          }});

          blocks.appendChild(block);
        }}

        laneRow.appendChild(blocks);
        laneRow.style.height = laneHeight + 'px';
        // Position the lane within the absolute lanes container.
        laneRow.style.position = 'absolute';
        laneRow.style.left = '0px';
        laneRow.style.right = '0px';
        laneRow.style.top = (laneId * laneHeight) + 'px';
        lanesEl.appendChild(laneRow);
      }}
    }}

    function showDetails(seg) {{
      const detailsMuted = document.getElementById('detailsMuted');
      const detailsBlock = document.getElementById('detailsBlock');
      const detailsTitle = document.getElementById('detailsTitle');
      const detailsText = document.getElementById('detailsText');
      const detailsTimes = document.getElementById('detailsTimes');
      const wordList = document.getElementById('wordList');

      if (!detailsMuted || !detailsBlock || !detailsTitle || !detailsText || !detailsTimes || !wordList) return;

      detailsMuted.style.display = 'none';
      detailsBlock.style.display = 'block';

      detailsTitle.textContent = (seg.speaker ? seg.speaker : 'UNKNOWN');
      detailsText.textContent = seg.full_text || seg.display_text || '';
      detailsTimes.textContent = 'Time: ' + formatTime(seg.start_sec) + ' - ' + formatTime(seg.end_sec);

      if (!seg.words || seg.words.length === 0) {{
        wordList.textContent = '(No word-level timestamps available for this segment.)';
        return;
      }}

      const lines = [];
      for (const w of seg.words) {{
        const st = (typeof w.start === 'number') ? w.start.toFixed(3) + 's' : '';
        const en = (typeof w.end === 'number') ? w.end.toFixed(3) + 's' : '';
        const time = st || en ? (st && en ? (st + '-' + en) : (st || en)) : '';
        const speaker = w.speaker ? (' (' + w.speaker + ')') : '';
        if (time) {{
          lines.push(w.word + ' ' + time + speaker);
        }} else {{
          lines.push(w.word + speaker);
        }}
      }}
      wordList.textContent = lines.join('\\n');
    }}

    function render(pxPerSec) {{
      buildTicks(pxPerSec);
      buildLanesAndBlocks(pxPerSec);
    }}

    // Init
    (function init() {{
      initTimelineResize();
      buildLegend();
      const zoomRange = document.getElementById('zoomRange');
      const zoomVal = document.getElementById('zoomVal');
      const zoomReset = document.getElementById('zoomReset');
      const textScaleRange = document.getElementById('textScaleRange');
      const textScaleVal = document.getElementById('textScaleVal');

      const defaultPx = MODEL.px_per_sec || 8;
      const defaultTextScale = 1.1;

      if (zoomRange) {{
        const mn = parseFloat(zoomRange.min);
        const mx = parseFloat(zoomRange.max);
        let v = parseFloat(zoomRange.value);
        if (!isFinite(v)) v = defaultPx;
        v = Math.min(mx, Math.max(mn, v));
        zoomRange.value = String(v);
      }}

      if (textScaleRange) {{
        TEXT_SCALE = parseFloat(textScaleRange.value) || defaultTextScale;
      }}

      function updateTextScaleLabel() {{
        if (textScaleVal) textScaleVal.textContent = Math.round(TEXT_SCALE * 100) + '%';
      }}
      updateTextScaleLabel();

      function apply(px) {{
        if (zoomVal) zoomVal.textContent = formatZoomLabel(px);
        render(px);
      }}

      if (zoomRange) {{
        zoomRange.addEventListener('input', () => {{
          const px = parseFloat(zoomRange.value);
          apply(px);
        }});
      }}
      if (textScaleRange) {{
        textScaleRange.addEventListener('input', () => {{
          TEXT_SCALE = parseFloat(textScaleRange.value) || defaultTextScale;
          updateTextScaleLabel();
          const px = parseFloat(zoomRange && zoomRange.value ? zoomRange.value : defaultPx);
          apply(px);
        }});
      }}
      if (zoomReset) {{
        zoomReset.addEventListener('click', () => {{
          if (zoomRange) zoomRange.value = String(defaultPx);
          if (textScaleRange) textScaleRange.value = String(defaultTextScale);
          TEXT_SCALE = defaultTextScale;
          updateTextScaleLabel();
          apply(defaultPx);
        }});
      }}

      // --- Real-time timeline scroll (Play / Stop) ---
      let playRafId = null;
      let isTimelinePlaying = false;
      let playWallStart = 0;
      let playTimeStart = 0;

      function scrollTimelineToTime(tSec, pxPerSec) {{
        const scroll = document.getElementById('timelineScroll');
        if (!scroll || !isFinite(pxPerSec) || pxPerSec <= 0) return;
        const padL = MODEL.timeline_padding_left_px;
        const lead = 80;
        const apply = () => {{
          const maxS = Math.max(0, scroll.scrollWidth - scroll.clientWidth);
          let left;
          if (MODEL.rtl_timeline) {{
            // Time 0 is on the right; play by scrolling left at exactly pxPerSec pixels per second.
            // (Using playhead-x minus margin always clamped to maxS kept the view stuck on the right
            // until late in the file when xTime-lead finally dropped below maxS.)
            left = maxS - tSec * pxPerSec;
          }} else {{
            const xTime = padL + tSec * pxPerSec;
            left = xTime - lead;
          }}
          left = Math.max(0, Math.min(maxS, left));
          scroll.scrollLeft = left;
        }};
        apply();
        requestAnimationFrame(apply);
      }}

      function setTimelineScrubLocked(locked) {{
        if (zoomRange) zoomRange.disabled = locked;
        if (textScaleRange) textScaleRange.disabled = locked;
        if (zoomReset) zoomReset.disabled = locked;
        const rh = document.getElementById('timelineResize');
        if (rh) {{
          rh.style.pointerEvents = locked ? 'none' : '';
          rh.style.opacity = locked ? '0.5' : '';
        }}
      }}

      function syncPlayButtons() {{
        const pb = document.getElementById('playTimelineBtn');
        if (pb) pb.disabled = isTimelinePlaying;
      }}

      function stopTimelinePlay(resetScroll) {{
        if (playRafId != null) {{
          cancelAnimationFrame(playRafId);
          playRafId = null;
        }}
        isTimelinePlaying = false;
        setTimelineScrubLocked(false);
        const px = parseFloat(zoomRange ? zoomRange.value : String(MODEL.px_per_sec || 8));
        const pxUse = isFinite(px) ? px : (MODEL.px_per_sec || 8);
        if (resetScroll !== false) {{
          scrollTimelineToTime(0, pxUse);
        }}
        syncPlayButtons();
      }}

      function playTimelineFrame() {{
        if (!isTimelinePlaying) return;
        const zr = document.getElementById('zoomRange');
        const px = zr ? parseFloat(zr.value) : NaN;
        if (!isFinite(px) || px <= 0) {{
          stopTimelinePlay(false);
          return;
        }}
        const dur = MODEL.duration_seconds || 0;
        if (dur <= 0) {{
          stopTimelinePlay(true);
          return;
        }}
        const elapsed = playTimeStart + (performance.now() - playWallStart) / 1000;
        if (elapsed >= dur) {{
          if (playRafId != null) {{
            cancelAnimationFrame(playRafId);
            playRafId = null;
          }}
          isTimelinePlaying = false;
          setTimelineScrubLocked(false);
          scrollTimelineToTime(dur, px);
          syncPlayButtons();
          return;
        }}
        scrollTimelineToTime(elapsed, px);
        playRafId = requestAnimationFrame(playTimelineFrame);
      }}

      function startTimelinePlay() {{
        if (isTimelinePlaying || !zoomRange) return;
        const px = parseFloat(zoomRange.value);
        if (!isFinite(px) || px <= 0) return;
        isTimelinePlaying = true;
        playWallStart = performance.now();
        playTimeStart = 0;
        setTimelineScrubLocked(true);
        syncPlayButtons();
        scrollTimelineToTime(0, px);
        requestAnimationFrame(() => {{
          scrollTimelineToTime(0, px);
          playRafId = requestAnimationFrame(playTimelineFrame);
        }});
      }}

      const playTimelineBtn = document.getElementById('playTimelineBtn');
      const stopTimelineBtn = document.getElementById('stopTimelineBtn');
      if (playTimelineBtn) playTimelineBtn.addEventListener('click', startTimelinePlay);
      if (stopTimelineBtn) stopTimelineBtn.addEventListener('click', () => stopTimelinePlay(true));

      const initialPx = zoomRange ? parseFloat(zoomRange.value) : defaultPx;
      apply(isFinite(initialPx) ? initialPx : defaultPx);
      syncPlayButtons();
    }})();
  </script>
</body>
</html>
"""


def save_html(html_text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(html_text)


def export_timeline_from_json_file(
    json_path: Path,
    html_path: Path,
    *,
    px_per_sec: float = 8.0,
    direction: str = "auto",
    rtl_timeline: bool | None = None,
    rtl_text_only: bool | None = None,
) -> None:
    """Load pipeline JSON and write a standalone interactive HTML timeline (same output as ``python -m app.timeline_exporter``)."""
    result = load_result_json(json_path)
    rtl_t, rtl_to = _resolve_direction_flags(
        result,
        rtl_timeline=rtl_timeline,
        rtl_text_only=rtl_text_only,
        direction=direction,
    )
    model = build_timeline_model(result, rtl_timeline=rtl_t, rtl_text_only=rtl_to, px_per_sec=px_per_sec)
    save_html(render_timeline_html(model), html_path)


def _resolve_direction_flags(result: dict[str, Any], *, rtl_timeline: bool | None, rtl_text_only: bool | None, direction: str) -> tuple[bool, bool]:
    language = _detect_language(result)
    # direction: auto|ltr|rtl
    if direction == "ltr":
        return (False, bool(rtl_text_only or False))
    if direction == "rtl":
        # Full RTL timeline.
        return (True, bool(rtl_text_only if rtl_text_only is not None else True))
    if direction == "auto":
        is_he = _language_is_hebrew(language) or _transcript_suggests_hebrew(result)
        if is_he:
            # Hebrew: RTL timeline (time axis and blocks flow right-to-left) and RTL text.
            # Explicit rtl_timeline / rtl_text_only from CLI still override when not None.
            rt = True if rtl_timeline is None else bool(rtl_timeline)
            rto = True if rtl_text_only is None else bool(rtl_text_only)
            return (rt, rto)
        return (bool(rtl_timeline or False), bool(rtl_text_only or False))

    return (bool(rtl_timeline or False), bool(rtl_text_only or False))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export WhisperX JSON into a standalone interactive HTML transcript timeline.")
    p.add_argument("--input", required=True, type=Path, help="Pipeline result JSON path.")
    p.add_argument("--output", required=True, type=Path, help="Output HTML path (e.g. output_timeline.html).")
    p.add_argument("--px-per-sec", type=float, default=8.0, help="Initial pixels per second (zoom baseline).")
    p.add_argument(
        "--direction",
        choices=["auto", "ltr", "rtl"],
        default="auto",
        help="Timeline mode: auto uses RTL timeline + RTL text for Hebrew (language field or Hebrew letters in transcript).",
    )
    p.add_argument("--rtl-timeline", action="store_true", help="Force the whole timeline to be RTL (right-to-left).")
    p.add_argument("--ltr-timeline", action="store_true", help="Force the whole timeline to be LTR.")
    p.add_argument("--rtl-text-only", action="store_true", help="Make only transcript text RTL (timeline stays LTR).")
    p.add_argument("--ltr-text-only", action="store_true", help="Make only transcript text LTR.")

    args = p.parse_args(argv)

    rtl_timeline_flag: bool | None = None
    if args.rtl_timeline:
        rtl_timeline_flag = True
    elif args.ltr_timeline:
        rtl_timeline_flag = False

    rtl_text_only_flag: bool | None = None
    if args.rtl_text_only:
        rtl_text_only_flag = True
    elif args.ltr_text_only:
        rtl_text_only_flag = False

    export_timeline_from_json_file(
        args.input,
        args.output,
        px_per_sec=args.px_per_sec,
        direction=args.direction,
        rtl_timeline=rtl_timeline_flag,
        rtl_text_only=rtl_text_only_flag,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

