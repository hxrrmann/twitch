
import os
import re
import io
import json
import math
import time
import uuid
import wave
import shutil
import queue
import audioop
import threading
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


# ============================================================
# Paths and app
# ============================================================

ROOT = Path(__file__).resolve().parent
TEMPLATES_DIR = ROOT / "templates"
STATIC_DIR = ROOT / "static"
DATA_DIR = ROOT / "data"
JOBS_DIR = DATA_DIR / "jobs"
TMP_DIR = DATA_DIR / "tmp"

DATA_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Stream Intel")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ============================================================
# Config
# ============================================================

DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "small")
DEFAULT_COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8")
DEFAULT_LANG = os.environ.get("WHISPER_LANG", "de")
WORKERS = max(1, int(os.environ.get("WORKERS", "4")))
SEGMENT_SECONDS = max(15, min(30, int(os.environ.get("SEGMENT_SECONDS", "20"))))
VAD_FRAME_MS = 30
VAD_WINDOW_MS = 400
MIN_SPEECH_MS = 700
MAX_SILENCE_MS = 650
TRANSCRIPT_MIN_CHARS = 2
STATUS_POLL_SECONDS = 1.0
MAX_KEEP_JOBS = max(10, int(os.environ.get("MAX_KEEP_JOBS", "250")))
DOWNLOAD_TIMEOUT = int(os.environ.get("DOWNLOAD_TIMEOUT", "10800"))
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")


# ============================================================
# Model lazy load
# ============================================================

_MODEL = None
_MODEL_LOCK = threading.Lock()


def get_model():
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            from faster_whisper import WhisperModel
            _MODEL = WhisperModel(
                DEFAULT_MODEL,
                device="cpu",
                compute_type=DEFAULT_COMPUTE,
                cpu_threads=max(1, WORKERS),
                num_workers=max(1, WORKERS),
            )
        return _MODEL


# ============================================================
# Utilities
# ============================================================

def now_ts() -> float:
    return time.time()


def safe_slug(value: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip())
    return out.strip("_") or "item"


def hms(seconds: Optional[int]) -> str:
    if seconds is None:
        return ""
    s = max(0, int(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    r = s % 60
    if h > 0:
        return f"{h}:{m:02d}:{r:02d}"
    return f"{m}:{r:02d}"


def fmt_ts(seconds: float) -> str:
    s = max(0, int(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    r = s % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{r:02d}"
    return f"{m:02d}:{r:02d}"


def twitch_timestamp_link(vod_url: str, seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    suffix = ""
    if h > 0:
        suffix += f"{h}h"
    if m > 0 or h > 0:
        suffix += f"{m}m"
    suffix += f"{s}s"
    join = "&" if "?" in vod_url else "?"
    return f"{vod_url}{join}t={suffix}"


def list_job_files() -> List[Path]:
    files = [p for p in JOBS_DIR.glob("*.json") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def prune_old_jobs() -> None:
    files = list_job_files()
    for p in files[MAX_KEEP_JOBS:]:
        try:
            p.unlink()
        except Exception:
            pass


def job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def temp_job_dir(job_id: str) -> Path:
    d = TMP_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def default_job(job_id: str, vod_url: str = "", brand_alerts: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "id": job_id,
        "vod_url": vod_url,
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "eta_seconds": None,
        "eta_hms": "",
        "created_at": now_ts(),
        "updated_at": now_ts(),
        "paused": False,
        "error": None,
        "brand_alerts": brand_alerts or [],
        "artifacts": {
            "audio_wav": "",
            "segments_json": "",
            "job_dir": "",
        },
        "state": {
            "download_done": False,
            "speech_segments_done": False,
            "transcription_done": False,
            "analysis_done": False,
            "report_done": False,
        },
        "metrics": {
            "audio_seconds": 0,
            "speech_seconds": 0,
            "processed_segments": 0,
            "total_segments": 0,
            "mentions": 0,
        },
        "result": {
            "transcript": "",
            "segments": [],
            "brand_mentions": [],
            "hidden_context": [],
            "stream_summary": "",
            "timeline": [],
            "sponsor_report": {},
        },
    }


_JOB_LOCKS: Dict[str, threading.Lock] = {}
_RUNNING_THREADS: Dict[str, threading.Thread] = {}


def get_job_lock(job_id: str) -> threading.Lock:
    if job_id not in _JOB_LOCKS:
        _JOB_LOCKS[job_id] = threading.Lock()
    return _JOB_LOCKS[job_id]


def load_job(job_id: str) -> Dict[str, Any]:
    p = job_path(job_id)
    if not p.exists():
        raise KeyError(job_id)
    return read_json(p)


def save_job(job: Dict[str, Any]) -> None:
    job["updated_at"] = now_ts()
    job["eta_hms"] = hms(job.get("eta_seconds"))
    atomic_write_json(job_path(job["id"]), job)


def update_job(job_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    with get_job_lock(job_id):
        job = load_job(job_id)
        deep_update(job, patch)
        save_job(job)
        return job


def deep_update(target: Dict[str, Any], patch: Dict[str, Any]) -> None:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            deep_update(target[k], v)
        else:
            target[k] = v


def create_job(vod_url: str, brand_alerts: Optional[List[str]] = None) -> Dict[str, Any]:
    job_id = uuid.uuid4().hex
    job = default_job(job_id, vod_url=vod_url, brand_alerts=brand_alerts or [])
    d = temp_job_dir(job_id)
    job["artifacts"]["job_dir"] = str(d)
    save_job(job)
    prune_old_jobs()
    return job


def latest_matching_job(vod_url: str) -> Optional[Dict[str, Any]]:
    for p in list_job_files():
        try:
            job = read_json(p)
        except Exception:
            continue
        if job.get("vod_url") == vod_url and job.get("status") in {"running", "paused", "queued"}:
            return job
    return None


# ============================================================
# Routes
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if not TEMPLATES_DIR.exists():
        return HTMLResponse("<h1>templates directory not found</h1>", status_code=500)
    if (TEMPLATES_DIR / "index.html").exists():
        return templates.TemplateResponse("index.html", {"request": request})
    return HTMLResponse("<h1>index.html not found</h1>", status_code=500)


@app.get("/health")
async def health():
    return {"ok": True, "workers": WORKERS, "segment_seconds": SEGMENT_SECONDS}


@app.post("/api/job")
async def api_create_job(payload: Dict[str, Any]):
    vod_url = (payload or {}).get("vod_url", "").strip()
    if not vod_url:
        raise HTTPException(status_code=400, detail="vod_url fehlt")
    brand_alerts = normalize_brand_alerts((payload or {}).get("brand_alerts"))

    existing = latest_matching_job(vod_url)
    if existing:
        if brand_alerts:
            existing["brand_alerts"] = dedupe_strings(existing.get("brand_alerts", []) + brand_alerts)
            save_job(existing)
        ensure_runner(existing["id"])
        return {
            "job_id": existing["id"],
            "resumed": True,
            "status": existing.get("status"),
        }

    job = create_job(vod_url, brand_alerts=brand_alerts)
    ensure_runner(job["id"])
    return {"job_id": job["id"], "resumed": False, "status": job["status"]}


@app.get("/api/job/{job_id}")
async def api_get_job(job_id: str):
    try:
        job = load_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job nicht gefunden")
    return job


@app.post("/api/job/{job_id}/pause")
async def api_pause_job(job_id: str):
    try:
        job = update_job(job_id, {"paused": True, "status": "paused"})
    except KeyError:
        raise HTTPException(status_code=404, detail="job nicht gefunden")
    return {"ok": True, "job_id": job_id, "status": job["status"]}


@app.post("/api/job/{job_id}/resume")
async def api_resume_job(job_id: str):
    try:
        job = update_job(job_id, {"paused": False, "status": "queued"})
    except KeyError:
        raise HTTPException(status_code=404, detail="job nicht gefunden")
    ensure_runner(job_id)
    return {"ok": True, "job_id": job_id, "status": job["status"]}


@app.get("/api/job_stream/{job_id}")
async def api_job_stream(job_id: str):
    try:
        load_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job nicht gefunden")

    async def gen():
        last = None
        while True:
            try:
                job = load_job(job_id)
            except KeyError:
                break
            cur = json.dumps(job, ensure_ascii=False)
            if cur != last:
                last = cur
                yield f"data: {cur}\n\n"
            if job.get("status") in {"done", "error"}:
                break
            import asyncio
            await asyncio.sleep(STATUS_POLL_SECONDS)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/brand_alerts")
async def api_brand_alerts(payload: Dict[str, Any]):
    job_id = (payload or {}).get("job_id", "").strip()
    brands = normalize_brand_alerts((payload or {}).get("brands"))
    if not job_id:
        raise HTTPException(status_code=400, detail="job_id fehlt")
    try:
        job = load_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job nicht gefunden")

    merged = dedupe_strings(job.get("brand_alerts", []) + brands)
    job["brand_alerts"] = merged
    recompute_mentions(job)
    save_job(job)
    return {"ok": True, "brands": merged, "mentions": len(job["result"].get("brand_mentions", []))}


@app.get("/api/sponsor_report/{job_id}")
async def api_sponsor_report(job_id: str):
    try:
        job = load_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job nicht gefunden")
    return JSONResponse(job["result"].get("sponsor_report", {}))


@app.get("/api/sponsor_report_pdf/{job_id}")
async def api_sponsor_report_pdf(job_id: str):
    try:
        job = load_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job nicht gefunden")

    report = job["result"].get("sponsor_report") or {}
    if not report:
        raise HTTPException(status_code=400, detail="kein sponsor report vorhanden")

    pdf = build_pdf_bytes(report, job)
    headers = {
        "Content-Disposition": f'attachment; filename="sponsor_report_{job_id}.pdf"'
    }
    return Response(content=pdf, media_type="application/pdf", headers=headers)


@app.get("/api/stream_summary/{job_id}")
async def api_stream_summary(job_id: str):
    try:
        job = load_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job nicht gefunden")
    return {"summary": job["result"].get("stream_summary", "")}


@app.get("/api/hidden_context/{job_id}")
async def api_hidden_context(job_id: str):
    try:
        job = load_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job nicht gefunden")
    return {"items": job["result"].get("hidden_context", [])}


@app.get("/api/brand_mentions/{job_id}")
async def api_brand_mentions(job_id: str):
    try:
        job = load_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job nicht gefunden")
    return {"mentions": job["result"].get("brand_mentions", [])}


# Backward compatible aliases that many frontends use
@app.post("/api/start")
async def api_start_alias(payload: Dict[str, Any]):
    return await api_create_job(payload)


@app.post("/api/stop")
async def api_stop_alias(payload: Dict[str, Any]):
    job_id = (payload or {}).get("job_id", "").strip()
    return await api_pause_job(job_id)


@app.post("/api/resume")
async def api_resume_alias(payload: Dict[str, Any]):
    job_id = (payload or {}).get("job_id", "").strip()
    return await api_resume_job(job_id)


# ============================================================
# Normalization
# ============================================================

def normalize_brand_alerts(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,;\n]+", value)
    elif isinstance(value, list):
        parts = [str(x) for x in value]
    else:
        parts = [str(value)]
    clean = []
    for part in parts:
        p = part.strip()
        if p:
            clean.append(p)
    return dedupe_strings(clean)


def dedupe_strings(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


# ============================================================
# Pipeline entry
# ============================================================

def ensure_runner(job_id: str) -> None:
    existing = _RUNNING_THREADS.get(job_id)
    if existing and existing.is_alive():
        return
    thread = threading.Thread(target=run_job_safe, args=(job_id,), daemon=True)
    _RUNNING_THREADS[job_id] = thread
    thread.start()


def run_job_safe(job_id: str) -> None:
    try:
        run_job(job_id)
    except Exception as exc:
        try:
            update_job(job_id, {
                "status": "error",
                "stage": "error",
                "error": str(exc),
                "eta_seconds": 0,
            })
        except Exception:
            pass


def wait_if_paused(job_id: str) -> None:
    while True:
        job = load_job(job_id)
        if not job.get("paused"):
            return
        time.sleep(0.8)


def run_job(job_id: str) -> None:
    job = load_job(job_id)
    vod_url = job.get("vod_url", "")
    jdir = Path(job["artifacts"]["job_dir"])
    jdir.mkdir(parents=True, exist_ok=True)

    # 1 download and wav
    if not job["state"].get("download_done"):
        update_job(job_id, {
            "status": "running",
            "stage": "download",
            "progress": 2,
            "eta_seconds": None,
            "error": None,
        })
        wait_if_paused(job_id)
        audio_path = download_and_prepare_audio(vod_url, jdir)
        duration = wav_duration_seconds(audio_path)
        update_job(job_id, {
            "artifacts": {"audio_wav": str(audio_path)},
            "metrics": {"audio_seconds": int(duration)},
            "state": {"download_done": True},
            "progress": 12,
            "stage": "audio_ready",
            "eta_seconds": estimate_eta_from_audio(duration, phase="speech"),
        })
    else:
        audio_path = Path(load_job(job_id)["artifacts"]["audio_wav"])

    # 2 speech detection
    job = load_job(job_id)
    speech_json = jdir / "speech_segments.json"
    if not job["state"].get("speech_segments_done"):
        update_job(job_id, {
            "status": "running",
            "stage": "speech_detection",
            "progress": 18,
            "eta_seconds": estimate_eta_from_audio(job["metrics"].get("audio_seconds", 0), phase="speech"),
        })
        wait_if_paused(job_id)
        speech_segments = detect_speech_segments(audio_path)
        atomic_write_json(speech_json, {"segments": speech_segments})
        speech_seconds = int(sum(max(0.0, b - a) for a, b in speech_segments))
        update_job(job_id, {
            "state": {"speech_segments_done": True},
            "metrics": {
                "speech_seconds": speech_seconds,
                "total_segments": len(speech_segments),
            },
            "progress": 30,
            "stage": "speech_ready",
            "eta_seconds": estimate_eta_from_audio(speech_seconds, phase="transcribe"),
        })
    else:
        speech_segments = read_json(speech_json)["segments"]

    # 3 transcription with resume
    job = load_job(job_id)
    segments_json = Path(job["artifacts"].get("segments_json") or (jdir / "segments.json"))
    if not segments_json.exists():
        atomic_write_json(segments_json, {"segments": [], "done_indexes": []})

    if not job["state"].get("transcription_done"):
        update_job(job_id, {
            "status": "running",
            "stage": "transcribing",
            "progress": max(job.get("progress", 30), 32),
            "artifacts": {"segments_json": str(segments_json)},
            "eta_seconds": estimate_eta_from_audio(job["metrics"].get("speech_seconds", 0), phase="transcribe"),
        })
        transcribe_with_resume(job_id, audio_path, speech_segments, segments_json)

    # 4 analysis
    job = load_job(job_id)
    if not job["state"].get("analysis_done"):
        update_job(job_id, {
            "status": "running",
            "stage": "analysis",
            "progress": 85,
            "eta_seconds": 40,
        })
        wait_if_paused(job_id)
        build_analysis(job_id, vod_url, segments_json)

    # 5 report done
    update_job(job_id, {
        "state": {"report_done": True},
        "stage": "done",
        "status": "done",
        "progress": 100,
        "eta_seconds": 0,
    })


# ============================================================
# Audio prep
# ============================================================

def run_cmd(cmd: List[str], timeout: Optional[int] = None) -> None:
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout,
    )


def download_and_prepare_audio(vod_url: str, jdir: Path) -> Path:
    source = jdir / "source.%(ext)s"
    wav = jdir / "audio.wav"

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "outtmpl": str(source),
        "format": "bestaudio/best",
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 5,
        "continuedl": True,
        "overwrites": True,
    }

    from yt_dlp import YoutubeDL
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([vod_url])

    # find downloaded source
    media_files = [p for p in jdir.iterdir() if p.is_file() and p.name != "audio.wav" and not p.name.endswith(".json")]
    if not media_files:
        raise RuntimeError("audio download fehlgeschlagen")
    src = max(media_files, key=lambda p: p.stat().st_size)

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i", str(src),
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        str(wav),
    ]
    run_cmd(cmd, timeout=DOWNLOAD_TIMEOUT)
    if not wav.exists():
        raise RuntimeError("wav erzeugung fehlgeschlagen")
    return wav


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return 0.0 if rate <= 0 else frames / float(rate)


# ============================================================
# Simple speech detection
# ============================================================

def read_wave(path: Path) -> Tuple[bytes, int]:
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1:
            raise RuntimeError("wav muss mono sein")
        if wf.getsampwidth() != 2:
            raise RuntimeError("wav muss 16 bit sein")
        sr = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
        return pcm, sr


def frame_bytes(sr: int, frame_ms: int) -> int:
    return int(sr * frame_ms / 1000) * 2


def detect_speech_segments(path: Path) -> List[Tuple[float, float]]:
    pcm, sr = read_wave(path)
    fb = frame_bytes(sr, VAD_FRAME_MS)
    if fb <= 0:
        return []

    total_frames = len(pcm) // fb
    if total_frames <= 0:
        return []

    energies = []
    for i in range(total_frames):
        chunk = pcm[i * fb:(i + 1) * fb]
        rms = audioop.rms(chunk, 2)
        energies.append(rms)

    if not energies:
        return []

    sorted_e = sorted(energies)
    noise_floor = sorted_e[max(0, int(len(sorted_e) * 0.20) - 1)]
    threshold = max(250, int(noise_floor * 2.2))

    frame_sec = VAD_FRAME_MS / 1000.0
    min_speech = MIN_SPEECH_MS / 1000.0
    max_silence = MAX_SILENCE_MS / 1000.0

    segments: List[Tuple[float, float]] = []
    in_speech = False
    start = 0.0
    silence_run = 0.0

    for idx, rms in enumerate(energies):
        t0 = idx * frame_sec
        is_speech = rms >= threshold

        if is_speech and not in_speech:
            in_speech = True
            start = t0
            silence_run = 0.0
        elif in_speech:
            if is_speech:
                silence_run = 0.0
            else:
                silence_run += frame_sec
                if silence_run >= max_silence:
                    end = max(start, t0 - silence_run + frame_sec)
                    if end - start >= min_speech:
                        segments.append((round(start, 3), round(end, 3)))
                    in_speech = False
                    silence_run = 0.0

    if in_speech:
        end = total_frames * frame_sec
        if end - start >= min_speech:
            segments.append((round(start, 3), round(end, 3)))

    # merge close gaps
    merged: List[Tuple[float, float]] = []
    for s, e in segments:
        if not merged:
            merged.append((s, e))
            continue
        ps, pe = merged[-1]
        if s - pe <= 0.45:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))

    # split very long ranges to target size
    final: List[Tuple[float, float]] = []
    for s, e in merged:
        length = e - s
        if length <= SEGMENT_SECONDS:
            final.append((s, e))
            continue
        cur = s
        while cur < e:
            nxt = min(e, cur + SEGMENT_SECONDS)
            final.append((round(cur, 3), round(nxt, 3)))
            cur = nxt
    return final


# ============================================================
# Transcription
# ============================================================

def estimate_eta_from_audio(seconds: int, phase: str) -> int:
    seconds = int(seconds or 0)
    if phase == "speech":
        return max(10, int(seconds / 120) + 8)
    if phase == "transcribe":
        # small int8 on cpu best effort
        return max(15, int(seconds / max(1, WORKERS * 6)) + 20)
    return max(5, int(seconds / 10))


def extract_clip_to_wav(full_wav: Path, out_path: Path, start: float, end: float) -> None:
    dur = max(0.1, end - start)
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i", str(full_wav),
        "-ss", f"{start:.3f}",
        "-t", f"{dur:.3f}",
        "-ac", "1",
        "-ar", "16000",
        str(out_path),
    ]
    run_cmd(cmd, timeout=max(30, int(dur * 3 + 30)))


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    text = text.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    text = re.sub(r"([a-zA-ZäöüÄÖÜß0-9])\s+([,.!?])", r"\1\2", text)
    return text.strip()


def likely_fragment(text: str) -> bool:
    t = normalize_text(text)
    if not t:
        return True
    words = t.split()
    if len(t) < TRANSCRIPT_MIN_CHARS:
        return True
    if len(words) == 1:
        w = words[0].lower()
        if re.fullmatch(r"\d+", w):
            return True
        if len(w) <= 2:
            return True
    return False


def punctuate_merge(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not lines:
        return []

    out: List[Dict[str, Any]] = []
    cur = dict(lines[0])

    for item in lines[1:]:
        gap = float(item["start"]) - float(cur["end"])
        joinable = gap <= 1.4 and len(cur["text"]) < 260
        if joinable:
            merged_text = normalize_text((cur["text"] + " " + item["text"]).strip())
            cur["text"] = merged_text
            cur["end"] = item["end"]
            cur["link"] = item["link"]
        else:
            cur["text"] = finalize_sentence(cur["text"])
            out.append(cur)
            cur = dict(item)

    cur["text"] = finalize_sentence(cur["text"])
    out.append(cur)
    return [x for x in out if not likely_fragment(x["text"])]


def finalize_sentence(text: str) -> str:
    t = normalize_text(text)
    if not t:
        return ""
    if t[-1] not in ".!?":
        t += "."
    return t[0].upper() + t[1:] if len(t) > 1 else t.upper()


def transcribe_single_clip(clip_path: Path) -> str:
    model = get_model()
    seg_iter, _info = model.transcribe(
        str(clip_path),
        language=DEFAULT_LANG,
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False,
        vad_filter=True,
        word_timestamps=False,
    )
    parts = []
    for seg in seg_iter:
        txt = normalize_text(seg.text)
        if txt:
            parts.append(txt)
    text = normalize_text(" ".join(parts))
    return text


def transcribe_with_resume(job_id: str, full_wav: Path, speech_segments: List[Tuple[float, float]], segments_json: Path) -> None:
    store = read_json(segments_json)
    existing_segments = store.get("segments", [])
    done_indexes = set(store.get("done_indexes", []))

    total = len(speech_segments)
    if total == 0:
        update_job(job_id, {
            "state": {"transcription_done": True},
            "metrics": {"processed_segments": 0, "total_segments": 0},
            "progress": 80,
            "stage": "transcribing_done",
            "eta_seconds": 8,
            "result": {"segments": [], "transcript": ""},
        })
        return

    pending = [(idx, rng) for idx, rng in enumerate(speech_segments) if idx not in done_indexes]
    clips_dir = temp_job_dir(job_id) / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    q: "queue.Queue[Tuple[int, Tuple[float, float]]]" = queue.Queue()
    for item in pending:
        q.put(item)

    results_lock = threading.Lock()
    stop_flag = {"stop": False}

    def worker():
        while True:
            try:
                idx, (start, end) = q.get_nowait()
            except queue.Empty:
                return
            try:
                while load_job(job_id).get("paused"):
                    time.sleep(0.8)
                clip = clips_dir / f"seg_{idx:06d}.wav"
                if not clip.exists():
                    extract_clip_to_wav(full_wav, clip, float(start), float(end))
                text = transcribe_single_clip(clip)
                # drop hallucinated tiny fragments
                if likely_fragment(text):
                    text = ""
                item = {
                    "index": idx,
                    "start": float(start),
                    "end": float(end),
                    "text": text,
                    "link": twitch_timestamp_link(load_job(job_id).get("vod_url", ""), float(start)),
                }
                with results_lock:
                    existing_segments.append(item)
                    done_indexes.add(idx)
                    atomic_write_json(segments_json, {
                        "segments": existing_segments,
                        "done_indexes": sorted(done_indexes),
                    })
                    processed = len(done_indexes)
                    pct = 30 + int((processed / max(1, total)) * 50)
                    remaining = max(0, total - processed)
                    eta = max(5, int(remaining * 6 / max(1, WORKERS)))
                    update_job(job_id, {
                        "metrics": {
                            "processed_segments": processed,
                            "total_segments": total,
                        },
                        "progress": min(80, pct),
                        "eta_seconds": eta,
                        "stage": "transcribing",
                    })
            finally:
                q.task_done()

    threads = []
    for _ in range(max(1, WORKERS)):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    clean_segments = [s for s in existing_segments if normalize_text(s.get("text", ""))]
    clean_segments.sort(key=lambda x: x["start"])
    merged = punctuate_merge(clean_segments)
    transcript = "\n".join([item["text"] for item in merged])

    atomic_write_json(segments_json, {
        "segments": merged,
        "done_indexes": list(range(total)),
    })
    update_job(job_id, {
        "state": {"transcription_done": True},
        "metrics": {
            "processed_segments": total,
            "total_segments": total,
        },
        "progress": 80,
        "stage": "transcribing_done",
        "eta_seconds": 12,
        "result": {
            "segments": merged,
            "transcript": transcript,
        },
        "artifacts": {"segments_json": str(segments_json)},
    })


# ============================================================
# Analysis
# ============================================================

SPONSOR_PATTERNS = [
    "sponsor", "gesponsert", "partner", "partnerschaft", "werbung", "anzeige",
    "code", "rabatt", "gutschein", "link in der beschreibung", "beschreibung",
    "checkt", "nutzt den code", "mit code", "aktion", "deal",
]

CONTEXT_MAP = {
    "crypto": ["wallet", "coin", "btc", "bitcoin", "ethereum", "trading", "broker", "exchange", "krypto"],
    "supplements": ["protein", "shake", "supplement", "whey", "pre workout", "creatine"],
    "hardware": ["maus", "tastatur", "headset", "monitor", "gpu", "grafikkarte", "pc", "setup"],
    "banking": ["karte", "bank", "konto", "depot", "investieren"],
    "gaming": ["skin", "ranked", "turnier", "scrim", "fortnite", "valorant", "gameplay"],
}

def build_analysis(job_id: str, vod_url: str, segments_json: Path) -> None:
    job = load_job(job_id)
    segments = read_json(segments_json).get("segments", [])

    brand_mentions = build_brand_mentions(vod_url, segments, job.get("brand_alerts", []))
    hidden_context = build_hidden_context(segments)
    summary = build_stream_summary(segments, brand_mentions, hidden_context)
    timeline = build_timeline(segments, brand_mentions)
    sponsor_report = build_sponsor_report(vod_url, segments, brand_mentions)

    job["result"]["brand_mentions"] = brand_mentions
    job["result"]["hidden_context"] = hidden_context
    job["result"]["stream_summary"] = summary
    job["result"]["timeline"] = timeline
    job["result"]["sponsor_report"] = sponsor_report
    job["state"]["analysis_done"] = True
    job["metrics"]["mentions"] = len(brand_mentions)
    save_job(job)


def build_brand_mentions(vod_url: str, segments: List[Dict[str, Any]], brands: List[str]) -> List[Dict[str, Any]]:
    out = []
    patterns = [b for b in brands if b.strip()]
    patterns_lower = [p.lower() for p in patterns]
    for seg in segments:
        text = (seg.get("text") or "").strip()
        low = text.lower()
        found = []
        for i, brand in enumerate(patterns):
            if patterns_lower[i] in low:
                found.append(brand)
        sponsor_language = [p for p in SPONSOR_PATTERNS if p in low]
        if found or sponsor_language:
            out.append({
                "start": seg["start"],
                "end": seg["end"],
                "timestamp": fmt_ts(seg["start"]),
                "text": text,
                "brands": found,
                "sponsor_signals": sponsor_language,
                "link": twitch_timestamp_link(vod_url, seg["start"]),
            })
    return out


def recompute_mentions(job: Dict[str, Any]) -> None:
    vod_url = job.get("vod_url", "")
    segments = job["result"].get("segments", [])
    mentions = build_brand_mentions(vod_url, segments, job.get("brand_alerts", []))
    job["result"]["brand_mentions"] = mentions
    job["result"]["sponsor_report"] = build_sponsor_report(vod_url, segments, mentions)
    job["metrics"]["mentions"] = len(mentions)


def build_hidden_context(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    examples: Dict[str, str] = {}
    for seg in segments:
        text = (seg.get("text") or "").lower()
        for topic, words in CONTEXT_MAP.items():
            if any(word in text for word in words):
                counts[topic] = counts.get(topic, 0) + 1
                examples.setdefault(topic, seg.get("text") or "")
    items = []
    for topic, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        items.append({
            "topic": topic,
            "count": count,
            "example": examples.get(topic, ""),
        })
    return items


def build_stream_summary(segments: List[Dict[str, Any]], brand_mentions: List[Dict[str, Any]], hidden_context: List[Dict[str, Any]]) -> str:
    if not segments:
        return ""
    topic_part = ""
    if hidden_context:
        top = [item["topic"] for item in hidden_context[:3]]
        topic_part = " Wichtige Themen: " + ", ".join(top) + "."
    sponsor_part = ""
    if brand_mentions:
        brands = dedupe_strings([b for m in brand_mentions for b in m.get("brands", [])])
        if brands:
            sponsor_part = " Erwähnte Marken: " + ", ".join(brands[:5]) + "."
        else:
            sponsor_part = " Es wurden mehrere Sponsoring Formulierungen erkannt."
    lead = " ".join([seg["text"] for seg in segments[: min(6, len(segments))]])
    lead = lead[:850].strip()
    summary = (lead + "." if lead and lead[-1] not in ".!?" else lead) + sponsor_part + topic_part
    return normalize_text(summary)


def build_timeline(segments: List[Dict[str, Any]], brand_mentions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    mention_starts = {(m["start"], m["end"]): m for m in brand_mentions}
    for seg in segments[:250]:
        label = "talk"
        for m in brand_mentions:
            if abs(float(seg["start"]) - float(m["start"])) < 0.01:
                label = "sponsor_talk"
                break
        out.append({
            "timestamp": fmt_ts(seg["start"]),
            "label": label,
            "text": seg["text"][:160],
            "link": seg["link"],
        })
    return out


def build_sponsor_report(vod_url: str, segments: List[Dict[str, Any]], brand_mentions: List[Dict[str, Any]]) -> Dict[str, Any]:
    brands = dedupe_strings([b for m in brand_mentions for b in m.get("brands", [])])
    total_talk_time = sum(max(0.0, float(m["end"]) - float(m["start"])) for m in brand_mentions)

    clips = []
    for mention in brand_mentions[:20]:
        clips.append({
            "timestamp": mention["timestamp"],
            "start": mention["start"],
            "end": mention["end"],
            "text": mention["text"],
            "link": mention["link"],
            "brands": mention.get("brands", []),
            "viewer_count": None,
        })

    return {
        "brands": brands,
        "mention_count": len(brand_mentions),
        "total_talk_time_seconds": int(total_talk_time),
        "total_talk_time_hms": hms(int(total_talk_time)),
        "clips_created": len(clips),
        "clips": clips,
    }


# ============================================================
# PDF export
# ============================================================

def build_pdf_bytes(report: Dict[str, Any], job: Dict[str, Any]) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "Sponsor Report")
    y -= 24

    c.setFont("Helvetica", 11)
    c.drawString(40, y, f"Job ID: {job['id']}")
    y -= 16
    c.drawString(40, y, f"VOD: {job.get('vod_url', '')[:90]}")
    y -= 16
    c.drawString(40, y, f"Brands: {', '.join(report.get('brands', [])) or 'Keine'}")
    y -= 16
    c.drawString(40, y, f"Mentions: {report.get('mention_count', 0)}")
    y -= 16
    c.drawString(40, y, f"Talk Time: {report.get('total_talk_time_hms', '0:00')}")
    y -= 24

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Mentions")
    y -= 18

    c.setFont("Helvetica", 10)
    for clip in report.get("clips", []):
        lines = [
            f"{clip.get('timestamp', '')}  {', '.join(clip.get('brands', []))}".strip(),
            clip.get("text", "")[:110],
            clip.get("link", "")[:110],
        ]
        for line in lines:
            c.drawString(40, y, line)
            y -= 14
            if y < 60:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 10)
        y -= 6

    c.showPage()
    c.save()
    return buf.getvalue()


# ============================================================
# Cloud Run start
# ============================================================

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
