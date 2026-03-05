
import os
import re
import json
import uuid
import time
import queue
import threading
import subprocess
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel
import webrtcvad
import wave

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

SEGMENT_SECONDS = int(os.environ.get("SEGMENT_SECONDS", "20"))
WORKERS = int(os.environ.get("WORKERS", "4"))
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
VOD_LANGUAGE = os.environ.get("VOD_LANGUAGE", "")

MAX_IN_MEMORY_SEGMENTS = int(os.environ.get("MAX_IN_MEMORY_SEGMENTS", "2500"))
MAX_HITS_PER_BRAND = int(os.environ.get("MAX_HITS_PER_BRAND", "200"))
MAX_HIDDEN_ITEMS = int(os.environ.get("MAX_HIDDEN_ITEMS", "200"))

app = FastAPI(title="Amar Stream AI Platform")
templates = Jinja2Templates(directory=str(ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

jobs: Dict[str, dict] = {}

HIDDEN_CONTEXT = {
    "crypto": ["krypto", "börse", "trading", "wallet", "bitcoin", "btc", "eth", "ethereum", "altcoin"],
    "hardware": ["maus", "keyboard", "tastatur", "headset", "logitech", "razer", "steelseries", "gpu", "grafikkarte"],
    "nutrition": ["shake", "protein", "creatin", "kreatin", "pulver", "makros", "kalorien"],
}

NOISE_WORDS = {
    "a", "uh", "um", "hm", "ah", "of", "ok", "okay"
}

def seconds_to_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h}h{m:02d}m{s:02d}s"

def fmt_hhmmss(seconds: float) -> str:
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def twitch_ts_link(url: str, seconds: float) -> str:
    ts = seconds_to_hms(seconds)
    joiner = "&" if "?" in url else "?"
    return f"{url}{joiner}t={ts}"

def _safe_word_regex(q: str) -> re.Pattern:
    return re.compile(rf"\b{re.escape(q)}\b", re.IGNORECASE)

def job_emit(job_id: str, item: dict):
    job = jobs.get(job_id)
    if not job:
        return
    job["sse_q"].put(item)

def _vod_direct_audio(url: str) -> Tuple[str, Optional[float]]:
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "format": "bestaudio/best",
        "noplaylist": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        direct = info.get("url")
        if not direct:
            raise RuntimeError("No direct audio url")
        dur = info.get("duration")
        try:
            dur_f = float(dur) if dur is not None else None
        except Exception:
            dur_f = None
        return direct, dur_f

def _run_vod_segmenter(vod_url: str, out_tmpl: str) -> Tuple[subprocess.Popen, Optional[float]]:
    direct, dur = _vod_direct_audio(vod_url)
    ffmpeg_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", direct,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        "-f", "segment",
        "-segment_time", str(SEGMENT_SECONDS),
        "-reset_timestamps", "1",
        out_tmpl,
    ]
    p = subprocess.Popen(ffmpeg_cmd)
    return p, dur

def clean_text(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    words = t.split()
    if len(words) < 3:
        return ""
    if len(t) <= 2:
        return ""
    if t.lower() in NOISE_WORDS:
        return ""
    return t

def wav_has_speech(path: Path, vad: webrtcvad.Vad) -> bool:
    try:
        with wave.open(str(path), "rb") as wf:
            if wf.getnchannels() != 1 or wf.getframerate() != 16000:
                return True
            if wf.getsampwidth() != 2:
                return True
            frame_ms = 30
            frame_bytes = int(16000 * frame_ms / 1000) * 2
            max_frames = 50
            voiced = 0
            for _ in range(max_frames):
                buf = wf.readframes(int(16000 * frame_ms / 1000))
                if not buf:
                    break
                if len(buf) < frame_bytes:
                    buf = buf + b"\x00" * (frame_bytes - len(buf))
                if vad.is_speech(buf, 16000):
                    voiced += 1
                    if voiced >= 2:
                        return True
            return False
    except Exception:
        return True

def _hidden_context_hits(text: str) -> List[str]:
    t = (text or "").lower()
    out: List[str] = []
    for k, words in HIDDEN_CONTEXT.items():
        if any(w in t for w in words):
            out.append(k)
    return out[:3]

def _append_transcript(job: dict, seg: dict):
    p = job["transcript_file"]
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(seg, ensure_ascii=False) + "\n")

def _update_sponsor(job: dict, seg: dict):
    if not job["watch"]:
        return
    text = seg.get("text", "")
    start = float(seg.get("start", 0.0))
    end = float(seg.get("end", 0.0))
    dur = max(0.0, min(20.0, end - start))
    for brand, pat in job["watch_patterns"].items():
        if pat.search(text):
            st = job["sponsor_report"].setdefault(brand, {"mentions": 0, "total_talktime_sec": 0.0, "hits": []})
            st["mentions"] += 1
            st["total_talktime_sec"] = round(float(st["total_talktime_sec"]) + dur, 1)
            if len(st["hits"]) < MAX_HITS_PER_BRAND:
                st["hits"].append({
                    "start": start,
                    "time": seconds_to_hms(start),
                    "text": text,
                    "link": twitch_ts_link(job["url"], start),
                })
            job_emit(job["id"], {"type": "alert", "word": brand, "start": start, "time": seconds_to_hms(start), "text": text, "link": twitch_ts_link(job["url"], start)})
            job_emit(job["id"], {"type": "sponsors", "report": job["sponsor_report"]})

def _update_hidden(job: dict, seg: dict):
    text = seg.get("text", "")
    start = float(seg.get("start", 0.0))
    tags = _hidden_context_hits(text)
    if not tags:
        return
    for tag in tags:
        prev = job["hidden_last"].get(tag)
        if prev is not None and abs(start - prev) < 25:
            continue
        job["hidden_last"][tag] = start
        if len(job["hidden_items"]) < MAX_HIDDEN_ITEMS:
            job["hidden_items"].append({
                "context": tag,
                "start": start,
                "time": seconds_to_hms(start),
                "text": text,
                "link": twitch_ts_link(job["url"], start),
            })
            job_emit(job["id"], {"type": "hidden_sponsor", "items": job["hidden_items"]})

def _emit_status(job: dict, status: str):
    pct = None
    eta = None
    if job.get("total_chunks"):
        pct = (job["done_chunks"] / job["total_chunks"]) * 100.0
        if job["avg_chunk_time"] and job["done_chunks"] > 0:
            remaining = max(0, job["total_chunks"] - job["done_chunks"])
            eta = int(remaining * job["avg_chunk_time"])
    job_emit(job["id"], {"type": "status", "status": status, "pct": pct, "eta_sec": eta})

def _compute_summary(job: dict) -> dict:
    segs = job["segments"]
    words = 0
    for s in segs:
        words += len((s.get("text", "") or "").split())
    brands = []
    for b, r in job["sponsor_report"].items():
        brands.append({"brand": b, "mentions": r.get("mentions", 0), "talktime_sec": r.get("total_talktime_sec", 0.0)})
    brands.sort(key=lambda x: (-x["mentions"], -x["talktime_sec"]))
    return {
        "segments": len(segs),
        "words": words,
        "brands": brands[:10],
        "segment_seconds": SEGMENT_SECONDS,
        "workers": WORKERS,
        "model": WHISPER_MODEL,
    }

def _segment_loop(job_id: str):
    job = jobs[job_id]
    chunks_dir = job["dir"] / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(chunks_dir / "chunk_%08d.wav")
    try:
        _emit_status(job, "Segmenting")
        p, duration = _run_vod_segmenter(job["url"], out_tmpl)
        job["ffmpeg"] = p
        job["duration"] = duration
        if duration:
            job["total_chunks"] = int(math.ceil(duration / SEGMENT_SECONDS))
        _emit_status(job, "Processing")
    except Exception as e:
        job_emit(job_id, {"type": "error", "message": str(e)})
        job["stop"] = True
        return

    vad = webrtcvad.Vad(2)

    def feeder():
        last = -1
        while not job["stop"]:
            wavs = sorted(chunks_dir.glob("chunk_*.wav"))
            for w in wavs:
                idx = int(w.stem.split("_")[1])
                if idx <= last:
                    continue
                job["work_q"].put((idx, w))
                last = idx
            if job["ffmpeg"] and job["ffmpeg"].poll() is not None:
                break
            time.sleep(0.15)
        for _ in range(WORKERS):
            job["work_q"].put((None, None))

    def worker(worker_id: int):
        local_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        while True:
            idx, wav = job["work_q"].get()
            if idx is None:
                return
            if job["stop"]:
                return
            t0 = time.time()
            try:
                if not wav_has_speech(wav, vad):
                    job["done_chunks"] += 1
                    dt = max(0.001, time.time() - t0)
                    job["avg_chunk_time"] = (job["avg_chunk_time"] * 0.9) + (dt * 0.1) if job["avg_chunk_time"] else dt
                    _emit_status(job, "Processing")
                    try:
                        wav.unlink(missing_ok=True)
                    except Exception:
                        pass
                    continue

                offset = float(idx) * SEGMENT_SECONDS
                segments, _ = local_model.transcribe(
                    str(wav),
                    beam_size=5,
                    language=VOD_LANGUAGE or None
                )
                for s in segments:
                    seg = {
                        "start": float(s.start) + offset,
                        "end": float(s.end) + offset,
                        "text": clean_text(s.text),
                    }
                    if not seg["text"]:
                        continue
                    job["segments"].append(seg)
                    if len(job["segments"]) > MAX_IN_MEMORY_SEGMENTS:
                        job["segments"] = job["segments"][-MAX_IN_MEMORY_SEGMENTS:]
                    _append_transcript(job, seg)
                    job_emit(job_id, {"type": "segment", "segment": seg})
                    _update_sponsor(job, seg)
                    _update_hidden(job, seg)

                job["done_chunks"] += 1
                dt = max(0.001, time.time() - t0)
                job["avg_chunk_time"] = (job["avg_chunk_time"] * 0.9) + (dt * 0.1) if job["avg_chunk_time"] else dt
                if job["done_chunks"] % 5 == 0:
                    job_emit(job_id, {"type": "summary", "summary": _compute_summary(job)})
                _emit_status(job, "Processing")
            except Exception as e:
                job_emit(job_id, {"type": "error", "message": str(e)})
                job["stop"] = True
            finally:
                try:
                    wav.unlink(missing_ok=True)
                except Exception:
                    pass

    threading.Thread(target=feeder, daemon=True).start()
    for wid in range(WORKERS):
        threading.Thread(target=worker, args=(wid,), daemon=True).start()

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/start")
async def api_start(payload: dict):
    url = (payload.get("url") or "").strip()
    if not url:
        raise HTTPException(400, "missing url")
    watch = payload.get("watch") or []
    watch = [str(w).strip() for w in watch if str(w).strip()]
    job_id = uuid.uuid4().hex[:10]
    job_dir = DATA / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    transcript_file = job_dir / "transcript.jsonl"
    if transcript_file.exists():
        transcript_file.unlink()

    job = {
        "id": job_id,
        "url": url,
        "mode": "vod",
        "dir": job_dir,
        "sse_q": queue.Queue(),
        "work_q": queue.Queue(),
        "segments": [],
        "transcript_file": transcript_file,
        "watch": set(watch),
        "watch_patterns": {b: _safe_word_regex(b) for b in watch},
        "sponsor_report": {},
        "hidden_items": [],
        "hidden_last": {},
        "stop": False,
        "ffmpeg": None,
        "duration": None,
        "total_chunks": None,
        "done_chunks": 0,
        "avg_chunk_time": None,
    }
    jobs[job_id] = job
    threading.Thread(target=_segment_loop, args=(job_id,), daemon=True).start()
    return {"job_id": job_id}

@app.post("/api/stop/{job_id}")
async def api_stop(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"ok": False}, status_code=404)
    job["stop"] = True
    p = job.get("ffmpeg")
    if p and p.poll() is None:
        try:
            p.terminate()
        except Exception:
            pass
    job_emit(job_id, {"type": "status", "status": "Stopped", "pct": None, "eta_sec": None})
    return {"ok": True}

@app.get("/api/events/{job_id}")
async def api_events(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")

    def gen():
        yield f"data: {json.dumps({'type':'status','status':'Connected','pct':0,'eta_sec':None})}\n\n"
        q = job["sse_q"]
        while True:
            item = q.get()
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")

@app.get("/api/search/{job_id}")
async def api_search(job_id: str, q: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    q = (q or "").strip()
    if not q:
        return {"results": []}
    pat = re.compile(re.escape(q), re.IGNORECASE)
    results = []
    for s in job["segments"]:
        if pat.search(s.get("text", "")):
            start = float(s.get("start", 0.0))
            results.append({
                "start": start,
                "time": seconds_to_hms(start),
                "text": s.get("text", ""),
                "link": twitch_ts_link(job["url"], start),
            })
            if len(results) >= 80:
                break
    return {"results": results}

@app.post("/api/watch/{job_id}")
async def api_watch(job_id: str, payload: dict):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    words = payload.get("words") or []
    words = [str(w).strip() for w in words if str(w).strip()]
    job["watch"] = set(words)
    job["watch_patterns"] = {b: _safe_word_regex(b) for b in words}
    job["sponsor_report"] = {}
    return {"ok": True}
