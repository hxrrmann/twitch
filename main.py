import os
import re
import json
import time
import uuid
import queue
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from faster_whisper import WhisperModel
from yt_dlp import YoutubeDL

# Optional: PDF export for sponsor report
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

app = FastAPI(title="Hoermi VOD Analyzer")
templates = Jinja2Templates(directory=str(ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

# -----------------------------
# Config (tuned for speed)
# -----------------------------
MODEL_NAME = os.getenv("WHISPER_MODEL", "small")  # small is a good speed/quality compromise on CPU
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE", "int8")  # int8 is much faster on CPU
SEGMENT_SECONDS = int(os.getenv("SEGMENT_SECONDS", "20"))  # user asked 15-30s chunks
WORKERS = int(os.getenv("WORKERS", "4"))
CPU_THREADS_PER_WORKER = int(os.getenv("CPU_THREADS_PER_WORKER", "1"))

# Energy VAD (cheap, avoids word-fetzen from silence / game pauses)
RMS_WINDOW_MS = int(os.getenv("RMS_WINDOW_MS", "30"))
RMS_THRESHOLD = float(os.getenv("RMS_THRESHOLD", "0.010"))  # tuned: lower = more sensitive
MIN_VOICE_RATIO = float(os.getenv("MIN_VOICE_RATIO", "0.08"))  # fraction of frames above threshold to consider "speech"

# Merge logic
MERGE_GAP_S = float(os.getenv("MERGE_GAP_S", "1.2"))
MIN_TEXT_CHARS = int(os.getenv("MIN_TEXT_CHARS", "3"))

# -----------------------------
# Small helpers
# -----------------------------
def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}")

def _fmt_ts(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def _twitch_link(vod_url: str, seconds: float) -> str:
    # Twitch VOD supports ?t=1h2m3s format
    t = int(max(0, seconds))
    h = t // 3600
    m = (t % 3600) // 60
    s = t % 60
    suffix = ""
    if h:
        suffix += f"{h}h"
    if m or h:
        suffix += f"{m}m"
    suffix += f"{s}s"
    joiner = "&" if "?" in vod_url else "?"
    return f"{vod_url}{joiner}t={suffix}"

def _clean_text(t: str) -> str:
    t = (t or "").strip()
    # normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()
    # drop obvious garbage
    if len(t) < MIN_TEXT_CHARS:
        return ""
    if re.fullmatch(r"[\W\d_]+", t):
        return ""
    # remove repeated single tokens like "hier, hier, hier" -> keep once or twice
    t = re.sub(r"(\b\w+\b)(?:[ ,.!?]+\1){3,}", r"\1", t, flags=re.IGNORECASE)
    return t

def _energy_vad_has_voice(wav_path: Path) -> bool:
    """
    Very cheap VAD: read PCM16 mono 16k and check RMS in short windows.
    Avoids heavy deps + avoids webrtcvad build issues on py3.11.
    """
    import wave
    import numpy as np

    with wave.open(str(wav_path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            return True  # unexpected format; don't skip
        fr = w.getframerate()
        data = w.readframes(w.getnframes())
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        return False
    win = int(fr * (RMS_WINDOW_MS / 1000.0))
    if win <= 0:
        return True
    # pad
    n = (audio.size // win) * win
    if n <= 0:
        return False
    audio = audio[:n]
    frames = audio.reshape(-1, win)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    voice = (rms > RMS_THRESHOLD).mean()
    return voice >= MIN_VOICE_RATIO

# -----------------------------
# Model (lazy)
# -----------------------------
_model_lock = threading.Lock()
_model: Optional[WhisperModel] = None

def get_model() -> WhisperModel:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = WhisperModel(
                    MODEL_NAME,
                    device="cpu",
                    compute_type=COMPUTE_TYPE,
                    cpu_threads=max(1, CPU_THREADS_PER_WORKER),
                )
    return _model

# -----------------------------
# Jobs + event streaming
# -----------------------------
class Job:
    def __init__(self, vod_url: str, watch: Optional[List[str]] = None, alerts: Optional[List[str]] = None):
        self.id = uuid.uuid4().hex[:10]
        self.vod_url = vod_url
        self.created = time.time()
        self.status = "queued"
        self.progress = 0.0
        self.eta_s: Optional[float] = None
        self.total_chunks = 0
        self.done_chunks = 0
        self._events: "queue.Queue[dict]" = queue.Queue()
        self._stop = threading.Event()

        self.watch = [w.strip() for w in (watch or []) if w.strip()]
        self.alerts = [a.strip() for a in (alerts or []) if a.strip()]

        self.segments: List[dict] = []
        self.sponsor_hits: Dict[str, List[dict]] = {}
        self.hidden_context_hits: Dict[str, List[dict]] = {}
        self.summary: Dict[str, Any] = {}

        self._last_emit_t = 0.0

    def push(self, msg: dict) -> None:
        self._events.put(msg)

    def stop(self) -> None:
        self._stop.set()

JOBS: Dict[str, Job] = {}
JOBS_LOCK = threading.Lock()

def ensure_job(job_id: str) -> Job:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job")
    return job

def _emit_progress(job: Job) -> None:
    now = time.time()
    # limit progress event spam
    if now - job._last_emit_t < 0.25:
        return
    job._last_emit_t = now
    msg = {
        "type": "progress",
        "progress": round(job.progress, 4),
        "done": job.done_chunks,
        "total": job.total_chunks,
        "eta_s": None if job.eta_s is None else int(max(0, job.eta_s)),
        "status": job.status,
    }
    job.push(msg)

# -----------------------------
# Pipeline
# -----------------------------
SPONSOR_PHRASES = [
    "sponsored by", "sponsor", "werbung", "anzeige", "partner", "promotion", "promo code", "rabattcode",
    "mit freundlicher unterstützung", "unterstützt von", "in zusammenarbeit mit", "affiliate"
]
HIDDEN_CONTEXT = {
    "crypto": ["krypto", "crypto", "bitcoin", "btc", "ethereum", "eth", "wallet", "cold wallet", "börse", "exchange", "trading"],
    "hardware": ["hardware wallet", "ledger", "trezor", "cold storage"],
    "gaming": ["fortnite", "valorant", "aim", "dropmap", "ranked", "scrims"],
}

def _match_any(text: str, needles: List[str]) -> bool:
    tl = text.lower()
    return any(n.lower() in tl for n in needles)

def _update_sponsor(job: Job, start_s: float, end_s: float, text: str) -> None:
    # explicit brand mentions
    for brand in job.alerts:
        if brand.lower() in text.lower():
            job.sponsor_hits.setdefault(brand, []).append({
                "start": start_s, "end": end_s, "ts": _fmt_ts(start_s),
                "text": text, "link": _twitch_link(job.vod_url, start_s)
            })
    # sponsor intelligence (implicit)
    if _match_any(text, SPONSOR_PHRASES):
        job.hidden_context_hits.setdefault("sponsor_intel", []).append({
            "start": start_s, "end": end_s, "ts": _fmt_ts(start_s),
            "text": text, "link": _twitch_link(job.vod_url, start_s)
        })
    for key, words in HIDDEN_CONTEXT.items():
        if _match_any(text, words):
            job.hidden_context_hits.setdefault(key, []).append({
                "start": start_s, "end": end_s, "ts": _fmt_ts(start_s),
                "text": text, "link": _twitch_link(job.vod_url, start_s)
            })

def _finalize_reports(job: Job) -> None:
    rep = []
    for brand, hits in job.sponsor_hits.items():
        talk = 0.0
        for h in hits:
            talk += max(0.0, h["end"] - h["start"])
        rep.append({
            "brand": brand,
            "mentions": len(hits),
            "talk_time_s": int(talk),
            "clips_created": len(hits),  # placeholder: clip creation is separate export step
            "hits": hits
        })
    rep.sort(key=lambda x: (-x["mentions"], -x["talk_time_s"], x["brand"].lower()))
    job.summary = {
        "mode": "vod",
        "segments_in_memory": len(job.segments),
        "watch": job.watch,
        "sponsor_brands": [r["brand"] for r in rep],
    }
    job.push({"type": "sponsor_report", "report": rep})
    job.push({"type": "hidden_context", "items": job.hidden_context_hits})
    job.push({"type": "summary", "summary": job.summary})

def _download_audio(job: Job, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(out_dir / "audio.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_tmpl,
        "quiet": True,
        "noplaylist": True,
        "nocheckcertificate": True,
        "retries": 5,
        "consoletitle": False,
        "continuedl": True,
        "cachedir": False,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"},
        ],
        "postprocessor_args": ["-ac", "1", "-ar", "16000"],
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(job.vod_url, download=True)
        # yt-dlp returns final filename in requested_downloads or _filename; easiest: search for wav in folder
        for p in out_dir.glob("audio*.wav"):
            return p
    raise RuntimeError("Audio download failed")

def _split_wav(wav_path: Path, out_dir: Path, seg_s: int) -> List[Tuple[float, float, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Use ffmpeg segmenter; generates segment000.wav, ...
    pattern = str(out_dir / "seg%05d.wav")
    _run([
        "ffmpeg", "-y",
        "-i", str(wav_path),
        "-f", "segment",
        "-segment_time", str(seg_s),
        "-ac", "1", "-ar", "16000",
        pattern
    ])
    segs = sorted(out_dir.glob("seg*.wav"))
    result = []
    for i, p in enumerate(segs):
        start = i * seg_s
        end = start + seg_s
        result.append((start, end, p))
    return result

def _merge_and_emit(job: Job, start_s: float, end_s: float, text: str) -> None:
    text = _clean_text(text)
    if not text:
        return
    # Merge with previous if close + no punctuation end
    if job.segments:
        prev = job.segments[-1]
        prev_end = float(prev["end"])
        prev_text = prev["text"]
        if (start_s - prev_end) <= MERGE_GAP_S and not re.search(r"[.!?…]$", prev_text):
            merged = (prev_text + " " + text).strip()
            prev["end"] = end_s
            prev["text"] = merged
            prev["ts"] = _fmt_ts(float(prev["start"]))
            prev["link"] = _twitch_link(job.vod_url, float(prev["start"]))
            job.push({"type": "segment_update", "segment": prev})
            _update_sponsor(job, float(prev["start"]), float(prev["end"]), merged)
            return
    seg = {
        "start": start_s, "end": end_s,
        "ts": _fmt_ts(start_s),
        "text": text,
        "link": _twitch_link(job.vod_url, start_s)
    }
    job.segments.append(seg)
    job.push({"type": "segment", "segment": seg})
    _update_sponsor(job, start_s, end_s, text)

def _transcribe_chunk(chunk: Tuple[float, float, Path]) -> Tuple[float, float, str]:
    start_s, end_s, wav_path = chunk
    # skip if mostly silence
    try:
        if not _energy_vad_has_voice(wav_path):
            return (start_s, end_s, "")
    except Exception:
        pass

    model = get_model()
    segments, info = model.transcribe(
        str(wav_path),
        beam_size=1,
        best_of=1,
        temperature=0.0,
        vad_filter=True,
        word_timestamps=False,
    )
    texts = []
    for s in segments:
        t = _clean_text(s.text)
        if t:
            texts.append(t)
    return (start_s, end_s, " ".join(texts).strip())

def _worker(job: Job) -> None:
    job.status = "running"
    _emit_progress(job)

    job_dir = DATA / f"job_{job.id}"
    if job_dir.exists():
        # clean old
        for p in job_dir.glob("*"):
            if p.is_file():
                p.unlink()
            else:
                import shutil
                shutil.rmtree(p, ignore_errors=True)
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        job.push({"type": "log", "message": "Downloading audio..."})
        wav = _download_audio(job, job_dir / "dl")
        job.push({"type": "log", "message": "Splitting into chunks..."})
        chunks = _split_wav(wav, job_dir / "chunks", SEGMENT_SECONDS)
        job.total_chunks = len(chunks)
        job.done_chunks = 0
        t0 = time.time()

        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=max(1, WORKERS)) as ex:
            futures = {ex.submit(_transcribe_chunk, c): c for c in chunks}
            for fut in as_completed(futures):
                if job._stop.is_set():
                    job.status = "stopped"
                    break
                start_s, end_s, text = fut.result()
                _merge_and_emit(job, start_s, end_s, text)

                job.done_chunks += 1
                job.progress = job.done_chunks / max(1, job.total_chunks)
                # ETA from average time per chunk
                elapsed = time.time() - t0
                if job.done_chunks > 0:
                    per = elapsed / job.done_chunks
                    job.eta_s = per * (job.total_chunks - job.done_chunks)
                _emit_progress(job)

        if job.status != "stopped":
            job.status = "done"
            job.progress = 1.0
            job.eta_s = 0
            _emit_progress(job)
            _finalize_reports(job)
        job.push({"type": "done", "status": job.status})
    except Exception as e:
        job.status = "error"
        job.push({"type": "error", "message": str(e)})
        _emit_progress(job)
        job.push({"type": "done", "status": "error"})

# -----------------------------
# PDF export
# -----------------------------
def sponsor_report_pdf_bytes(job: Job) -> bytes:
    buf_path = DATA / f"job_{job.id}" / "sponsor_report.pdf"
    rep = []
    for brand, hits in job.sponsor_hits.items():
        talk = sum(max(0.0, h["end"] - h["start"]) for h in hits)
        rep.append((brand, len(hits), int(talk), hits))
    rep.sort(key=lambda x: (-x[1], -x[2], x[0].lower()))

    c = canvas.Canvas(str(buf_path), pagesize=A4)
    w, h = A4
    y = h - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "Sponsor Report")
    y -= 24
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"VOD: {job.vod_url}")
    y -= 18
    for brand, mentions, talk_s, hits in rep:
        if y < 80:
            c.showPage()
            y = h - 50
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, f"{brand} — Mentions: {mentions} — Talk time: {_fmt_ts(talk_s)}")
        y -= 16
        c.setFont("Helvetica", 9)
        for hh in hits[:50]:
            if y < 60:
                c.showPage()
                y = h - 50
                c.setFont("Helvetica", 9)
            line = f"{hh['ts']}  {hh['text']}"
            c.drawString(50, y, line[:120])
            y -= 12
        y -= 6
    c.save()
    data = buf_path.read_bytes()
    return data

# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/start")
async def api_start(payload: Dict[str, Any]):
    url = (payload.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="Missing url")
    watch = payload.get("watch") or []
    alerts = payload.get("alerts") or []
    job = Job(vod_url=url, watch=watch, alerts=alerts)
    with JOBS_LOCK:
        JOBS[job.id] = job
    t = threading.Thread(target=_worker, args=(job,), daemon=True)
    t.start()
    return {"job_id": job.id}

@app.post("/api/stop/{job_id}")
async def api_stop(job_id: str):
    job = ensure_job(job_id)
    job.stop()
    return {"ok": True}

@app.get("/api/events/{job_id}")
async def api_events(job_id: str):
    job = ensure_job(job_id)

    def gen():
        # send initial hello
        yield f"data: {json.dumps({'type':'hello','job_id':job.id,'vod_url':job.vod_url})}\n\n"
        while True:
            try:
                msg = job._events.get(timeout=15)
            except queue.Empty:
                # keep-alive
                yield "event: ping\ndata: {}\n\n"
                if job.status in ("done", "error", "stopped"):
                    # allow client to finish
                    continue
                continue
            yield f"data: {json.dumps(msg)}\n\n"
            if msg.get("type") == "done":
                break

    return StreamingResponse(gen(), media_type="text/event-stream")

@app.get("/api/search/{job_id}")
async def api_search(job_id: str, q: str = ""):
    job = ensure_job(job_id)
    q = (q or "").strip().lower()
    if not q:
        return {"results": []}
    results = []
    for seg in job.segments:
        if q in seg["text"].lower():
            results.append(seg)
    return {"results": results[:200]}

@app.post("/api/alerts/{job_id}")
async def api_alerts(job_id: str, payload: Dict[str, Any]):
    job = ensure_job(job_id)
    brands = payload.get("brands") or []
    job.alerts = [b.strip() for b in brands if str(b).strip()]
    return {"ok": True, "brands": job.alerts}

@app.get("/api/sponsor_report/{job_id}")
async def api_sponsor_report(job_id: str):
    job = ensure_job(job_id)
    rep = []
    for brand, hits in job.sponsor_hits.items():
        talk = sum(max(0.0, h["end"] - h["start"]) for h in hits)
        rep.append({"brand": brand, "mentions": len(hits), "talk_time_s": int(talk), "hits": hits})
    rep.sort(key=lambda x: (-x["mentions"], -x["talk_time_s"], x["brand"].lower()))
    return JSONResponse({"report": rep})

@app.get("/api/sponsor_report_pdf/{job_id}")
async def api_sponsor_report_pdf(job_id: str):
    job = ensure_job(job_id)
    pdf = sponsor_report_pdf_bytes(job)
    headers = {"Content-Disposition": f'attachment; filename="sponsor_report_{job.id}.pdf"'}
    return Response(content=pdf, media_type="application/pdf", headers=headers)

@app.get("/healthz")
async def healthz():
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
