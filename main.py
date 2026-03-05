import os
import re
import json
import time
import uuid
import wave
import queue
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from faster_whisper import WhisperModel
from yt_dlp import YoutubeDL
import webrtcvad

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

app = FastAPI(title="Hoermi")
templates = Jinja2Templates(directory=str(ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


SEGMENT_SECONDS_MIN = int(os.environ.get("SEGMENT_SECONDS_MIN", "15"))
SEGMENT_SECONDS_MAX = int(os.environ.get("SEGMENT_SECONDS_MAX", "30"))
WORKERS = int(os.environ.get("WORKERS", "4"))

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "int8")

VAD_MODE = int(os.environ.get("VAD_MODE", "2"))
VAD_RATIO_MIN = float(os.environ.get("VAD_RATIO_MIN", "0.12"))

MIN_CHARS = int(os.environ.get("MIN_CHARS", "14"))
MIN_WORDS = int(os.environ.get("MIN_WORDS", "3"))

SPONSOR_BRANDS = ["Bitpanda", "More Nutrition"]


whisper = WhisperModel(
    WHISPER_MODEL,
    device="cpu",
    compute_type=COMPUTE_TYPE,
)

vad = webrtcvad.Vad(VAD_MODE)

jobs: Dict[str, dict] = {}


def now_ts() -> float:
    return time.time()


def clamp_int(v: int, a: int, b: int) -> int:
    return max(a, min(b, v))


def segment_seconds() -> int:
    if SEGMENT_SECONDS_MAX < SEGMENT_SECONDS_MIN:
        return SEGMENT_SECONDS_MIN
    span = SEGMENT_SECONDS_MAX - SEGMENT_SECONDS_MIN
    if span == 0:
        return SEGMENT_SECONDS_MIN
    return SEGMENT_SECONDS_MIN + (span // 2)


def seconds_to_hms(seconds: float) -> str:
    s = int(max(0, seconds))
    h = s // 3600
    m = (s % 3600) // 60
    r = s % 60
    return f"{h}h{m:02d}m{r:02d}s"


def twitch_link(url: str, seconds: float) -> str:
    t = seconds_to_hms(seconds)
    joiner = "&" if "?" in url else "?"
    return f"{url}{joiner}t={t}"


def clean_text(t: str) -> str:
    t = (t or "").strip()
    if not t:
        return ""
    if len(t) < MIN_CHARS:
        return ""
    if len(t.split()) < MIN_WORDS:
        return ""
    low = t.lower().strip()

    bad = {
        "a", "ah", "äh", "eh", "hm", "hmm", "mhm", "uh", "um", "ok", "okay",
        "of", "the", "and", "to", "in", "on"
    }
    if low in bad:
        return ""

    if re.fullmatch(r"[a-zA-Z]{1,3}", t):
        return ""

    return t


def extract_direct_audio_url(vod_url: str) -> Tuple[str, Optional[float]]:
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "format": "bestaudio/best",
        "noplaylist": True,
        "nocheckcertificate": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(vod_url, download=False)
        direct = info.get("url")
        dur = info.get("duration")
        if not direct:
            raise RuntimeError("direct url missing")
        return direct, float(dur) if dur else None


def start_ffmpeg_segmenter(direct_audio_url: str, out_template: str, seg_s: int) -> subprocess.Popen:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", direct_audio_url,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        "-f", "segment",
        "-segment_time", str(seg_s),
        "-reset_timestamps", "1",
        out_template,
    ]
    return subprocess.Popen(cmd)


def wav_speech_ratio(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wf:
            ch = wf.getnchannels()
            rate = wf.getframerate()
            sw = wf.getsampwidth()
            if ch != 1 or rate != 16000 or sw != 2:
                return 1.0

            frame_ms = 30
            frame_len = int(rate * frame_ms / 1000)
            step = frame_len * 2
            raw = wf.readframes(wf.getnframes())
            if step <= 0 or len(raw) < step:
                return 0.0

            total = 0
            speech = 0
            for i in range(0, len(raw) - step + 1, step):
                frame = raw[i:i + step]
                total += 1
                if vad.is_speech(frame, rate):
                    speech += 1

            if total == 0:
                return 0.0
            return speech / total
    except Exception:
        return 1.0


def transcribe_wav(path: Path, offset: float) -> List[dict]:
    ratio = wav_speech_ratio(path)
    if ratio < VAD_RATIO_MIN:
        return []

    segments, _info = whisper.transcribe(
        str(path),
        beam_size=5,
        temperature=0.0,
        vad_filter=False,
        condition_on_previous_text=False,
        word_timestamps=False,
    )

    out: List[dict] = []
    for seg in segments:
        text = clean_text(seg.text)
        if not text:
            continue
        s = float(seg.start) + offset
        e = float(seg.end) + offset
        out.append({"start": s, "end": e, "text": text})
    return out


def sponsor_mentions_from_text(text: str) -> List[str]:
    found: List[str] = []
    low = (text or "").lower()
    for b in SPONSOR_BRANDS:
        if b.lower() in low:
            found.append(b)
    return found


def ensure_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


def new_job(vod_url: str) -> str:
    job_id = uuid.uuid4().hex[:10]
    job_dir = DATA / job_id
    chunks_dir = job_dir / "chunks"
    job_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    jobs[job_id] = {
        "id": job_id,
        "url": vod_url,
        "dir": job_dir,
        "chunks_dir": chunks_dir,
        "segments": [],
        "sponsor_events": [],
        "status": "starting",
        "progress": 0.0,
        "eta_seconds": None,
        "duration": None,
        "created_at": now_ts(),
        "done": False,
        "sse_q": queue.Queue(),
        "work_q": queue.Queue(),
        "seg_s": segment_seconds(),
        "processed_chunks": 0,
    }

    threading.Thread(target=pipeline_thread, args=(job_id,), daemon=True).start()
    return job_id


def push_event(job: dict, payload: dict) -> None:
    job["sse_q"].put(payload)


def update_progress(job: dict) -> None:
    dur = job.get("duration")
    seg_s = float(job.get("seg_s", 20))
    processed = int(job.get("processed_chunks", 0))

    if not dur or dur <= 0:
        job["progress"] = 0.0
        job["eta_seconds"] = None
        return

    total_chunks = max(1, int((dur + seg_s - 1) // seg_s))
    prog = processed / total_chunks
    prog = max(0.0, min(1.0, prog))
    job["progress"] = prog

    elapsed = now_ts() - float(job["created_at"])
    if prog > 0.03 and elapsed > 2:
        total_est = elapsed / prog
        eta = max(0.0, total_est - elapsed)
        job["eta_seconds"] = eta
    else:
        job["eta_seconds"] = None


def pipeline_thread(job_id: str) -> None:
    job = jobs[job_id]
    vod_url = job["url"]
    chunks_dir: Path = job["chunks_dir"]
    seg_s: int = int(job["seg_s"])

    job["status"] = "loading"
    push_event(job, {"type": "status", "status": job["status"], "progress": job["progress"], "eta_seconds": job["eta_seconds"]})

    direct, dur = extract_direct_audio_url(vod_url)
    job["duration"] = dur

    out_template = str(chunks_dir / "chunk_%08d.wav")
    proc = start_ffmpeg_segmenter(direct, out_template, seg_s)

    def feeder() -> None:
        last_idx = -1
        q = job["work_q"]

        while True:
            files = sorted(chunks_dir.glob("chunk_*.wav"))
            for f in files:
                try:
                    idx = int(f.stem.split("_")[1])
                except Exception:
                    continue
                if idx <= last_idx:
                    continue
                q.put((idx, f))
                last_idx = idx

            if proc.poll() is not None:
                break

            time.sleep(0.2)

        for _ in range(WORKERS):
            q.put((None, None))

    def worker() -> None:
        q = job["work_q"]
        while True:
            idx, wav_path = q.get()
            if idx is None:
                return

            offset = float(idx) * float(seg_s)
            segs = transcribe_wav(wav_path, offset)

            if segs:
                for s in segs:
                    s["link"] = twitch_link(vod_url, s["start"])
                    job["segments"].append(s)

                    for brand in sponsor_mentions_from_text(s["text"]):
                        ev = {
                            "brand": brand,
                            "start": s["start"],
                            "end": s["end"],
                            "text": s["text"],
                            "link": s["link"],
                        }
                        job["sponsor_events"].append(ev)
                        push_event(job, {"type": "sponsor_event", "event": ev})

                    push_event(job, {"type": "segment", "segment": s})

            job["processed_chunks"] = int(job.get("processed_chunks", 0)) + 1
            update_progress(job)
            push_event(job, {"type": "progress", "status": "transcribing", "progress": job["progress"], "eta_seconds": job["eta_seconds"]})

    job["status"] = "transcribing"
    push_event(job, {"type": "status", "status": job["status"], "progress": job["progress"], "eta_seconds": job["eta_seconds"]})

    threading.Thread(target=feeder, daemon=True).start()
    for _ in range(WORKERS):
        threading.Thread(target=worker, daemon=True).start()

    while proc.poll() is None:
        time.sleep(0.5)

    while True:
        qsize = job["work_q"].qsize()
        if qsize == 0:
            time.sleep(1.0)
            if job["work_q"].qsize() == 0:
                break
        else:
            time.sleep(1.0)

    job["done"] = True
    job["status"] = "done"
    update_progress(job)
    push_event(job, {"type": "done", "status": job["status"], "progress": 1.0, "eta_seconds": 0.0})


def sponsor_report(job: dict) -> dict:
    events = list(job.get("sponsor_events", []))

    per_brand: Dict[str, dict] = {}
    for b in SPONSOR_BRANDS:
        per_brand[b] = {"brand": b, "mentions": 0, "talk_time_seconds": 0.0, "clips_created": 0, "items": []}

    for ev in events:
        b = ev.get("brand")
        if b not in per_brand:
            continue
        per_brand[b]["mentions"] += 1
        dt = float(ev.get("end", 0.0)) - float(ev.get("start", 0.0))
        if dt > 0:
            per_brand[b]["talk_time_seconds"] += dt
        per_brand[b]["items"].append(ev)

    for b in SPONSOR_BRANDS:
        items = per_brand[b]["items"]
        per_brand[b]["clips_created"] = len(items)

    return {
        "job_id": job["id"],
        "url": job["url"],
        "brands": [per_brand[b] for b in SPONSOR_BRANDS],
    }


def pdf_bytes_for_report(report: dict) -> bytes:
    from io import BytesIO
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    x = 40
    y = h - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "Sponsor Report")
    y -= 22

    c.setFont("Helvetica", 10)
    c.drawString(x, y, f"VOD: {report.get('url', '')}")
    y -= 18

    for b in report.get("brands", []):
        if y < 120:
            c.showPage()
            y = h - 50

        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, f"Brand: {b.get('brand')}")
        y -= 14

        c.setFont("Helvetica", 10)
        mentions = int(b.get("mentions", 0))
        talk = float(b.get("talk_time_seconds", 0.0))
        clips = int(b.get("clips_created", 0))

        c.drawString(x, y, f"Mentions: {mentions}")
        y -= 12
        c.drawString(x, y, f"Total talk time: {seconds_to_hms(talk)}")
        y -= 12
        c.drawString(x, y, f"Clips created: {clips}")
        y -= 16

        c.setFont("Helvetica-Bold", 10)
        c.drawString(x, y, "Mentions with timestamps")
        y -= 12

        c.setFont("Helvetica", 9)
        for item in b.get("items", []):
            if y < 80:
                c.showPage()
                y = h - 50
                c.setFont("Helvetica", 9)

            start = seconds_to_hms(float(item.get("start", 0.0)))
            link = item.get("link", "")
            text = (item.get("text", "") or "").strip()
            if len(text) > 90:
                text = text[:90] + "..."

            c.drawString(x, y, f"{start}  {text}")
            y -= 11
            c.setFont("Helvetica-Oblique", 8)
            c.drawString(x, y, link)
            y -= 12
            c.setFont("Helvetica", 9)

        y -= 8

    c.showPage()
    c.save()
    return buf.getvalue()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/start")
async def api_start(payload: dict, response: Response):
    url = (payload or {}).get("url")
    if not url:
        raise HTTPException(400, "missing url")
    job_id = new_job(url)
    response.set_cookie("job_id", job_id)
    return {"job_id": job_id}


@app.get("/api/events/{job_id}")
async def api_events(job_id: str):
    job = ensure_job(job_id)

    def stream():
        q = job["sse_q"]
        while True:
            item = q.get()
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
            if isinstance(item, dict) and item.get("type") == "done":
                return

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/segments/{job_id}")
async def api_segments(job_id: str):
    job = ensure_job(job_id)
    return JSONResponse(
        {
            "status": job.get("status"),
            "progress": job.get("progress"),
            "eta_seconds": job.get("eta_seconds"),
            "segments": job.get("segments", []),
            "done": job.get("done", False),
        }
    )


@app.get("/api/sponsor_report/{job_id}")
async def api_sponsor_report(job_id: str):
    job = ensure_job(job_id)
    return JSONResponse(sponsor_report(job))


@app.get("/api/sponsor_report_pdf/{job_id}")
async def api_sponsor_report_pdf(job_id: str):
    job = ensure_job(job_id)
    rep = sponsor_report(job)
    pdf = pdf_bytes_for_report(rep)
    headers = {"Content-Disposition": f'attachment; filename="sponsor_report_{job_id}.pdf"'}
    return Response(content=pdf, media_type="application/pdf", headers=headers)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
