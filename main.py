import os
import re
import json
import time
import uuid
import threading
import queue
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(ROOT / "templates"))

app = FastAPI(title="Hoermi")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


def _lazy_import_whisper():
    from faster_whisper import WhisperModel
    return WhisperModel


def _lazy_import_ytdlp():
    from yt_dlp import YoutubeDL
    return YoutubeDL


def _lazy_import_reportlab():
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    return A4, canvas


def _now() -> float:
    return time.time()


def _safe_job_dir(job_id: str) -> Path:
    p = DATA / job_id
    p.mkdir(parents=True, exist_ok=True)
    return p


JOBS_LOCK = threading.Lock()
JOBS: Dict[str, Dict[str, Any]] = {}


def _job_default(job_id: str) -> Dict[str, Any]:
    return {
        "id": job_id,
        "status": "queued",
        "created_at": _now(),
        "updated_at": _now(),
        "progress": 0.0,
        "eta_seconds": None,
        "vod_url": None,
        "error": None,
        "result": None,
    }


def ensure_job(job_id: str) -> Dict[str, Any]:
    with JOBS_LOCK:
        if job_id not in JOBS:
            JOBS[job_id] = _job_default(job_id)
        return JOBS[job_id]


def update_job(job_id: str, **kwargs: Any) -> None:
    with JOBS_LOCK:
        j = ensure_job(job_id)
        j.update(kwargs)
        j["updated_at"] = _now()


def read_job(job_id: str) -> Dict[str, Any]:
    with JOBS_LOCK:
        if job_id not in JOBS:
            raise KeyError(job_id)
        return dict(JOBS[job_id])


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.post("/api/job")
async def create_job(payload: Dict[str, Any]):
    vod_url = (payload or {}).get("vod_url")
    if not vod_url or not isinstance(vod_url, str):
        raise HTTPException(status_code=400, detail="vod_url fehlt")

    job_id = uuid.uuid4().hex
    ensure_job(job_id)
    update_job(job_id, status="queued", vod_url=vod_url, progress=0.0, eta_seconds=None)

    t = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
    t.start()

    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
async def get_job(job_id: str):
    try:
        return read_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job nicht gefunden")


@app.get("/api/sponsor_report/{job_id}")
async def api_sponsor_report(job_id: str):
    try:
        job = read_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job nicht gefunden")

    rep = (job.get("result") or {}).get("sponsor_report")
    return JSONResponse(rep or {"ok": False, "reason": "noch keine daten"})


@app.get("/api/sponsor_report_pdf/{job_id}")
async def api_sponsor_report_pdf(job_id: str):
    try:
        job = read_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job nicht gefunden")

    rep = (job.get("result") or {}).get("sponsor_report")
    if not rep:
        raise HTTPException(status_code=400, detail="noch keine daten")

    pdf = _pdf_bytes_for_report(rep)
    headers = {"Content-Disposition": f'attachment; filename="sponsor_report_{job_id}.pdf"'}
    return Response(content=pdf, media_type="application/pdf", headers=headers)


def _run_job(job_id: str) -> None:
    started = _now()
    try:
        update_job(job_id, status="running", progress=0.02, eta_seconds=999)

        job = read_job(job_id)
        vod_url = job["vod_url"]
        job_dir = _safe_job_dir(job_id)

        audio_path = job_dir / "audio.wav"
        update_job(job_id, progress=0.05)

        _download_audio(vod_url, audio_path)
        update_job(job_id, progress=0.20)

        segments = _transcribe_audio(audio_path, job_id=job_id)
        update_job(job_id, progress=0.75)

        sponsor_rep = _build_sponsor_report_stub(segments)

        result = {
            "segments": segments,
            "sponsor_report": sponsor_rep,
        }

        (job_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        dur = _now() - started
        update_job(job_id, status="done", progress=1.0, eta_seconds=0, result=result)
    except Exception as e:
        update_job(job_id, status="error", error=str(e), eta_seconds=None)


def _download_audio(vod_url: str, out_wav: Path) -> None:
    YoutubeDL = _lazy_import_ytdlp()

    tmp_dir = out_wav.parent
    tmp_template = str(tmp_dir / "dl.%(ext)s")

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "outtmpl": tmp_template,
        "format": "bestaudio/best",
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([vod_url])

    dl_wav = tmp_dir / "dl.wav"
    if not dl_wav.exists():
        raise RuntimeError("audio download fehlgeschlagen")
    dl_wav.replace(out_wav)


def _transcribe_audio(audio_path: Path, job_id: str) -> list[dict]:
    WhisperModel = _lazy_import_whisper()

    model_size = os.environ.get("WHISPER_MODEL", "small")
    compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

    model = WhisperModel(model_size, device="cpu", compute_type=compute_type)

    seg_seconds = int(os.environ.get("SEGMENT_SECONDS", "20"))
    seg_seconds = max(15, min(30, seg_seconds))

    lang = os.environ.get("WHISPER_LANG", "de")

    update_job(job_id, eta_seconds=300)

    segments_out: list[dict] = []

    seg_iter, info = model.transcribe(
        str(audio_path),
        language=lang,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
    )

    last_progress = 0.20
    last_t = _now()

    for s in seg_iter:
        text = (s.text or "").strip()
        if not text:
            continue

        text = re.sub(r"\s+", " ", text).strip()

        start = float(getattr(s, "start", 0.0))
        end = float(getattr(s, "end", 0.0))

        segments_out.append(
            {
                "start": start,
                "end": end,
                "text": text,
            }
        )

        now = _now()
        if now - last_t > 2.0:
            last_t = now
            p = min(0.70, last_progress + 0.01)
            last_progress = p
            update_job(job_id, progress=p, eta_seconds=max(5, int(180 * (1.0 - p))))

    merged = _merge_to_chunks(segments_out, target_seconds=seg_seconds)
    return merged


def _merge_to_chunks(segs: list[dict], target_seconds: int) -> list[dict]:
    if not segs:
        return []

    out: list[dict] = []
    cur = {"start": segs[0]["start"], "end": segs[0]["end"], "text": segs[0]["text"]}

    for s in segs[1:]:
        if (s["end"] - cur["start"]) <= float(target_seconds):
            cur["end"] = s["end"]
            cur["text"] = (cur["text"] + " " + s["text"]).strip()
        else:
            out.append(cur)
            cur = {"start": s["start"], "end": s["end"], "text": s["text"]}

    out.append(cur)
    return out


def _build_sponsor_report_stub(segments: list[dict]) -> Dict[str, Any]:
    brand = os.environ.get("SPONSOR_BRAND", "Bitpanda")
    mentions = []
    total = 0.0

    for s in segments:
        if brand.lower() in s["text"].lower():
            mentions.append(
                {
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"],
                }
            )
            total += max(0.0, float(s["end"]) - float(s["start"]))

    return {
        "brand": brand,
        "mentions": len(mentions),
        "total_talk_time_seconds": int(total),
        "clips_created": max(0, min(3, len(mentions))),
        "clips": mentions,
    }


def _pdf_bytes_for_report(rep: Dict[str, Any]) -> bytes:
    A4, canvas = _lazy_import_reportlab()

    import io
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    w, h = A4
    y = h - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Sponsor Report")
    y -= 30

    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Brand: {rep.get('brand', '')}")
    y -= 18
    c.drawString(50, y, f"Mentions: {rep.get('mentions', 0)}")
    y -= 18
    c.drawString(50, y, f"Total talk time: {rep.get('total_talk_time_seconds', 0)}s")
    y -= 24

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Clips")
    y -= 16

    c.setFont("Helvetica", 10)
    for m in rep.get("clips", []):
        line = f"{_fmt_ts(m.get('start', 0.0))}  {str(m.get('text', ''))[:120]}"
        c.drawString(50, y, line)
        y -= 14
        if y < 60:
            c.showPage()
            y = h - 60
            c.setFont("Helvetica", 10)

    c.showPage()
    c.save()
    return buf.getvalue()


def _fmt_ts(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
