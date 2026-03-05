import os
import json
import uuid
import time
import queue
import threading
import subprocess
import wave
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from faster_whisper import WhisperModel
from yt_dlp import YoutubeDL
import webrtcvad


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

SEGMENT_SECONDS = int(os.environ.get("SEGMENT_SECONDS", "20"))
WORKERS = int(os.environ.get("WORKERS", "4"))

VAD_MODE = int(os.environ.get("VAD_MODE", "2"))
VAD_RATIO_MIN = float(os.environ.get("VAD_RATIO_MIN", "0.12"))

MIN_WORDS = int(os.environ.get("MIN_WORDS", "3"))
MIN_CHARS = int(os.environ.get("MIN_CHARS", "10"))
MIN_SEGMENT_SECONDS = float(os.environ.get("MIN_SEGMENT_SECONDS", "0.8"))

MODEL_NAME = os.environ.get("WHISPER_MODEL", "base")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "int8")

app = FastAPI(title="Hoermi VOD Analyzer")

templates = Jinja2Templates(directory=str(ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

jobs: Dict[str, dict] = {}

whisper = WhisperModel(MODEL_NAME, device="cpu", compute_type=COMPUTE_TYPE)
vad = webrtcvad.Vad(VAD_MODE)


def _clean_text(t: str) -> str:
    t = (t or "").strip()
    if not t:
        return ""
    low = t.lower().strip()

    bad = {
        "a", "ah", "äh", "eh", "hm", "hmm", "mhm", "uh", "um", "jo", "ja", "ok", "okay",
        "of", "the", "and", "to", "in", "on"
    }
    if low in bad:
        return ""

    if len(t) < MIN_CHARS:
        return ""

    if len(t.split()) < MIN_WORDS:
        return ""

    return t


def _seconds_to_hms(seconds: float) -> str:
    s = int(max(0, seconds))
    h = s // 3600
    m = (s % 3600) // 60
    r = s % 60
    return f"{h}h{m:02d}m{r:02d}s"


def _twitch_ts_link(url: str, seconds: float) -> str:
    ts = _seconds_to_hms(seconds)
    joiner = "&" if "?" in url else "?"
    return f"{url}{joiner}t={ts}"


def _get_audio_direct_url(vod_url: str) -> Tuple[str, Optional[float]]:
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "format": "bestaudio/best",
        "noplaylist": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(vod_url, download=False)
        direct = info.get("url")
        dur = info.get("duration")
        if not direct:
            raise RuntimeError("no direct audio url")
        return direct, float(dur) if dur else None


def _start_ffmpeg_segmenter(direct_audio_url: str, out_template: str) -> subprocess.Popen:
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
        "-segment_time", str(SEGMENT_SECONDS),
        "-reset_timestamps", "1",
        out_template,
    ]
    return subprocess.Popen(cmd)


def _wav_has_speech(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            rate = wf.getframerate()
            sampwidth = wf.getsampwidth()

            if channels != 1 or rate != 16000 or sampwidth != 2:
                return True

            frame_ms = 30
            frame_len = int(rate * frame_ms / 1000)
            raw = wf.readframes(wf.getnframes())

            step = frame_len * 2
            if step <= 0:
                return False

            total = 0
            speech = 0

            for i in range(0, len(raw) - step + 1, step):
                frame = raw[i:i + step]
                total += 1
                if vad.is_speech(frame, rate):
                    speech += 1

            if total == 0:
                return False

            ratio = speech / total
            return ratio >= VAD_RATIO_MIN
    except Exception:
        return True


def _transcribe_wav(path: Path, offset_seconds: float) -> List[dict]:
    if not _wav_has_speech(path):
        return []

    segments, info = whisper.transcribe(
        str(path),
        beam_size=5,
        vad_filter=False,
        condition_on_previous_text=False,
        temperature=0.0,
    )

    out: List[dict] = []
    for seg in segments:
        if (seg.end - seg.start) < MIN_SEGMENT_SECONDS:
            continue

        text = _clean_text(seg.text)
        if not text:
            continue

        start = float(seg.start) + offset_seconds
        end = float(seg.end) + offset_seconds

        out.append(
            {
                "start": start,
                "end": end,
                "text": text,
            }
        )

    return out


def _ensure_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


def _new_job(vod_url: str) -> str:
    job_id = uuid.uuid4().hex[:10]
    job_dir = DATA / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    jobs[job_id] = {
        "id": job_id,
        "url": vod_url,
        "dir": job_dir,
        "segments": [],
        "sse_q": queue.Queue(),
        "work_q": queue.Queue(),
        "started_at": time.time(),
        "done": False,
    }

    threading.Thread(target=_pipeline_thread, args=(job_id,), daemon=True).start()
    return job_id


def _pipeline_thread(job_id: str) -> None:
    job = jobs[job_id]
    vod_url = job["url"]

    chunks_dir = job["dir"] / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    direct, dur = _get_audio_direct_url(vod_url)

    out_template = str(chunks_dir / "chunk_%08d.wav")
    proc = _start_ffmpeg_segmenter(direct, out_template)

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

            time.sleep(0.25)

        for _ in range(WORKERS):
            q.put((None, None))

        job["done"] = True
        job["sse_q"].put({"done": True, "duration": dur})

    def worker() -> None:
        q = job["work_q"]
        while True:
            idx, wav_path = q.get()
            if idx is None:
                return

            offset = float(idx) * float(SEGMENT_SECONDS)

            segs = _transcribe_wav(wav_path, offset)
            for s in segs:
                s["link"] = _twitch_ts_link(vod_url, s["start"])
                job["segments"].append(s)
                job["sse_q"].put({"segment": s})

    threading.Thread(target=feeder, daemon=True).start()
    for _ in range(WORKERS):
        threading.Thread(target=worker, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/start")
async def api_start(payload: dict, response: Response):
    url = (payload or {}).get("url")
    if not url:
        raise HTTPException(400, "missing url")

    job_id = _new_job(url)
    response.set_cookie("job_id", job_id)
    return {"job_id": job_id}


@app.get("/api/events/{job_id}")
async def api_events(job_id: str):
    job = _ensure_job(job_id)

    def event_stream():
        q = job["sse_q"]
        while True:
            item = q.get()
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
            if isinstance(item, dict) and item.get("done") is True:
                return

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/segments/{job_id}")
async def api_segments(job_id: str):
    job = _ensure_job(job_id)
    return {"segments": job["segments"], "done": job["done"]}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
