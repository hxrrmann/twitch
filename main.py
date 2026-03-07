import os
import re
import uuid
import time
import json
import queue
import asyncio
import threading
import subprocess
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

app = FastAPI(title="Amar Stream AI")

STATIC_DIR = "static" if os.path.isdir("static") else "."
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def home():
    if os.path.exists("templates/index.html"):
        return FileResponse("templates/index.html")
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    raise HTTPException(500, "index file missing")


MODEL: Optional[WhisperModel] = None
MODEL_LOCK = threading.Lock()
JOB_QUEUE: "queue.Queue[str]" = queue.Queue()
JOBS: Dict[str, dict] = {}
STREAMS: Dict[str, asyncio.Queue] = {}

CHUNK_SECONDS = 30
BATCH_SIZE = 4
AUDIO_RATE = 16000


def get_model() -> WhisperModel:
    global MODEL
    with MODEL_LOCK:
        if MODEL is None:
            device = os.environ.get("WHISPER_DEVICE", "cuda")
            compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
            try:
                MODEL = WhisperModel(
                    "distil-large-v3",
                    device=device,
                    compute_type=compute_type,
                )
            except Exception:
                MODEL = WhisperModel(
                    "distil-large-v3",
                    device="cpu",
                    compute_type="int8",
                )
        return MODEL


def sanitize_alerts(alerts: str) -> List[str]:
    return [x.strip() for x in alerts.split(",") if x.strip()]


def twitch_ts(sec: float) -> str:
    total = max(0, int(sec))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def eta_format(sec: float) -> str:
    if sec <= 0:
        return "0s"
    m = int(sec // 60)
    s = int(sec % 60)
    if m <= 0:
        return f"{s}s"
    return f"{m}m {s}s"


def llm_summary(text: str) -> str:
    provider = os.environ.get("LLM_PROVIDER", "none")

    if provider == "openai":
        try:
            from openai import OpenAI

            client = OpenAI()
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Create a concise stream summary with the most important topics, "
                            "brand mentions and highlights. Text: " + text[:6000]
                        ),
                    }
                ],
            )
            return (r.choices[0].message.content or "").strip()
        except Exception:
            pass

    if provider == "anthropic":
        try:
            import anthropic

            client = anthropic.Anthropic()
            r = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Create a concise stream summary with the most important topics, "
                            "brand mentions and highlights. Text: " + text[:6000]
                        ),
                    }
                ],
            )
            return r.content[0].text.strip()
        except Exception:
            pass

    words = text.split()[:140]
    return " ".join(words) if words else "Keine Zusammenfassung verfügbar."


def parse_chunk_offset(path: str) -> int:
    m = re.search(r"chunk_(\d+)\.wav$", path)
    if not m:
        return 0
    return int(m.group(1)) * CHUNK_SECONDS


def update_job(job_id: str, **kwargs):
    if job_id in JOBS:
        JOBS[job_id].update(kwargs)


async def push(job_id: str, data: dict):
    if job_id in STREAMS:
        await STREAMS[job_id].put(data)


async def push_status(job_id: str):
    if job_id in JOBS:
        await push(job_id, {"type": "status", **JOBS[job_id]})


async def push_error(job_id: str, message: str):
    update_job(job_id, status="error", step="failed", error=message)
    await push(job_id, {"type": "error", "message": message})
    await push_status(job_id)



def download_stream(url: str, job_id: str, chunk_queue: queue.Queue):
    chunk_dir = f"/tmp/{job_id}"
    os.makedirs(chunk_dir, exist_ok=True)

    ytdlp = [
        "yt-dlp",
        "-f",
        "bestaudio/best",
        "-o",
        "-",
        url,
    ]

    ffmpeg = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(AUDIO_RATE),
        "-c:a",
        "pcm_s16le",
        "-f",
        "segment",
        "-segment_time",
        str(CHUNK_SECONDS),
        "-reset_timestamps",
        "1",
        f"{chunk_dir}/chunk_%03d.wav",
    ]

    p1 = subprocess.Popen(ytdlp, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p2 = subprocess.Popen(ffmpeg, stdin=p1.stdout, stderr=subprocess.PIPE)

    if p1.stdout is not None:
        p1.stdout.close()

    seen = set()

    while True:
        if JOBS.get(job_id, {}).get("cancel"):
            for proc in (p1, p2):
                try:
                    proc.terminate()
                except Exception:
                    pass
            break

        try:
            files = [
                os.path.join(chunk_dir, f)
                for f in os.listdir(chunk_dir)
                if f.endswith(".wav")
            ]
        except FileNotFoundError:
            files = []

        for file_path in sorted(files):
            if file_path not in seen:
                seen.add(file_path)
                chunk_queue.put(file_path)

        if p2.poll() is not None:
            break

        time.sleep(1)

    stderr_yt = p1.stderr.read().decode("utf-8", errors="ignore") if p1.stderr else ""
    stderr_ff = p2.stderr.read().decode("utf-8", errors="ignore") if p2.stderr else ""

    if JOBS.get(job_id, {}).get("cancel"):
        return

    if p1.returncode not in (0, None):
        raise RuntimeError(stderr_yt.strip() or "yt-dlp konnte den Stream nicht laden")

    if p2.returncode not in (0, None):
        raise RuntimeError(stderr_ff.strip() or "ffmpeg konnte den Audio Stream nicht verarbeiten")



def transcribe_batch(paths: List[str]) -> List[dict]:
    model = get_model()
    segments_out: List[dict] = []

    for path in paths:
        base_offset = parse_chunk_offset(path)
        segments, _ = model.transcribe(
            path,
            beam_size=1,
            best_of=1,
            vad_filter=True,
            language="de",
        )

        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            start = base_offset + float(seg.start)
            end = base_offset + float(seg.end)
            segments_out.append({"text": text, "start": start, "end": end})

    return segments_out



def process_batch(job_id: str, batch: List[str], alerts: List[str], transcript: List[dict], talktime: Dict[str, float], start_time: float, processed_chunks: int) -> int:
    segs = transcribe_batch(batch)

    for seg in segs:
        transcript.append(seg)
        timestamp = twitch_ts(seg["start"])
        asyncio.run(
            push(
                job_id,
                {
                    "type": "segment",
                    "text": seg["text"],
                    "timestamp": timestamp,
                    "seconds": seg["start"],
                },
            )
        )

        text_lower = seg["text"].lower()
        for alert in alerts:
            if alert.lower() in text_lower:
                duration = max(0.0, seg["end"] - seg["start"])
                talktime[alert] = talktime.get(alert, 0.0) + duration
                asyncio.run(
                    push(
                        job_id,
                        {
                            "type": "mention",
                            "brand": alert,
                            "text": seg["text"],
                            "timestamp": timestamp,
                            "seconds": seg["start"],
                        },
                    )
                )

    processed_chunks += len(batch)
    elapsed = max(1.0, time.time() - start_time)
    speed = processed_chunks / elapsed
    update_job(
        job_id,
        percent=min(99, processed_chunks * 3),
        eta=eta_format((60 / speed) if speed > 0 else 0),
        step="transcribing",
        transcript_count=len(transcript),
    )
    asyncio.run(push_status(job_id))
    return processed_chunks



def worker():
    while True:
        job_id = JOB_QUEUE.get()
        job = JOBS[job_id]
        url = job["url"]
        alerts = job["alerts"]

        chunk_queue: queue.Queue = queue.Queue()
        transcript: List[dict] = []
        talktime: Dict[str, float] = {}
        processed = 0
        batch: List[str] = []
        start_time = time.time()
        download_error = None

        update_job(job_id, step="starting", status="running", percent=1, error="")
        asyncio.run(push_status(job_id))

        def run_download():
            nonlocal download_error
            try:
                download_stream(url, job_id, chunk_queue)
            except Exception as exc:
                download_error = str(exc)

        dl = threading.Thread(target=run_download, daemon=True)
        dl.start()

        while True:
            if JOBS.get(job_id, {}).get("cancel"):
                update_job(job_id, percent=100, eta="0s", step="stopped", status="stopped")
                asyncio.run(push_status(job_id))
                break

            try:
                chunk = chunk_queue.get(timeout=3)
                batch.append(chunk)
            except queue.Empty:
                if download_error:
                    asyncio.run(push_error(job_id, download_error))
                    break
                if not dl.is_alive():
                    if batch:
                        try:
                            processed = process_batch(job_id, batch, alerts, transcript, talktime, start_time, processed)
                        except Exception as exc:
                            asyncio.run(push_error(job_id, f"Transkription fehlgeschlagen: {exc}"))
                            break
                        batch = []
                    break
                continue

            if len(batch) >= BATCH_SIZE:
                try:
                    processed = process_batch(job_id, batch, alerts, transcript, talktime, start_time, processed)
                except Exception as exc:
                    asyncio.run(push_error(job_id, f"Transkription fehlgeschlagen: {exc}"))
                    break
                batch = []

        if JOBS.get(job_id, {}).get("status") in {"error", "stopped"}:
            continue

        summary_text = llm_summary(" ".join(item["text"] for item in transcript))
        sponsor_report = {
            brand: {
                "seconds": round(seconds, 1),
                "time": eta_format(seconds),
            }
            for brand, seconds in sorted(talktime.items(), key=lambda item: item[1], reverse=True)
        }

        update_job(
            job_id,
            percent=100,
            eta="0s",
            step="completed",
            status="finished",
            transcript_count=len(transcript),
            sponsor_talktime=sponsor_report,
            summary=summary_text,
        )
        asyncio.run(push(
            job_id,
            {
                "type": "summary",
                "text": summary_text,
                "sponsor_talktime": sponsor_report,
            },
        ))
        asyncio.run(push_status(job_id))


threading.Thread(target=worker, daemon=True).start()


@app.post("/api/start")
def start(payload: dict):
    url = (payload.get("url") or "").strip()
    alerts = sanitize_alerts(payload.get("alerts", ""))

    if not url:
        raise HTTPException(400, "url missing")

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "id": job_id,
        "url": url,
        "alerts": alerts,
        "percent": 0,
        "eta": "",
        "step": "queued",
        "status": "running",
        "error": "",
        "cancel": False,
        "summary": "",
        "sponsor_talktime": {},
        "transcript_count": 0,
    }
    STREAMS[job_id] = asyncio.Queue()
    JOB_QUEUE.put(job_id)
    return {"job_id": job_id}


@app.post("/api/stop/{job_id}")
def stop(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "job missing")
    JOBS[job_id]["cancel"] = True
    JOBS[job_id]["status"] = "stopping"
    JOBS[job_id]["step"] = "stopping"
    return {"ok": True}


@app.get("/api/events/{job_id}")
async def events(job_id: str):
    if job_id not in STREAMS:
        raise HTTPException(404, "job missing")

    async def gen():
        yield f"data: {json.dumps({'type': 'hello', 'job_id': job_id})}\n\n"
        q = STREAMS[job_id]
        while True:
            data = await q.get()
            yield f"data: {json.dumps(data)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/status/{job_id}")
def status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "job missing")
    return JOBS[job_id]


@app.get("/api/health")
def health():
    return {"status": "ok"}
