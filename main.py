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
from concurrent.futures import ThreadPoolExecutor

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
BATCH_SIZE = 8
MAX_PARALLEL_STREAMS = 3

TRANSCRIBE_POOL = ThreadPoolExecutor(max_workers=3)

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
                    compute_type=compute_type
                )
            except Exception:
                MODEL = WhisperModel(
                    "distil-large-v3",
                    device="cpu",
                    compute_type="int8"
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


def parse_chunk_offset(path: str) -> int:
    m = re.search(r"chunk_(\d+)\.wav$", path)
    if not m:
        return 0
    return int(m.group(1)) * CHUNK_SECONDS


async def push(job_id: str, data: dict):
    if job_id in STREAMS:
        await STREAMS[job_id].put(data)


def download_stream(url: str, job_id: str, chunk_queue: queue.Queue):

    chunk_dir = f"/tmp/{job_id}"
    os.makedirs(chunk_dir, exist_ok=True)

    ytdlp = [
        "yt-dlp",
        "-f",
        "bestaudio/best",
        "-o",
        "-",
        url
    ]

    ffmpeg = [
        "ffmpeg",
        "-loglevel", "error",
        "-i", "pipe:0",
        "-vn",
        "-ac", "1",
        "-ar", str(AUDIO_RATE),
        "-c:a", "pcm_s16le",
        "-f", "segment",
        "-segment_time", str(CHUNK_SECONDS),
        "-reset_timestamps", "1",
        f"{chunk_dir}/chunk_%03d.wav"
    ]

    p1 = subprocess.Popen(ytdlp, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(ffmpeg, stdin=p1.stdout)

    if p1.stdout:
        p1.stdout.close()

    seen = set()

    while True:

        try:
            files = [
                os.path.join(chunk_dir, f)
                for f in os.listdir(chunk_dir)
                if f.endswith(".wav")
            ]
        except:
            files = []

        for f in sorted(files):
            if f not in seen:
                seen.add(f)
                chunk_queue.put(f)

        if p2.poll() is not None:
            break

        time.sleep(1)


def transcribe_batch(paths: List[str]) -> List[dict]:

    model = get_model()

    segments_out: List[dict] = []

    for path in paths:

        offset = parse_chunk_offset(path)

        segments, _ = model.transcribe(
            path,
            beam_size=1,
            best_of=1,
            vad_filter=True
        )

        for seg in segments:

            text = seg.text.strip()

            if not text:
                continue

            start = offset + float(seg.start)
            end = offset + float(seg.end)

            segments_out.append({
                "text": text,
                "start": start,
                "end": end
            })

    return segments_out


def process_batch(job_id, batch, alerts, transcript, talktime):

    segs = transcribe_batch(batch)

    for seg in segs:

        transcript.append(seg)

        ts = twitch_ts(seg["start"])

        asyncio.run(push(job_id, {
            "type": "segment",
            "text": seg["text"],
            "timestamp": ts,
            "seconds": seg["start"]
        }))

        text_lower = seg["text"].lower()

        for alert in alerts:

            if alert.lower() in text_lower:

                dur = seg["end"] - seg["start"]

                talktime[alert] = talktime.get(alert, 0) + dur

                asyncio.run(push(job_id, {
                    "type": "mention",
                    "brand": alert,
                    "timestamp": ts,
                    "text": seg["text"]
                }))


def stream_worker(job_id):

    job = JOBS[job_id]
    url = job["url"]
    alerts = job["alerts"]

    chunk_queue = queue.Queue()

    transcript = []
    talktime = {}

    batch = []

    dl = threading.Thread(
        target=download_stream,
        args=(url, job_id, chunk_queue),
        daemon=True
    )

    dl.start()

    while True:

        try:
            chunk = chunk_queue.get(timeout=5)
        except:
            if not dl.is_alive():
                break
            continue

        batch.append(chunk)

        if len(batch) >= BATCH_SIZE:

            TRANSCRIBE_POOL.submit(
                process_batch,
                job_id,
                batch.copy(),
                alerts,
                transcript,
                talktime
            )

            batch = []

    summary = " ".join(x["text"] for x in transcript)[:1200]

    asyncio.run(push(job_id, {
        "type": "summary",
        "text": summary
    }))


def scheduler():

    while True:

        job_id = JOB_QUEUE.get()

        threading.Thread(
            target=stream_worker,
            args=(job_id,),
            daemon=True
        ).start()


for _ in range(MAX_PARALLEL_STREAMS):
    threading.Thread(target=scheduler, daemon=True).start()


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
        "alerts": alerts
    }

    STREAMS[job_id] = asyncio.Queue()

    JOB_QUEUE.put(job_id)

    return {"job_id": job_id}


@app.get("/api/events/{job_id}")
async def events(job_id: str):

    if job_id not in STREAMS:
        raise HTTPException(404)

    async def gen():

        q = STREAMS[job_id]

        while True:

            data = await q.get()

            yield f"data: {json.dumps(data)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/health")
def health():
    return {"status": "ok"}
