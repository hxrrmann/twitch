import os
import uuid
import time
import json
import queue
import asyncio
import threading
import subprocess
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

app = FastAPI(title="Amar Stream AI")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse("templates/index.html")


MODEL = None
MODEL_LOCK = threading.Lock()

JOB_QUEUE = queue.Queue()
JOBS: Dict[str, dict] = {}
STREAMS: Dict[str, asyncio.Queue] = {}

MAX_STREAMS = 5
CHUNK_SECONDS = 30
BATCH_SIZE = 6

EXECUTOR = ThreadPoolExecutor(max_workers=5)


def get_model():

    global MODEL

    with MODEL_LOCK:

        if MODEL is None:

            device = os.environ.get("WHISPER_DEVICE", "cpu")

            if device == "cuda":

                MODEL = WhisperModel(
                    "distil-large-v3",
                    device="cuda",
                    compute_type="float16"
                )

            else:

                MODEL = WhisperModel(
                    "distil-large-v3",
                    device="cpu",
                    compute_type="int8"
                )

    return MODEL


def download_stream(url, job_id, chunk_queue):

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
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-f", "segment",
        "-segment_time", str(CHUNK_SECONDS),
        f"{chunk_dir}/chunk_%03d.wav"
    ]

    p1 = subprocess.Popen(ytdlp, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(ffmpeg, stdin=p1.stdout)

    if p1.stdout:
        p1.stdout.close()

    seen = set()

    while True:

        files = [
            os.path.join(chunk_dir, f)
            for f in os.listdir(chunk_dir)
            if f.endswith(".wav")
        ]

        for f in sorted(files):

            if f not in seen:
                seen.add(f)
                chunk_queue.put(f)

        if p2.poll() is not None:
            break

        time.sleep(1)


def transcribe_batch(paths):

    model = get_model()

    segments_out = []

    for p in paths:

        segments, _ = model.transcribe(
            p,
            beam_size=1,
            best_of=1,
            vad_filter=True
        )

        for s in segments:

            text = s.text.strip()

            if not text:
                continue

            segments_out.append({
                "text": text,
                "start": s.start,
                "end": s.end
            })

    return segments_out


async def push(job_id, data):

    if job_id in STREAMS:

        await STREAMS[job_id].put(data)


def process_batch(job_id, batch, alerts, transcript, talktime):

    segs = transcribe_batch(batch)

    for seg in segs:

        transcript.append(seg)

        ts = int(seg["start"])

        asyncio.run(push(job_id, {
            "type": "segment",
            "text": seg["text"],
            "timestamp": ts
        }))

        text_lower = seg["text"].lower()

        for alert in alerts:

            if alert.lower() in text_lower:

                dur = seg["end"] - seg["start"]

                talktime[alert] = talktime.get(alert, 0) + dur

                asyncio.run(push(job_id, {
                    "type": "mention",
                    "brand": alert,
                    "text": seg["text"],
                    "timestamp": ts
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

            EXECUTOR.submit(
                process_batch,
                job_id,
                batch.copy(),
                alerts,
                transcript,
                talktime
            )

            batch = []

    summary = " ".join(x["text"] for x in transcript)[:1500]

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


for _ in range(MAX_STREAMS):
    threading.Thread(target=scheduler, daemon=True).start()


@app.post("/api/start")
def start(payload: dict):

    url = payload.get("url")

    alerts = payload.get("alerts", "")

    if not url:

        raise HTTPException(400)

    alerts_list = [x.strip() for x in alerts.split(",") if x.strip()]

    job_id = str(uuid.uuid4())

    JOBS[job_id] = {
        "id": job_id,
        "url": url,
        "alerts": alerts_list
    }

    STREAMS[job_id] = asyncio.Queue()

    JOB_QUEUE.put(job_id)

    return {"job_id": job_id}


@app.get("/api/events/{job_id}")
async def events(job_id):

    async def gen():

        q = STREAMS[job_id]

        while True:

            data = await q.get()

            yield f"data: {json.dumps(data)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/health")
def health():

    return {"status": "ok"}


import uvicorn

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8080))

    uvicorn.run(app, host="0.0.0.0", port=port)
