
import os
import uuid
import time
import json
import queue
import threading
import subprocess
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

PORT = int(os.environ.get("PORT", 8080))

app = FastAPI(title="Amar Stream AI")

app.mount("/static", StaticFiles(directory="static"), name="static")

GPU_MODEL = WhisperModel(
    "distil-large-v3",
    device="cuda",
    compute_type="float16"
)

JOB_QUEUE = queue.Queue()
ACTIVE_JOBS: Dict[str, dict] = {}

BATCH_SIZE = 4
CHUNK_SECONDS = 40

def eta_format(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s"

def download_stream(url, job_id, chunk_queue):

    chunk_dir = f"/tmp/{job_id}"
    os.makedirs(chunk_dir, exist_ok=True)

    ytdlp = ["yt-dlp", "-o", "-", url]

    ffmpeg = [
        "ffmpeg",
        "-loglevel","quiet",
        "-i","pipe:0",
        "-f","segment",
        "-segment_time",str(CHUNK_SECONDS),
        "-c","copy",
        f"{chunk_dir}/chunk_%03d.wav"
    ]

    p1 = subprocess.Popen(ytdlp, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(ffmpeg, stdin=p1.stdout)
    p1.stdout.close()

    seen = set()

    while True:

        files = [
            os.path.join(chunk_dir,f)
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

    results = []

    for p in paths:

        segments, _ = GPU_MODEL.transcribe(
            p,
            beam_size=1,
            best_of=1,
            vad_filter=True
        )

        text = ""

        for s in segments:
            text += s.text + " "

        results.append(text.strip())

    return results

def worker_loop():

    while True:

        job_id, url, alerts = JOB_QUEUE.get()

        job = ACTIVE_JOBS[job_id]

        chunk_queue = queue.Queue()

        dl_thread = threading.Thread(
            target=download_stream,
            args=(url, job_id, chunk_queue)
        )

        dl_thread.start()

        transcript = []
        brands = []

        processed = 0
        batch = []

        start = time.time()

        while True:

            try:
                chunk = chunk_queue.get(timeout=5)
            except:
                if not dl_thread.is_alive():
                    break
                continue

            batch.append(chunk)

            if len(batch) >= BATCH_SIZE:

                texts = transcribe_batch(batch)

                for t in texts:

                    transcript.append(t)

                    for w in alerts:
                        if w.lower() in t.lower():
                            brands.append({"word": w, "text": t})

                processed += len(batch)

                elapsed = time.time() - start
                speed = processed / elapsed if elapsed > 0 else 0
                eta = eta_format(60/speed if speed > 0 else 0)

                job["percent"] = min(99, processed * 2)
                job["eta"] = eta
                job["step"] = "transcribing"
                job["transcript"] = transcript
                job["brands"] = brands

                batch = []

        job["percent"] = 100
        job["step"] = "completed"
        job["status"] = "finished"

threading.Thread(target=worker_loop, daemon=True).start()

@app.get("/", response_class=HTMLResponse)
def home():
    with open("templates/index.html") as f:
        return f.read()

@app.post("/api/start")
def start(payload: dict):

    url = payload.get("url")
    alerts = payload.get("alerts","")

    if not url:
        raise HTTPException(400, "url missing")

    words = [x.strip() for x in alerts.split(",") if x.strip()]

    job_id = str(uuid.uuid4())

    ACTIVE_JOBS[job_id] = {
        "id": job_id,
        "percent": 0,
        "eta": "",
        "step": "queued",
        "status": "running",
        "transcript": [],
        "brands": []
    }

    JOB_QUEUE.put((job_id, url, words))

    return {"job_id": job_id}

@app.get("/api/status/{job_id}")
def status(job_id):

    if job_id not in ACTIVE_JOBS:
        raise HTTPException(404)

    return ACTIVE_JOBS[job_id]

@app.get("/api/transcript/{job_id}")
def transcript(job_id):

    if job_id not in ACTIVE_JOBS:
        raise HTTPException(404)

    return ACTIVE_JOBS[job_id]["transcript"]

@app.get("/api/brands/{job_id}")
def brands(job_id):

    if job_id not in ACTIVE_JOBS:
        raise HTTPException(404)

    return ACTIVE_JOBS[job_id]["brands"]

@app.get("/api/health")
def health():
    return {"status":"ok"}
