
import os
import time
import uuid
import json
import threading
import subprocess
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

app = FastAPI(title="Amar Stream AI Platform GPU")

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

CPU_COUNT = os.cpu_count() or 8
WORKERS = CPU_COUNT
MODEL_COUNT = min(4, CPU_COUNT)
CHUNK_SECONDS = 45

TMP_DIR = "/tmp"
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)

model_pool = Queue()

for _ in range(MODEL_COUNT):
    model_pool.put(
        WhisperModel(
            "distil-large-v3",
            device="cuda",
            compute_type="float16"
        )
    )

jobs: Dict[str, dict] = {}

def eta_format(seconds):
    if seconds <= 0:
        return "0s"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s"

def save_job(job_id):
    with open(f"{DATA_DIR}/{job_id}.json", "w") as f:
        json.dump(jobs[job_id], f)

def load_jobs():
    if not os.path.exists(DATA_DIR):
        return
    for file in os.listdir(DATA_DIR):
        if file.endswith(".json"):
            with open(f"{DATA_DIR}/{file}") as f:
                data = json.load(f)
                jobs[data["id"]] = data

def download_audio(url, output):
    subprocess.run([
        "yt-dlp",
        "-N","8",
        "-x",
        "--audio-format","wav",
        "-o",output,
        url
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def split_audio(input_file, job_id):
    chunk_dir = f"{TMP_DIR}/{job_id}_chunks"
    os.makedirs(chunk_dir, exist_ok=True)

    subprocess.run([
        "ffmpeg",
        "-threads","8",
        "-i",input_file,
        "-f","segment",
        "-segment_time",str(CHUNK_SECONDS),
        "-c","copy",
        f"{chunk_dir}/chunk_%03d.wav"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return sorted([
        os.path.join(chunk_dir,f)
        for f in os.listdir(chunk_dir)
        if f.endswith(".wav")
    ])

def language_correction(text: str) -> str:
    text = text.strip()
    if len(text) == 0:
        return text
    text = text[0].upper() + text[1:]
    if not text.endswith("."):
        text += "."
    return text

def reconstruct_sentences(transcript: List[str]) -> List[str]:
    merged = []
    buffer = ""
    for t in transcript:
        if len(buffer) < 120:
            buffer += " " + t
        else:
            merged.append(buffer.strip())
            buffer = t
    if buffer:
        merged.append(buffer.strip())
    return merged

def transcribe_chunk(path):
    model = model_pool.get()
    try:
        segments, _ = model.transcribe(
            path,
            beam_size=1,
            best_of=1,
            vad_filter=True
        )
        text = ""
        for s in segments:
            text += s.text + " "
        return text.strip()
    finally:
        model_pool.put(model)

def process_vod(job_id, url, alerts):
    job = jobs[job_id]
    audio_file = f"{TMP_DIR}/{job_id}.wav"

    job["step"] = "Downloading"
    download_audio(url, audio_file)

    job["step"] = "Splitting"
    chunks = split_audio(audio_file, job_id)

    total = len(chunks)
    transcript = []
    brands = []

    start = time.time()
    job["step"] = "Transcribing"

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        results = list(executor.map(transcribe_chunk, chunks))

    for i, text in enumerate(results):
        if job["status"] == "stopped":
            return

        text = language_correction(text)
        transcript.append(text)

        for word in alerts:
            if word.lower() in text.lower():
                brands.append({
                    "word":word,
                    "segment":i,
                    "text":text
                })

        progress = int((i / total) * 100)
        elapsed = time.time() - start
        speed = i / elapsed if elapsed > 0 else 0
        remaining = (total - i) / speed if speed > 0 else 0

        job["progress"] = progress
        job["eta"] = eta_format(remaining)
        job["transcript"] = transcript
        job["brands"] = brands

        save_job(job_id)

    transcript = reconstruct_sentences(transcript)

    job["transcript"] = transcript
    job["progress"] = 100
    job["eta"] = "0s"
    job["status"] = "finished"
    job["step"] = "Completed"

    save_job(job_id)

@app.post("/api/start")
def start(payload: dict):
    url = payload.get("url")
    alerts = payload.get("alerts","")

    if not url:
        raise HTTPException(400,"URL missing")

    alert_words = [x.strip() for x in alerts.split(",") if x.strip()]
    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "id":job_id,
        "status":"running",
        "progress":0,
        "eta":"",
        "step":"Starting",
        "transcript":[],
        "brands":[]
    }

    save_job(job_id)

    thread = threading.Thread(
        target=process_vod,
        args=(job_id,url,alert_words)
    )

    thread.start()

    return {"job_id":job_id}

@app.get("/api/status/{job_id}")
def status(job_id):
    if job_id not in jobs:
        raise HTTPException(404)
    return jobs[job_id]

@app.get("/api/transcript/{job_id}")
def transcript(job_id):
    if job_id not in jobs:
        raise HTTPException(404)
    return jobs[job_id]["transcript"]

@app.get("/api/brands/{job_id}")
def brands(job_id):
    if job_id not in jobs:
        raise HTTPException(404)
    return jobs[job_id]["brands"]

@app.post("/api/stop")
def stop(payload: dict):
    job_id = payload.get("job_id")
    if job_id not in jobs:
        raise HTTPException(404)
    jobs[job_id]["status"] = "stopped"
    save_job(job_id)
    return {"status":"stopped"}

@app.get("/api/health")
def health():
    return {"status":"ok"}

load_jobs()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT",8080))
    uvicorn.run("Main:app", host="0.0.0.0", port=port)
