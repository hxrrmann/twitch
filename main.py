
import os
import time
import uuid
import threading
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from faster_whisper import WhisperModel
except:
    WhisperModel = None

app = FastAPI(title="Amar Stream AI Platform")

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

jobs: Dict[str, dict] = {}

model = None
if WhisperModel:
    try:
        model = WhisperModel("base", compute_type="int8")
    except:
        model = None

def eta_format(seconds):
    if seconds <= 0:
        return "0s"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s"

def transcribe_worker(job_id: str, url: str, alerts: List[str]):

    job = jobs[job_id]

    job["status"] = "downloading"
    job["step"] = "Downloading VOD"

    time.sleep(2)

    duration = 1800
    processed = 0

    transcript = []
    brands = []

    start = time.time()

    job["status"] = "processing"
    job["step"] = "Transcribing"

    while processed < duration:

        if job["status"] == "stopped":
            return

        time.sleep(0.4)

        processed += 6

        progress = int((processed / duration) * 100)

        elapsed = time.time() - start

        speed = processed / elapsed if elapsed > 0 else 0

        remaining = (duration - processed) / speed if speed > 0 else 0

        job["progress"] = min(progress, 100)
        job["eta"] = eta_format(remaining)

        text = f"Stream segment around {processed}s"

        segment = {
            "time": processed,
            "text": text
        }

        transcript.append(segment)

        for a in alerts:
            if a.lower() in text.lower():
                brands.append({
                    "word": a,
                    "time": processed
                })

        job["transcript"] = transcript
        job["brands"] = brands

    job["status"] = "finished"
    job["step"] = "Completed"
    job["progress"] = 100
    job["eta"] = "0s"


@app.post("/api/start")
def start_job(payload: dict):

    url = payload.get("url")
    alerts = payload.get("alerts", "")

    if not url:
        raise HTTPException(400, "No URL")

    alert_words = [x.strip() for x in alerts.split(",") if x.strip()]

    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "id": job_id,
        "url": url,
        "status": "starting",
        "progress": 0,
        "eta": "",
        "step": "Starting",
        "transcript": [],
        "brands": []
    }

    thread = threading.Thread(
        target=transcribe_worker,
        args=(job_id, url, alert_words)
    )

    thread.start()

    return {"job_id": job_id}


@app.post("/api/stop")
def stop_job(payload: dict):

    job_id = payload.get("job_id")

    if job_id not in jobs:
        raise HTTPException(404)

    jobs[job_id]["status"] = "stopped"

    return {"status": "stopped"}


@app.get("/api/status/{job_id}")
def status(job_id: str):

    if job_id not in jobs:
        raise HTTPException(404)

    job = jobs[job_id]

    return {
        "progress": job["progress"],
        "eta": job["eta"],
        "step": job["step"],
        "status": job["status"],
        "brands": job["brands"],
        "segments": len(job["transcript"])
    }


@app.get("/api/transcript/{job_id}")
def transcript(job_id: str):

    if job_id not in jobs:
        raise HTTPException(404)

    return jobs[job_id]["transcript"]


@app.get("/api/brands/{job_id}")
def brands(job_id: str):

    if job_id not in jobs:
        raise HTTPException(404)

    return jobs[job_id]["brands"]


@app.get("/api/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":

    import uvicorn

    port = int(os.environ.get("PORT", 8080))

    uvicorn.run(
        "Main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
