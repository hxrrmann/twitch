
import os
import uuid
import time
import json
import queue
import threading
import subprocess
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel

app = FastAPI(title="Amar Stream AI Ultra Fixed")

# serve static ui
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# root route so browser does not return Not Found
@app.get("/")
def root():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"status": "server running"}

TMP_DIR = "/tmp"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

CPU_COUNT = os.cpu_count() or 8
BATCH_SIZE = 6
CHUNK_SECONDS = 40

model = WhisperModel(
    "distil-large-v3",
    device="cuda",
    compute_type="float16"
)

jobs: Dict[str, dict] = {}

def save_job(job_id):
    with open(f"{DATA_DIR}/{job_id}.json","w") as f:
        json.dump(jobs[job_id],f)

def load_jobs():
    if not os.path.exists(DATA_DIR):
        return
    for file in os.listdir(DATA_DIR):
        if file.endswith(".json"):
            with open(f"{DATA_DIR}/{file}") as f:
                data=json.load(f)
                jobs[data["id"]] = data

def eta_format(sec):
    if sec <= 0:
        return "0s"
    m=int(sec//60)
    s=int(sec%60)
    return f"{m}m {s}s"

def stream_download_and_split(url, job_id, chunk_queue):

    chunk_dir=f"{TMP_DIR}/{job_id}_chunks"
    os.makedirs(chunk_dir,exist_ok=True)

    cmd=[
        "yt-dlp",
        "-o","-",
        url
    ]

    ffmpeg=[
        "ffmpeg",
        "-loglevel","quiet",
        "-i","pipe:0",
        "-f","segment",
        "-segment_time",str(CHUNK_SECONDS),
        "-c","copy",
        f"{chunk_dir}/chunk_%03d.wav"
    ]

    p1=subprocess.Popen(cmd,stdout=subprocess.PIPE)
    p2=subprocess.Popen(ffmpeg,stdin=p1.stdout)
    p1.stdout.close()

    known=set()

    while True:

        files=[
            os.path.join(chunk_dir,f)
            for f in os.listdir(chunk_dir)
            if f.endswith(".wav")
        ]

        for f in sorted(files):
            if f not in known:
                known.add(f)
                chunk_queue.put(f)

        if p2.poll() is not None:
            break

        time.sleep(1)

def batch_transcribe(paths):

    results=[]

    for p in paths:
        segments,_=model.transcribe(
            p,
            beam_size=1,
            best_of=1,
            vad_filter=True
        )

        txt=""
        for s in segments:
            txt+=s.text+" "

        results.append(txt.strip())

    return results

def process_vod(job_id,url,alerts):

    job=jobs[job_id]

    chunk_queue=queue.Queue()

    dl_thread=threading.Thread(
        target=stream_download_and_split,
        args=(url,job_id,chunk_queue)
    )

    dl_thread.start()

    transcript=[]
    brands=[]
    processed=0
    batch=[]

    start=time.time()

    while True:

        try:
            chunk=chunk_queue.get(timeout=5)
        except:
            if not dl_thread.is_alive():
                break
            continue

        batch.append(chunk)

        if len(batch)>=BATCH_SIZE:

            texts=batch_transcribe(batch)

            for t in texts:

                transcript.append(t)

                for w in alerts:
                    if w.lower() in t.lower():
                        brands.append({
                            "word":w,
                            "text":t
                        })

            processed+=len(batch)

            elapsed=time.time()-start
            speed=processed/elapsed if elapsed>0 else 0

            job["progress"]=processed
            job["percent"]=min(99, processed*2)
            job["eta"]=eta_format(60/speed if speed>0 else 0)
            job["step"]="transcribing"
            job["transcript"]=transcript
            job["brands"]=brands

            save_job(job_id)

            batch=[]

    job["percent"]=100
    job["status"]="finished"
    job["step"]="completed"
    job["transcript"]=transcript

    save_job(job_id)

@app.post("/api/start")
def start(payload:dict):

    url=payload.get("url")
    alerts=payload.get("alerts","")

    if not url:
        raise HTTPException(400,"URL missing")

    alert_words=[x.strip() for x in alerts.split(",") if x.strip()]

    job_id=str(uuid.uuid4())

    jobs[job_id]={
        "id":job_id,
        "status":"running",
        "percent":0,
        "progress":0,
        "eta":"",
        "step":"starting",
        "transcript":[],
        "brands":[]
    }

    save_job(job_id)

    t=threading.Thread(
        target=process_vod,
        args=(job_id,url,alert_words)
    )

    t.start()

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

@app.get("/api/health")
def health():
    return {"status":"ok"}

load_jobs()

if __name__=="__main__":

    import uvicorn

    port=int(os.environ.get("PORT",8080))

    uvicorn.run(
        "Main:app",
        host="0.0.0.0",
        port=port
    )
