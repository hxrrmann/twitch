import os
import uuid
import time
import json
import queue
import asyncio
import threading
import subprocess
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

app=FastAPI(title="Amar Stream AI")

if os.path.exists("static"):
    app.mount("/static",StaticFiles(directory="static"),name="static")

@app.get("/")
def home():
    if os.path.exists("templates/index.html"):
        return FileResponse("templates/index.html")
    return FileResponse("index.html")

model=None

JOB_QUEUE=queue.Queue()
JOBS:Dict[str,dict]={}
STREAMS:Dict[str,asyncio.Queue]={}

CHUNK_SECONDS=30
BATCH_SIZE=4

def twitch_ts(sec):
    h=int(sec//3600)
    m=int((sec%3600)//60)
    s=int(sec%60)
    return f"{h}h{m}m{s}s"

def eta_format(sec):
    if sec<=0: return "0s"
    m=int(sec//60)
    s=int(sec%60)
    return f"{m}m {s}s"

def llm_summary(text):

    provider=os.environ.get("LLM_PROVIDER","none")

    if provider=="openai":
        try:
            from openai import OpenAI
            client=OpenAI()
            r=client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":"summarize this stream: "+text[:6000]}]
            )
            return r.choices[0].message.content
        except:
            pass

    if provider=="anthropic":
        try:
            import anthropic
            client=anthropic.Anthropic()
            r=client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=300,
                messages=[{"role":"user","content":"summarize this stream: "+text[:6000]}]
            )
            return r.content[0].text
        except:
            pass

    words=text.split()[:120]
    return " ".join(words)

def download_stream(url,job_id,chunk_queue):

    chunk_dir=f"/tmp/{job_id}"
    os.makedirs(chunk_dir,exist_ok=True)

    ytdlp=["yt-dlp","-o","-",url]

    ffmpeg=[
        "ffmpeg",
        "-loglevel","quiet",
        "-i","pipe:0",
        "-f","segment",
        "-segment_time",str(CHUNK_SECONDS),
        "-c","copy",
        f"{chunk_dir}/chunk_%03d.wav"
    ]

    p1=subprocess.Popen(ytdlp,stdout=subprocess.PIPE)
    p2=subprocess.Popen(ffmpeg,stdin=p1.stdout)
    p1.stdout.close()

    seen=set()

    while True:

        files=[
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

    global model

    if model is None:
        model=WhisperModel(
            "distil-large-v3",
            device="cuda",
            compute_type="float16"
        )

    segs=[]

    for p in paths:

        segments,_=model.transcribe(
            p,
            beam_size=1,
            best_of=1,
            vad_filter=True
        )

        for s in segments:
            segs.append({"text":s.text.strip(),"start":s.start,"end":s.end})

    return segs

async def push(job_id,data):
    if job_id in STREAMS:
        await STREAMS[job_id].put(data)

def worker():

    while True:

        job_id,url,alerts=JOB_QUEUE.get()
        job=JOBS[job_id]

        chunk_queue=queue.Queue()

        dl=threading.Thread(
            target=download_stream,
            args=(url,job_id,chunk_queue)
        )
        dl.start()

        transcript=[]
        talktime={}
        processed=0
        batch=[]

        start=time.time()

        while True:

            try:
                chunk=chunk_queue.get(timeout=5)
            except:
                if not dl.is_alive():
                    break
                continue

            batch.append(chunk)

            if len(batch)>=BATCH_SIZE:

                segs=transcribe_batch(batch)

                for s in segs:

                    transcript.append(s["text"])

                    ts=twitch_ts(s["start"])

                    asyncio.run(push(job_id,{
                        "type":"segment",
                        "text":s["text"],
                        "timestamp":ts
                    }))

                    for w in alerts:
                        if w.lower() in s["text"].lower():

                            dur=s["end"]-s["start"]

                            talktime[w]=talktime.get(w,0)+dur

                processed+=len(batch)

                elapsed=time.time()-start
                speed=processed/elapsed if elapsed>0 else 0

                job["percent"]=min(99,processed*2)
                job["eta"]=eta_format((60/speed) if speed>0 else 0)
                job["step"]="transcribing"

                batch=[]

        job["percent"]=100
        job["step"]="completed"
        job["status"]="finished"

        job["sponsor_talktime"]=talktime

        summary=llm_summary(" ".join(transcript))

        asyncio.run(push(job_id,{
            "type":"summary",
            "text":summary
        }))

threading.Thread(target=worker,daemon=True).start()

@app.post("/api/start")
def start(payload:dict):

    url=payload.get("url")
    alerts=payload.get("alerts","")

    if not url:
        raise HTTPException(400,"url missing")

    words=[x.strip() for x in alerts.split(",") if x.strip()]

    job_id=str(uuid.uuid4())

    JOBS[job_id]={
        "id":job_id,
        "percent":0,
        "eta":"",
        "step":"queued",
        "status":"running",
        "sponsor_talktime":{}
    }

    STREAMS[job_id]=asyncio.Queue()

    JOB_QUEUE.put((job_id,url,words))

    return {"job_id":job_id}

@app.get("/api/events/{job_id}")
async def events(job_id):

    if job_id not in STREAMS:
        raise HTTPException(404)

    async def gen():
        q=STREAMS[job_id]
        while True:
            data=await q.get()
            yield f"data: {json.dumps(data)}\n\n"

    return StreamingResponse(gen(),media_type="text/event-stream")

@app.get("/api/status/{job_id}")
def status(job_id):

    if job_id not in JOBS:
        raise HTTPException(404)

    return JOBS[job_id]

@app.get("/api/health")
def health():
    return {"status":"ok"}
