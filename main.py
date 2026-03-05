
import os, re, json, uuid, time, queue, threading, subprocess
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from faster_whisper import WhisperModel
from yt_dlp import YoutubeDL
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

SEGMENT_SECONDS = 20
WORKERS = 4

app = FastAPI(title="Amar Stream AI Platform")
templates = Jinja2Templates(directory=str(ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

jobs: Dict[str, dict] = {}

model = WhisperModel("base", device="cpu", compute_type="int8")
vad_model = load_silero_vad()

def seconds_to_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h}h{m:02d}m{s:02d}s"

def twitch_ts_link(url: str, seconds: float) -> str:
    ts = seconds_to_hms(seconds)
    joiner = "&" if "?" in url else "?"
    return f"{url}{joiner}t={ts}"

def _vod_direct_audio(url: str):
    ydl_opts = {"quiet": True,"skip_download": True,"format": "bestaudio/best"}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info.get("url"), info.get("duration")

def _run_vod_segmenter(vod_url: str, out_tmpl: str):
    direct, dur = _vod_direct_audio(vod_url)
    ffmpeg_cmd = [
        "ffmpeg","-hide_banner","-loglevel","error",
        "-i",direct,
        "-vn","-ac","1","-ar","16000",
        "-f","segment",
        "-segment_time",str(SEGMENT_SECONDS),
        "-reset_timestamps","1",
        out_tmpl,
    ]
    p = subprocess.Popen(ffmpeg_cmd)
    return p,dur

def clean_text(text:str):
    t=text.strip()
    if len(t.split()) < 3:
        return ""
    if t.lower() in ["a","uh","um","ah","hm","of"]:
        return ""
    return t

def speech_filter(audio_path):
    wav = torch.from_numpy(
        __import__("soundfile").read(str(audio_path))[0]
    )
    speech = get_speech_timestamps(wav, vad_model, sampling_rate=16000)
    return len(speech) > 0

def transcribe_file(path:Path, offset:float):
    if not speech_filter(path):
        return []
    segments,_ = model.transcribe(str(path), beam_size=5)
    out=[]
    for s in segments:
        text = clean_text(s.text)
        if not text:
            continue
        out.append({
            "start": float(s.start)+offset,
            "end": float(s.end)+offset,
            "text": text
        })
    return out

def _apply_segment(job, seg):
    job["segments"].append(seg)
    job["sse_q"].put({"type":"segment","segment":seg})

def _segment_loop(job_id,url):
    job=jobs[job_id]
    chunks_dir=job["dir"]/ "chunks"
    chunks_dir.mkdir(parents=True,exist_ok=True)
    out=str(chunks_dir/"chunk_%08d.wav")

    p,duration=_run_vod_segmenter(url,out)
    job["duration"]=duration

    q=job["work_q"]

    def feeder():
        last=-1
        while True:
            wavs=sorted(chunks_dir.glob("chunk_*.wav"))
            for w in wavs:
                idx=int(w.stem.split("_")[1])
                if idx<=last: continue
                q.put((idx,w))
                last=idx
            if p.poll() is not None:
                break
            time.sleep(0.2)
        q.put((None,None))

    def worker():
        while True:
            idx,wav=q.get()
            if idx is None:
                q.put((None,None))
                return
            offset=idx*SEGMENT_SECONDS
            segs=transcribe_file(wav,offset)
            for s in segs:
                _apply_segment(job,s)

    threading.Thread(target=feeder,daemon=True).start()

    for _ in range(WORKERS):
        threading.Thread(target=worker,daemon=True).start()

def _new_job(url):
    job_id=uuid.uuid4().hex[:10]
    job_dir=DATA/job_id
    job_dir.mkdir(parents=True,exist_ok=True)

    jobs[job_id]={
        "id":job_id,
        "url":url,
        "dir":job_dir,
        "segments":[],
        "sse_q":queue.Queue(),
        "work_q":queue.Queue(),
        "duration":None
    }

    threading.Thread(target=_segment_loop,args=(job_id,url),daemon=True).start()
    return job_id

@app.get("/",response_class=HTMLResponse)
def home(request:Request):
    return templates.TemplateResponse("index.html",{"request":request})

@app.post("/api/start")
async def api_start(payload:dict,response:Response):
    url=(payload.get("url") or "").strip()
    if not url:
        raise HTTPException(400,"Missing url")
    job_id=_new_job(url)
    response.set_cookie("job_id",job_id)
    return {"job_id":job_id}

@app.get("/api/events/{job_id}")
async def api_events(job_id:str):
    job=jobs.get(job_id)
    if not job:
        raise HTTPException(404,"job not found")

    def gen():
        q=job["sse_q"]
        while True:
            item=q.get()
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(gen(),media_type="text/event-stream")
