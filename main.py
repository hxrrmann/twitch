import os, re, json, uuid, time, queue, threading, subprocess, shutil
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import whisper
from yt_dlp import YoutubeDL

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

SEGMENT_SECONDS = int(os.environ.get("SEGMENT_SECONDS", "15"))
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "tiny")
VOD_LANGUAGE = os.environ.get("VOD_LANGUAGE", "")

WORKERS = int(os.environ.get("WORKERS", "1"))
MAX_IN_MEMORY_SEGMENTS = int(os.environ.get("MAX_IN_MEMORY_SEGMENTS", "4000"))

app = FastAPI(title="Amar Stream AI Platform")
templates = Jinja2Templates(directory=str(ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

jobs: Dict[str, dict] = {}


def seconds_to_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h}h{m:02d}m{s:02d}s"


def fmt_hhmmss(seconds: float) -> str:
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def twitch_ts_link(url: str, seconds: float) -> str:
    ts = seconds_to_hms(seconds)
    joiner = "&" if "?" in url else "?"
    return f"{url}{joiner}t={ts}"


def _safe_word_regex(q: str) -> re.Pattern:
    return re.compile(rf"\b{re.escape(q)}\b", re.IGNORECASE)


def job_emit(job_id: str, item: dict):
    jobs[job_id]["sse_q"].put(item)


def _run_live_segmenter(url: str, out_tmpl: str) -> subprocess.Popen:
    streamlink_cmd = ["streamlink", url, "best", "-O"]
    ffmpeg_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "segment",
        "-segment_time", str(SEGMENT_SECONDS),
        "-reset_timestamps", "1",
        out_tmpl,
    ]
    p1 = subprocess.Popen(streamlink_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p2 = subprocess.Popen(ffmpeg_cmd, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p1, p2


def _vod_direct_audio(url: str) -> tuple[str, Optional[float]]:
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "format": "bestaudio/best",
        "noplaylist": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        direct = info.get("url")
        if not direct:
            raise RuntimeError("Konnte keine direkte Audio URL finden")
        dur = info.get("duration")
        try:
            dur_f = float(dur) if dur is not None else None
        except Exception:
            dur_f = None
        return direct, dur_f


def _run_vod_segmenter(vod_url: str, out_tmpl: str) -> tuple[subprocess.Popen, Optional[float]]:
    direct, dur = _vod_direct_audio(vod_url)
    ffmpeg_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", direct,
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "segment",
        "-segment_time", str(SEGMENT_SECONDS),
        "-reset_timestamps", "1",
        out_tmpl,
    ]
    p = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p, dur


def transcribe_file(model, path: Path, offset_seconds: float = 0.0, language: str = "") -> List[dict]:
    kwargs = {"verbose": False}
    if language:
        kwargs["language"] = language
    result = model.transcribe(str(path), **kwargs)
    segs = []
    for s in result.get("segments", []):
        segs.append({
            "start": float(s["start"]) + offset_seconds,
            "end": float(s["end"]) + offset_seconds,
            "text": (s["text"] or "").strip(),
        })
    return segs


HIDDEN_CONTEXT = {
    "crypto": ["krypto", "börse", "trading", "wallet", "bitcoin", "btc", "eth", "ethereum", "altcoin"],
    "hardware": ["maus", "keyboard", "tastatur", "headset", "logitech", "razer", "steelseries", "gpu", "grafikkarte"],
    "nutrition": ["shake", "protein", "creatin", "kreatin", "pulver", "makros", "kalorien"],
}


def hidden_context_hits(text: str) -> List[str]:
    t = text.lower()
    out = []
    for k, words in HIDDEN_CONTEXT.items():
        if any(w in t for w in words):
            out.append(k)
    return out[:3]


def _append_transcript(job: dict, seg: dict):
    p = job["transcript_file"]
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(seg, ensure_ascii=False) + "\n")


def _iter_transcript(job: dict):
    p = job.get("transcript_file")
    if not p or not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def compute_sponsor_report(job: dict):
    brands = sorted(list(job["watch"]))
    report = {}
    if not brands:
        job["sponsor_report"] = {}
        return
    patterns = {b: _safe_word_regex(b) for b in brands}
    temp_hits = {b: [] for b in brands}
    talk = {b: 0.0 for b in brands}

    for s in _iter_transcript(job):
        for b, pat in patterns.items():
            if pat.search(s.get("text", "")):
                dur = max(0.0, float(s.get("end", 0.0) - s.get("start", 0.0)))
                talk[b] += min(dur, 20.0)
                temp_hits[b].append({
                    "start": s.get("start", 0.0),
                    "time": seconds_to_hms(s.get("start", 0.0)),
                    "text": s.get("text", ""),
                    "link": twitch_ts_link(job["url"], s.get("start", 0.0)) if job["mode"] == "vod" else None,
                })

    for b in brands:
        hits = temp_hits[b]
        if hits:
            report[b] = {
                "mentions": len(hits),
                "total_talktime_sec": round(talk[b], 1),
                "hits": hits,
            }
    job["sponsor_report"] = report


def compute_hidden_context(job: dict):
    items = []
    last = {}
    for s in _iter_transcript(job):
        tags = hidden_context_hits(s.get("text", ""))
        if not tags:
            continue
        for tag in tags:
            prev = last.get(tag)
            st = float(s.get("start", 0.0))
            if prev is not None and abs(st - prev) < 25:
                continue
            last[tag] = st
            items.append({
                "context": tag,
                "start": st,
                "time": seconds_to_hms(st),
                "text": s.get("text", ""),
                "link": twitch_ts_link(job["url"], st) if job["mode"] == "vod" else None,
            })
        if len(items) >= 200:
            break
    job["hidden_sponsor"] = items


def stream_summary(job: dict):
    summary = {
        "mode": job["mode"],
        "segments_in_memory": len(job["segments"]),
        "watch": sorted(list(job["watch"])),
        "sponsor_brands": list(job.get("sponsor_report", {}).keys()),
    }
    job["summary"] = summary


def _apply_segment(job: dict, seg: dict):
    job["segments"].append(seg)
    if len(job["segments"]) > MAX_IN_MEMORY_SEGMENTS:
        job["segments"] = job["segments"][-MAX_IN_MEMORY_SEGMENTS:]

    _append_transcript(job, seg)

    for w in list(job["watch"]):
        if _safe_word_regex(w).search(seg.get("text", "")):
            job_emit(job["id"], {
                "type": "alert",
                "word": w,
                "time": seconds_to_hms(seg.get("start", 0.0)),
                "start": seg.get("start", 0.0),
                "text": seg.get("text", ""),
                "link": twitch_ts_link(job["url"], seg.get("start", 0.0)) if job["mode"] == "vod" else None,
            })


def _safe_pct(processed_sec: float, duration_sec: Optional[float]) -> Optional[float]:
    if not duration_sec or duration_sec <= 0:
        return None
    return max(0.0, min(100.0, (processed_sec / duration_sec) * 100.0))


def _segment_loop(job_id: str, mode: str, url: str):
    job = jobs[job_id]
    chunks_dir = job["dir"] / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(chunks_dir / "chunk_%08d.wav")

    duration = None
    procs = []

    try:
        if mode == "live":
            p1, p2 = _run_live_segmenter(url, out_tmpl)
            procs = [p1, p2]
        else:
            p, duration = _run_vod_segmenter(url, out_tmpl)
            procs = [p]
    except Exception as e:
        job["status"] = "error"
        job_emit(job_id, {"type": "error", "message": f"Segmenter error: {e}"})
        return

    job["procs"] = procs
    job["duration"] = duration

    job["status"] = "transcribing"
    job_emit(job_id, {"type": "status", "status": job["status"], "mode": job["mode"], "pct": 0})

    model = whisper.load_model(WHISPER_MODEL)

    q: queue.Queue = job["work_q"]
    stop_flag = job["stop_event"]

    last_idx = -1

    def feeder():
        nonlocal last_idx
        while not stop_flag.is_set():
            wavs = sorted(chunks_dir.glob("chunk_*.wav"))
            for w in wavs:
                m = re.search(r"chunk_(\d+)\.wav$", w.name)
                if not m:
                    continue
                idx = int(m.group(1))
                if idx <= last_idx:
                    continue

                s1 = w.stat().st_size
                time.sleep(0.25)
                if not w.exists():
                    continue
                if w.stat().st_size != s1:
                    continue

                q.put((idx, w))
                last_idx = idx

            if mode == "vod":
                p = procs[0]
                if p.poll() is not None:
                    break
            else:
                p2 = procs[1]
                if p2.poll() is not None:
                    break

            time.sleep(0.15)

        q.put((None, None))

    model_lock = threading.Lock()

    def worker(worker_id: int):
        while True:
            idx, wav_path = q.get()
            if idx is None:
                q.put((None, None))
                return
            try:
                offset = float(idx) * float(SEGMENT_SECONDS)
                with model_lock:
                    segs = transcribe_file(model, wav_path, offset_seconds=offset, language=VOD_LANGUAGE if mode == "vod" else "")
                for s in segs:
                    _apply_segment(job, s)
                    job_emit(job_id, {"type": "segment", "segment": s})

                processed = (float(idx) + 1.0) * float(SEGMENT_SECONDS)
                pct = _safe_pct(processed, duration)
                job_emit(job_id, {"type": "status", "status": job["status"], "mode": job["mode"], "pct": pct})

                try:
                    wav_path.unlink(missing_ok=True)
                except Exception:
                    pass
            except Exception as e:
                job_emit(job_id, {"type": "error", "message": f"Transcribe error: {e}"})

    feeder_t = threading.Thread(target=feeder, daemon=True)
    feeder_t.start()

    worker_threads = []
    for i in range(max(1, WORKERS)):
        t = threading.Thread(target=worker, args=(i,), daemon=True)
        t.start()
        worker_threads.append(t)

    feeder_t.join()

    while any(t.is_alive() for t in worker_threads):
        time.sleep(0.2)
        if stop_flag.is_set():
            break

    compute_sponsor_report(job)
    compute_hidden_context(job)
    stream_summary(job)
    job_emit(job_id, {"type": "sponsors", "report": job.get("sponsor_report", {})})
    job_emit(job_id, {"type": "hidden_sponsor", "items": job.get("hidden_sponsor", [])})
    job_emit(job_id, {"type": "summary", "summary": job.get("summary", {})})

    job["status"] = "stopped" if stop_flag.is_set() else "done"
    job_emit(job_id, {"type": "status", "status": job["status"], "mode": job["mode"], "pct": 100 if mode == "vod" and duration else None})

    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass


def _new_job(url: str, mode: str, watch: List[str]) -> str:
    job_id = uuid.uuid4().hex[:10]
    job_dir = DATA / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    transcript_file = job_dir / "transcript.jsonl"
    if transcript_file.exists():
        transcript_file.unlink()

    jobs[job_id] = {
        "id": job_id,
        "url": url,
        "mode": mode,
        "status": "queued",
        "dir": job_dir,
        "segments": [],
        "watch": set([w.strip() for w in watch if isinstance(w, str) and w.strip()]),
        "sse_q": queue.Queue(),
        "stop_event": threading.Event(),
        "procs": [],
        "work_q": queue.Queue(maxsize=256),
        "transcript_file": transcript_file,
        "duration": None,
        "sponsor_report": {},
        "hidden_sponsor": [],
        "summary": {},
    }

    t = threading.Thread(target=_segment_loop, args=(job_id, mode, url), daemon=True)
    jobs[job_id]["thread"] = t
    t.start()
    return job_id


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/start")
async def api_start(payload: dict, response: Response):
    url = (payload.get("url") or "").strip()
    mode = (payload.get("mode") or "vod").strip().lower()
    watch = payload.get("watch") or []
    if not url:
        raise HTTPException(400, "Missing url")
    if mode not in ("vod", "live"):
        raise HTTPException(400, "mode must be 'vod' or 'live'")

    job_id = _new_job(url, mode, watch)
    response.set_cookie("job_id", job_id, httponly=False, samesite="lax")
    return {"job_id": job_id}


@app.post("/api/stop/{job_id}")
async def api_stop(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    job["stop_event"].set()
    return {"ok": True}


@app.post("/api/watch/{job_id}")
async def api_watch(job_id: str, payload: dict):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    words = payload.get("words") or []
    if not isinstance(words, list):
        raise HTTPException(400, "words must be list")
    job["watch"] = set([w.strip() for w in words if isinstance(w, str) and w.strip()])
    compute_sponsor_report(job)
    job_emit(job_id, {"type": "sponsors", "report": job.get("sponsor_report", {})})
    stream_summary(job)
    job_emit(job_id, {"type": "summary", "summary": job.get("summary", {})})
    return {"ok": True}


@app.get("/api/search/{job_id}")
async def api_search(job_id: str, q: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    q = (q or "").strip()
    if not q:
        return {"results": []}

    pat = _safe_word_regex(q)
    out = []
    for s in _iter_transcript(job):
        if pat.search(s.get("text", "")):
            start = float(s.get("start", 0.0))
            out.append({
                "start": start,
                "time": seconds_to_hms(start),
                "text": s.get("text", ""),
                "link": twitch_ts_link(job["url"], start) if job["mode"] == "vod" else None,
            })
            if len(out) >= 800:
                break

    return {"results": out}


@app.get("/api/events/{job_id}")
async def api_events(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")

    def gen():
        q = job["sse_q"]
        yield f"data: {json.dumps({'type':'status','status':job['status'],'mode':job['mode'],'pct': None})}\n\n"
        while True:
            try:
                item = q.get(timeout=15)
                yield f"data: {json.dumps(item)}\n\n"
            except queue.Empty:
                yield "data: {\"type\":\"keepalive\"}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def _cookie_job_id(request: Request) -> str:
    return request.cookies.get("job_id") or ""


@app.post("/start")
async def start_compat(payload: dict, response: Response):
    return await api_start(payload, response)


@app.get("/search")
async def search_compat(request: Request, q: str = ""):
    job_id = _cookie_job_id(request)
    if not job_id or job_id not in jobs:
        raise HTTPException(404, "job not found")
    return await api_search(job_id, q=q)


@app.post("/alerts")
async def alerts_compat(request: Request, payload: dict):
    job_id = _cookie_job_id(request)
    if not job_id or job_id not in jobs:
        raise HTTPException(404, "job not found")
    words = payload.get("words")
    if words is None:
        raw = payload.get("alerts") or payload.get("watch") or ""
        words = [w.strip() for w in str(raw).split(",") if w.strip()]
    return await api_watch(job_id, {"words": words})
