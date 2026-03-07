
import os, re, json, uuid, time, queue, threading, subprocess, shutil, hashlib
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import whisper
from yt_dlp import YoutubeDL
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

SEGMENT_SECONDS = int(os.environ.get("SEGMENT_SECONDS", "20"))
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
VOD_LANGUAGE = os.environ.get("VOD_LANGUAGE", "de")
ETA_SPEED = float(os.environ.get("ETA_SPEED", "10"))

TWITCH_CLIENT_ID = os.environ.get("TWITCH_CLIENT_ID", "")
TWITCH_OAUTH_TOKEN = os.environ.get("TWITCH_OAUTH_TOKEN", "")

app = FastAPI(title="Amar Stream AI Platform")
templates = Jinja2Templates(directory=str(ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

jobs: Dict[str, dict] = {}
state_lock = threading.Lock()


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


def _job_key(url: str) -> str:
    return hashlib.sha1(url.strip().encode("utf-8")).hexdigest()[:16]


def _state_path(job_dir: Path) -> Path:
    return job_dir / "state.json"


def _transcript_path(job_dir: Path) -> Path:
    return job_dir / "transcript.jsonl"


def _save_state(job: dict):
    safe = {
        "id": job["id"],
        "key": job["key"],
        "url": job["url"],
        "mode": job["mode"],
        "status": job["status"],
        "watch": sorted(list(job.get("watch", set()))),
        "pct": job.get("pct", 0),
        "eta": job.get("eta"),
        "last_status": job.get("last_status", ""),
        "last_index": job.get("last_index", -1),
        "duration_sec": job.get("duration_sec"),
        "summary": job.get("summary", {}),
        "sponsor_report": job.get("sponsor_report", {}),
        "hidden_sponsor": job.get("hidden_sponsor", []),
        "stats": job.get("stats", {"top_words": [], "heat": {}, "emotion_heat": {}}),
        "created_at": job.get("created_at", time.time()),
        "updated_at": time.time(),
    }
    _state_path(job["dir"]).write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_segments(job_dir: Path) -> List[dict]:
    p = _transcript_path(job_dir)
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _append_segment(job: dict, seg: dict):
    with _transcript_path(job["dir"]).open("a", encoding="utf-8") as f:
        f.write(json.dumps(seg, ensure_ascii=False) + "\n")


def _load_or_create_job(url: str) -> dict:
    key = _job_key(url)
    job_dir = DATA / key
    job_dir.mkdir(parents=True, exist_ok=True)
    state_p = _state_path(job_dir)
    if state_p.exists():
        data = json.loads(state_p.read_text(encoding="utf-8"))
        job = {
            "id": data["id"],
            "key": key,
            "url": data["url"],
            "mode": "vod",
            "status": data.get("status", "idle"),
            "dir": job_dir,
            "segments": _load_segments(job_dir),
            "watch": set(data.get("watch", [])),
            "sse_q": queue.Queue(),
            "stop": False,
            "thread": None,
            "pct": data.get("pct", 0),
            "eta": data.get("eta"),
            "last_status": data.get("last_status", "Bereit"),
            "last_index": data.get("last_index", -1),
            "duration_sec": data.get("duration_sec"),
            "stats": data.get("stats", {"top_words": [], "heat": {}, "emotion_heat": {}}),
            "summary": data.get("summary", {}),
            "sponsor_report": data.get("sponsor_report", {}),
            "hidden_sponsor": data.get("hidden_sponsor", []),
            "created_at": data.get("created_at", time.time()),
        }
        jobs[job["id"]] = job
        return job
    job_id = uuid.uuid4().hex[:10]
    job = {
        "id": job_id,
        "key": key,
        "url": url.strip(),
        "mode": "vod",
        "status": "idle",
        "dir": job_dir,
        "segments": [],
        "watch": set(),
        "sse_q": queue.Queue(),
        "stop": False,
        "thread": None,
        "pct": 0,
        "eta": None,
        "last_status": "Bereit",
        "last_index": -1,
        "duration_sec": None,
        "stats": {"top_words": [], "heat": {}, "emotion_heat": {}},
        "summary": {},
        "sponsor_report": {},
        "hidden_sponsor": [],
        "created_at": time.time(),
    }
    jobs[job_id] = job
    _save_state(job)
    return job


def job_emit(job_id: str, item: dict):
    job = jobs.get(job_id)
    if job:
        job["sse_q"].put(item)


def set_progress(job: dict, status: str, pct: Optional[float] = None, eta: Optional[float] = None):
    job["last_status"] = status
    if pct is not None:
        job["pct"] = max(0, min(100, round(float(pct), 1)))
    if eta is not None:
        job["eta"] = None if eta is None else max(0, int(eta))
    _save_state(job)
    job_emit(job["id"], {"type": "status", "status": status, "pct": job.get("pct", 0), "eta": job.get("eta")})


def download_vod_audio(url: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    uid = uuid.uuid4().hex
    tmpl = str(out_dir / f"{uid}_%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": tmpl,
        "quiet": True,
        "noplaylist": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
        duration = info.get("duration")
        try:
            duration = float(duration) if duration is not None else None
        except Exception:
            duration = None
        return path, duration


def transcribe_file(model, path: Path, language: str = "") -> List[dict]:
    kwargs = {"verbose": False, "word_timestamps": False, "condition_on_previous_text": False}
    if language:
        kwargs["language"] = language
    result = model.transcribe(str(path), **kwargs)
    segs = []
    for s in result.get("segments", []):
        text = (s.get("text") or "").strip()
        if not text:
            continue
        segs.append({
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": text,
        })
    return segs


def clean_text(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    t = re.sub(r"(^|\s)([a-zA-ZäöüÄÖÜß])(?=\s|$)", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) < 3:
        return ""
    if not re.search(r"[a-zA-ZäöüÄÖÜß]", t):
        return ""
    if t and t[-1] not in ".!?":
        t += "."
    return t


def merge_segments(segs: List[dict]) -> List[dict]:
    if not segs:
        return []
    out = []
    cur = {"start": segs[0]["start"], "end": segs[0]["end"], "text": segs[0]["text"]}
    for s in segs[1:]:
        gap = float(s["start"] - cur["end"])
        cur_text = cur["text"]
        nxt_text = s["text"]
        if gap <= 1.4 and len(cur_text) < 220:
            cur["end"] = s["end"]
            cur["text"] = (cur_text.rstrip(".?!") + " " + nxt_text).strip()
        else:
            cur["text"] = clean_text(cur["text"])
            if cur["text"]:
                out.append(cur)
            cur = {"start": s["start"], "end": s["end"], "text": s["text"]}
    cur["text"] = clean_text(cur["text"])
    if cur["text"]:
        out.append(cur)
    return out


def toxicity_score(text: str) -> float:
    bad = ["idiot", "trash", "scheiße", "noob", "stfu"]
    t = text.lower()
    return min(1.0, sum(1 for x in bad if x in t) / 2.0)


def emotion_label(text: str) -> str:
    t = text.lower()
    if any(x in t for x in ["lets go", "let's go", "gg", "insane", "krank"]) or "!" in text:
        return "Hype"
    if any(x in t for x in ["fuck", "scheiße", "bro", "unfair"]):
        return "Rage"
    if any(x in t for x in ["haha", "lol", "lmao"]):
        return "Funny"
    return "Neutral"


def rebuild_word_stats(job: dict):
    freq: Dict[str, int] = {}
    heat: Dict[int, int] = {}
    emo_heat = {}
    for s in job["segments"]:
        minute = int(float(s["start"]) // 60)
        heat[minute] = heat.get(minute, 0) + 1
        emo = emotion_label(s["text"])
        emo_heat.setdefault(minute, {"Hype": 0, "Rage": 0, "Funny": 0, "Neutral": 0})
        emo_heat[minute][emo] += 1
        for w in re.findall(r"[\w']{3,}", s["text"].lower()):
            if w.isdigit():
                continue
            freq[w] = freq.get(w, 0) + 1
    job["stats"] = {"top_words": sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:50], "heat": heat, "emotion_heat": emo_heat}


def compute_sponsor_report(job: dict):
    brands = sorted(list(job["watch"]))
    report = {}
    cta_terms = ["code", "rabatt", "link", "beschreibung", "partner", "sponsor"]
    for b in brands:
        pat = _safe_word_regex(b)
        hits = []
        talktime = 0.0
        for s in job["segments"]:
            txt = s["text"]
            lower = txt.lower()
            if pat.search(txt) or any(term in lower for term in cta_terms if b.lower() in lower or any(term in lower for term in cta_terms)):
                dur = max(0.0, float(s["end"] - s["start"]))
                talktime += min(dur, 20.0)
                hits.append({
                    "start": s["start"],
                    "time": seconds_to_hms(s["start"]),
                    "text": txt,
                    "link": twitch_ts_link(job["url"], s["start"]),
                })
        if hits:
            report[b] = {"mentions": len(hits), "total_talktime_sec": round(talktime, 1), "hits": hits}
    job["sponsor_report"] = report


def compute_hidden_sponsor(job: dict):
    contexts = {
        "crypto": ["krypto", "börse", "trading", "wallet", "bitcoin", "btc", "eth", "ethereum", "altcoin"],
        "nutrition": ["protein", "creatin", "kreatin", "shake", "kalorien", "supplement"],
        "hardware": ["maus", "keyboard", "tastatur", "headset", "logitech", "razer", "steelseries", "gpu", "grafikkarte"],
        "banking": ["karte", "broker", "bank", "konto", "zahlung"],
    }
    found = []
    for s in job["segments"]:
        t = s["text"].lower()
        for k, words in contexts.items():
            if any(w in t for w in words):
                found.append({
                    "context": k,
                    "start": s["start"],
                    "time": seconds_to_hms(s["start"]),
                    "text": s["text"],
                    "link": twitch_ts_link(job["url"], s["start"]),
                })
    out = []
    for it in sorted(found, key=lambda x: x["start"]):
        if out and it["context"] == out[-1]["context"] and abs(it["start"] - out[-1]["start"]) < 30:
            continue
        out.append(it)
    job["hidden_sponsor"] = out[:200]


def stream_summary(job: dict):
    top = job.get("stats", {}).get("top_words", [])[:12]
    summary = {
        "segments": len(job["segments"]),
        "top_words": [{"word": w, "count": c} for w, c in top],
        "sponsor_brands": list(job.get("sponsor_report", {}).keys()),
        "current_status": job.get("last_status", ""),
    }
    job["summary"] = summary


def _apply_segment(job: dict, seg: dict):
    job["segments"].append(seg)
    _append_segment(job, seg)
    for w in list(job["watch"]):
        if _safe_word_regex(w).search(seg["text"]):
            job_emit(job["id"], {"type": "alert", "word": w, "time": seconds_to_hms(seg["start"]), "start": seg["start"], "text": seg["text"], "link": twitch_ts_link(job["url"], seg["start"])})


def run_vod_job(job_id: str):
    job = jobs[job_id]
    try:
        job["stop"] = False
        job["status"] = "downloading"
        set_progress(job, "Downloading VOD", 2, None)
        audio_path, duration = download_vod_audio(job["url"], job["dir"] / "media")
        if duration:
            job["duration_sec"] = duration
            est = int(max(20, duration / ETA_SPEED))
        else:
            est = None
        set_progress(job, "Preparing audio", 10, est)
        model = whisper.load_model(WHISPER_MODEL)
        set_progress(job, "Transcribing", 15, est)
        raw = transcribe_file(model, audio_path, language=VOD_LANGUAGE)
        merged = merge_segments(raw)
        total = max(1, len(merged))
        start_index = int(job.get("last_index", -1)) + 1
        if start_index < 0:
            start_index = 0
        for idx, seg in enumerate(merged):
            if idx < start_index:
                continue
            if job.get("stop"):
                job["status"] = "paused"
                set_progress(job, "Paused", job.get("pct", 0), job.get("eta"))
                return
            _apply_segment(job, seg)
            job_emit(job_id, {"type": "segment", "segment": seg})
            job["last_index"] = idx
            pct = 15 + (idx + 1) / total * 70
            remaining = total - (idx + 1)
            eta = int(remaining * 2) if remaining > 0 else 0
            set_progress(job, "Transcribing", pct, eta)
        set_progress(job, "Analysing", 90, 10)
        rebuild_word_stats(job)
        compute_sponsor_report(job)
        compute_hidden_sponsor(job)
        stream_summary(job)
        job_emit(job_id, {"type": "stats", "stats": job["stats"]})
        job_emit(job_id, {"type": "sponsors", "report": job["sponsor_report"]})
        job_emit(job_id, {"type": "hidden_sponsor", "items": job["hidden_sponsor"]})
        job_emit(job_id, {"type": "summary", "summary": job["summary"]})
        job["status"] = "done"
        set_progress(job, "Finished", 100, 0)
    except Exception as e:
        job["status"] = "error"
        _save_state(job)
        job_emit(job_id, {"type": "error", "message": f"{e}"})


def start_or_resume_job(job: dict):
    if job.get("thread") and job["thread"].is_alive():
        return
    t = threading.Thread(target=run_vod_job, args=(job["id"],), daemon=True)
    job["thread"] = t
    t.start()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/start")
async def api_start(payload: dict, response: Response):
    url = (payload.get("url") or "").strip()
    watch = payload.get("watch") or []
    if not url:
        raise HTTPException(400, "Missing url")
    job = _load_or_create_job(url)
    if isinstance(watch, list):
        job["watch"] = set([w.strip() for w in watch if isinstance(w, str) and w.strip()])
    start_or_resume_job(job)
    response.set_cookie("job_id", job["id"], httponly=False, samesite="lax")
    _save_state(job)
    return {"job_id": job["id"]}


@app.post("/api/stop/{job_id}")
async def api_stop(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    job["stop"] = True
    job["status"] = "pausing"
    _save_state(job)
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
    _save_state(job)
    job_emit(job_id, {"type": "sponsors", "report": job["sponsor_report"]})
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
    for s in job["segments"]:
        if pat.search(s["text"]):
            out.append({"start": s["start"], "time": seconds_to_hms(s["start"]), "text": s["text"], "link": twitch_ts_link(job["url"], s["start"])})
    return {"results": out[:800]}


@app.get("/api/status/{job_id}")
async def api_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return {"status": job.get("last_status"), "pct": job.get("pct"), "eta": job.get("eta")}


@app.get("/api/events/{job_id}")
async def api_events(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    def gen():
        yield f"data: {json.dumps({'type':'status','status':job.get('last_status'),'pct':job.get('pct',0),'eta':job.get('eta')})}\n\n"
        # replay state on reconnect
        if job.get("stats"):
            yield f"data: {json.dumps({'type':'stats','stats':job['stats']})}\n\n"
        if job.get("sponsor_report"):
            yield f"data: {json.dumps({'type':'sponsors','report':job['sponsor_report']})}\n\n"
        if job.get("hidden_sponsor"):
            yield f"data: {json.dumps({'type':'hidden_sponsor','items':job['hidden_sponsor']})}\n\n"
        if job.get("summary"):
            yield f"data: {json.dumps({'type':'summary','summary':job['summary']})}\n\n"
        q = job["sse_q"]
        while True:
            try:
                item = q.get(timeout=15)
                yield f"data: {json.dumps(item)}\n\n"
            except queue.Empty:
                yield "data: {"type":"keepalive"}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/download/{job_id}/{filename}")
async def download(job_id: str, filename: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    p = job["dir"] / "exports" / filename
    if not p.exists():
        raise HTTPException(404, "file not found")
    return FileResponse(str(p), filename=filename)


@app.post("/start")
async def start_compat(payload: dict, response: Response):
    return await api_start(payload, response)


@app.get("/search")
async def search_compat(request: Request, q: str = ""):
    job_id = request.cookies.get("job_id") or ""
    if not job_id or job_id not in jobs:
        raise HTTPException(404, "job not found")
    return await api_search(job_id, q=q)


@app.post("/alerts")
async def alerts_compat(request: Request, payload: dict):
    job_id = request.cookies.get("job_id") or ""
    if not job_id or job_id not in jobs:
        raise HTTPException(404, "job not found")
    words = payload.get("words")
    if words is None:
        raw = payload.get("alerts") or payload.get("watch") or ""
        words = [w.strip() for w in str(raw).split(",") if w.strip()]
    return await api_watch(job_id, {"words": words})
