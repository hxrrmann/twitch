
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

from faster_whisper import WhisperModel
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ===== CONFIG =====
# NOTE: Cloud Run/Render must start FAST. Do NOT load the Whisper model at import time.
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
CHUNK_SECONDS = int(os.environ.get("CHUNK_SECONDS", "20"))  # 15–30s range

_model = None
_model_lock = threading.Lock()


def get_model() -> WhisperModel:
    """Lazy-load the model so the web server can bind to $PORT immediately."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            # int8 on CPU is the best speed/RAM tradeoff on cheap instances.
            _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return _model

def extract_audio(video_path: str) -> str:
    """Extract 16kHz mono wav for Whisper."""
    audio_path = "audio.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            audio_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return audio_path

def split_audio(audio_path: str, segment_length: int = CHUNK_SECONDS) -> list[str]:
    """Split into ~15–30s chunks for parallelism + better VAD."""
    out_dir = "chunks"
    os.makedirs(out_dir, exist_ok=True)

    # Make filenames stable and sorted.
    for f in os.listdir(out_dir):
        try:
            os.remove(os.path.join(out_dir, f))
        except OSError:
            pass

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            audio_path,
            "-f",
            "segment",
            "-segment_time",
            str(segment_length),
            "-reset_timestamps",
            "1",
            "-c",
            "copy",
            f"{out_dir}/out%05d.wav",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    files = sorted(
        (os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".wav")),
        key=lambda p: p,
    )
    return files

def transcribe_chunk(path: str) -> str:
    """Transcribe using VAD filtering to reduce 'word fragments' and background noise."""
    model = get_model()

    segments, _info = model.transcribe(
        path,
        beam_size=1,
        best_of=1,
        temperature=0.0,
        vad_filter=True,
        vad_parameters={
            # stricter than defaults => fewer random syllables
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 150,
        },
        condition_on_previous_text=False,
        # If Whisper isn't confident it should prefer silence.
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
    )

    parts: list[str] = []
    for seg in segments:
        t = (seg.text or "").strip()
        if not t:
            continue
        parts.append(t)
    return " ".join(parts)

def merge_sentences(text: str) -> str:
    """Turn many small chunks into readable sentences.

    We keep it simple and robust: accumulate until punctuation OR length threshold.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    out: list[str] = []
    buf = ""
    for token in text.split(" "):
        if not token:
            continue
        if not buf:
            buf = token
        else:
            buf += " " + token

        if buf.endswith((".", "!", "?")) or len(buf) >= 120:
            out.append(buf.strip())
            buf = ""
    if buf:
        out.append(buf.strip())
    return "\n".join(out)

def detect_sponsors(text):
    patterns = ["code","rabatt","sponsor","partner","link","beschreibung"]
    found=[]
    lower=text.lower()
    for p in patterns:
        if p in lower:
            found.append(p)
    return found

def stream_summary(text):
    words=text.split()
    return " ".join(words[:60]) + "..."


@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload",methods=["POST"])
def upload():
    f=request.files["file"]
    video="input.mp4"
    f.save(video)

    audio = extract_audio(video)
    chunks = split_audio(audio)

    # Start the model lazily (first request). This keeps boot time low.
    get_model()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as exe:
        results = list(exe.map(transcribe_chunk, chunks))

    transcript = " ".join(r for r in results if r)
    transcript = merge_sentences(transcript)

    sponsors=detect_sponsors(transcript)
    summary=stream_summary(transcript)

    return jsonify({
        "transcript":transcript,
        "sponsors":sponsors,
        "summary":summary
    })

if __name__=="__main__":
    port = int(os.environ.get("PORT", "8080"))
    # Flask dev server is fine for quick tests; in production use gunicorn.
    app.run(host="0.0.0.0", port=port)
