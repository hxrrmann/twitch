
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ===== CONFIG =====
MODEL_SIZE = "base"
NUM_WORKERS = 4

model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

def extract_audio(video_path):
    audio_path = "audio.wav"
    subprocess.run([
        "ffmpeg","-y","-i",video_path,"-ac","1","-ar","16000",audio_path
    ],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    return audio_path

def split_audio(audio_path, segment_length=60):
    out_dir = "chunks"
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run([
        "ffmpeg","-i",audio_path,"-f","segment","-segment_time",str(segment_length),
        "-c","copy",f"{out_dir}/out%03d.wav"
    ],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    return [os.path.join(out_dir,f) for f in os.listdir(out_dir)]

def transcribe_chunk(path):
    segments,_ = model.transcribe(path, beam_size=1)
    text = ""
    for seg in segments:
        text += seg.text + " "
    return text

def merge_sentences(text):
    sentences = text.replace("  "," ").split(".")
    merged = []
    buffer = ""
    for s in sentences:
        if len(s.strip()) < 5:
            buffer += " " + s
        else:
            merged.append((buffer + " " + s).strip())
            buffer=""
    return ". ".join(merged)

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

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload",methods=["POST"])
def upload():
    f=request.files["file"]
    video="input.mp4"
    f.save(video)

    audio=extract_audio(video)
    chunks=split_audio(audio)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as exe:
        results=list(exe.map(transcribe_chunk,chunks))

    transcript=" ".join(results)
    transcript=merge_sentences(transcript)

    sponsors=detect_sponsors(transcript)
    summary=stream_summary(transcript)

    return jsonify({
        "transcript":transcript,
        "sponsors":sponsors,
        "summary":summary
    })

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080)
