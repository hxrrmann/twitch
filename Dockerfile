
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
ffmpeg \
python3 \
python3-pip

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

ENV PORT=8080

CMD uvicorn Main:app --host 0.0.0.0 --port $PORT
