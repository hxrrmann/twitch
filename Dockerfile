
FROM python:3.11-slim

WORKDIR /app

# ffmpeg is required for audio extraction + chunking
RUN apt-get update \
  && apt-get install -y --no-install-recommends ffmpeg \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

# Production server. Cloud Run provides $PORT.
CMD ["sh","-c","gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 8 --timeout 0 main:app"]
