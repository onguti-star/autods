FROM python:3.11-slim

WORKDIR /app

# build-essential: in case any dependency needs to build from source on your
# VPS's CPU architecture (most wheels below have prebuilt binaries for both
# x86_64 and arm64/aarch64, so this is usually just a safety net).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend ./backend
# Serving the frontend from the backend too is harmless and keeps
# `python backend/run.py` working exactly like local dev, even though the
# Netlify+VPS split deployment serves the frontend from Netlify instead.
COPY frontend ./frontend

# Session data (uploaded datasets, trained models) is pickled to disk here --
# see backend/store.py's _SESSION_DIR. Mount a volume at this path (see
# docker-compose.yml) so it survives container restarts/rebuilds.
RUN mkdir -p /app/.sessions

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["python", "backend/run.py"]
