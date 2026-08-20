# AI Bettor — production image.
#
# Two stages so the compiler toolchain psycopg2/numpy may need does not ship in
# the final image. Dependencies install from requirements.txt (exact pins) and
# the app runs as a non-root user.

FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libpq-dev \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# Everything lands in /install so the runtime stage can copy one directory.
RUN pip install --prefix=/install -r requirements.txt


FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOST=0.0.0.0 \
    PORT=8000

# libpq5 is psycopg2's runtime library; curl backs the healthcheck below.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libpq5 curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 bettor

WORKDIR /app

COPY --from=builder /install /usr/local

COPY pyproject.toml requirements.txt ./
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY alembic.ini ./
COPY alembic/ ./alembic/

# The SQLite dev fallback (if ever enabled) writes into the working directory.
RUN chown -R bettor:bettor /app

USER bettor

EXPOSE 8000

# /health is public by design, so the check works with API_TOKEN set. It reports
# "degraded" when an API key is missing, which is still a live process — only a
# non-200 (no database, app dead) marks the container unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health > /dev/null || exit 1

# One worker: the automation scheduler is in-process, and a second worker would
# run a second scheduler against the same database.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
