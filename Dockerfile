# icore — container image
# Multi-stage: build deps in a builder, copy into a slim runtime.

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build deps for native extensions (asyncpg, oracledb, sasl, ...)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libsasl2-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.txt ./
COPY icore ./icore

# Install the package with default extras (core only; override EXTRAS for more)
ARG EXTRAS=""
RUN pip wheel --wheel-dir /wheels ".${EXTRAS}"

# ------------------------------------------------------------------
# Runtime image
# ------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ICORE_CONFIG_DIR=/app/config

WORKDIR /app

# Copy installed wheels from builder
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# Copy config (can be overridden by a volume mount)
COPY config ./config

EXPOSE 8000

# Run via the console script entry point
CMD ["icore", "--host", "0.0.0.0", "--port", "8000"]
