# ─── Base image shared by all step containers ─────────────────────────────
# All heavy dependencies (torch, torchvision, zenml, dvc) live here.
# Per-step images extend this with only their lightweight additions.

FROM python:3.11-slim AS openmlops-base

LABEL org.opencontainers.image.title="openmlops-base"
LABEL org.opencontainers.image.description="Base image for OpenMLOps CIFAR-10 step containers"

WORKDIR /app

# ── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python base requirements ───────────────────────────────────────────────
COPY requirements/base.txt /tmp/requirements_base.txt
RUN pip install --no-cache-dir -r /tmp/requirements_base.txt

# ── Source code ────────────────────────────────────────────────────────────
COPY src/ /app/src/
COPY requirements/ /app/requirements/
COPY scripts/docker_entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

RUN mkdir -p /app/data /app/artifacts /app/monitoring

ENTRYPOINT ["/entrypoint.sh"]
