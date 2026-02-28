FROM python:3.11-slim

WORKDIR /app

# System deps
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

# Install DVC with S3 support first (needed for data ingestion)
RUN pip install --no-cache-dir "dvc[s3]==3.55.2"

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source code
COPY . .

# Python path & buffering
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create required directories
RUN mkdir -p /app/data /app/artifacts /app/monitoring

# Copy entrypoint
COPY scripts/docker_entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "src/pipelines/run_training.py"]
