# ── Step: ingest_data ──────────────────────────────────────────────────────
# Pulls CIFAR-10 via DVC or downloads directly from torchvision.
# Needs: dvc[s3], torchvision (all in base)
FROM openmlops-base:latest

LABEL org.opencontainers.image.title="openmlops-step-ingest"
LABEL org.opencontainers.image.description="ZenML step: ingest_data — pull CIFAR-10 via DVC"

CMD ["python", "-c", "from src.steps.training_steps import ingest_data; ingest_data()"]
