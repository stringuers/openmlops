# ── Step: split_data ───────────────────────────────────────────────────────
# Splits CIFAR-10 into 80% train / 20% val and returns metadata.
# Needs: torch, torchvision (all in base)
FROM openmlops-base:latest

LABEL org.opencontainers.image.title="openmlops-step-split"
LABEL org.opencontainers.image.description="ZenML step: split_data — 80/20 train-val split"

CMD ["python", "-c", "from src.steps.training_steps import split_data; split_data()"]
