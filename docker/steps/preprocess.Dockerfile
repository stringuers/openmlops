# ── Step: preprocess ───────────────────────────────────────────────────────
# Computes and saves CIFAR-10 normalization statistics.
# Needs: torch (in base)
FROM openmlops-base:latest

LABEL org.opencontainers.image.title="openmlops-step-preprocess"
LABEL org.opencontainers.image.description="ZenML step: preprocess — normalization stats + augmentation config"

CMD ["python", "-c", "from src.steps.training_steps import preprocess; preprocess()"]
