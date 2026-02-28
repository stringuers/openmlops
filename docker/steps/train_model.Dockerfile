# ── Step: train_model ──────────────────────────────────────────────────────
# CNN training with Adam optimizer, CosineAnnealing, MLflow metric logging.
# Needs: torch, torchvision, mlflow (all in base)
FROM openmlops-base:latest

LABEL org.opencontainers.image.title="openmlops-step-train"
LABEL org.opencontainers.image.description="ZenML step: train_model — CNN training with MLflow tracking"

# No extra deps beyond base for training
CMD ["python", "-c", "from src.steps.training_steps import train_model; train_model()"]
