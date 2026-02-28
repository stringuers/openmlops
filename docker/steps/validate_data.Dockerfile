# ── Step: validate_data ────────────────────────────────────────────────────
# Validates CIFAR-10 shapes and sample counts.
# Needs: torchvision, torch (all in base)
FROM openmlops-base:latest

LABEL org.opencontainers.image.title="openmlops-step-validate"
LABEL org.opencontainers.image.description="ZenML step: validate_data — assert CIFAR-10 integrity"

CMD ["python", "-c", "from src.steps.training_steps import validate_data; validate_data()"]
