# ── Step: register_model ───────────────────────────────────────────────────
# Registers model in MLflow Model Registry & promotes to Staging.
# Needs: mlflow, torch (all in base)
FROM openmlops-base:latest

LABEL org.opencontainers.image.title="openmlops-step-register"
LABEL org.opencontainers.image.description="ZenML step: register_model — MLflow Model Registry + Staging promotion"

CMD ["python", "-c", "from src.steps.training_steps import register_model; register_model()"]
