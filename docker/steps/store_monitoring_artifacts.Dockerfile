# ── Step: store_monitoring_artifacts ───────────────────────────────────────
# Persists Evidently HTML report + monitoring summary JSON to MLflow.
# Needs: mlflow (in base)
FROM openmlops-base:latest

LABEL org.opencontainers.image.title="openmlops-step-store"
LABEL org.opencontainers.image.description="ZenML step: store_monitoring_artifacts — log Evidently report + summary to MLflow"

CMD ["python", "-c", "from src.steps.monitoring_steps import store_monitoring_artifacts; store_monitoring_artifacts()"]
