# ── Step: collect_inference_data ───────────────────────────────────────────
# Loads test set and model, extracts softmax probabilities as DataFrames.
# Needs: torch, torchvision, pandas, numpy (all in base)
FROM openmlops-base:latest

LABEL org.opencontainers.image.title="openmlops-step-collect"
LABEL org.opencontainers.image.description="ZenML step: collect_inference_data — extract model predictions for drift analysis"

CMD ["python", "-c", "from src.steps.monitoring_steps import collect_inference_data; collect_inference_data()"]
