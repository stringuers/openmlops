# ── Step: trigger_decision ─────────────────────────────────────────────────
# Evaluates drift_share vs threshold and sets should_retrain flag.
# Needs: base only (no extra deps)
FROM openmlops-base:latest

LABEL org.opencontainers.image.title="openmlops-step-trigger"
LABEL org.opencontainers.image.description="ZenML step: trigger_decision — retrain trigger based on drift threshold"

CMD ["python", "-c", "from src.steps.monitoring_steps import trigger_decision; trigger_decision()"]
