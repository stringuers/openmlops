# ── Step: run_evidently_report ─────────────────────────────────────────────
# Runs Evidently DataDrift + DataQuality report on reference vs current data.
# Extra deps: evidently, scikit-learn
FROM openmlops-base:latest

LABEL org.opencontainers.image.title="openmlops-step-evidently"
LABEL org.opencontainers.image.description="ZenML step: run_evidently_report — drift detection HTML report"

COPY requirements/monitoring.txt /tmp/requirements_monitoring.txt
RUN pip install --no-cache-dir -r /tmp/requirements_monitoring.txt

CMD ["python", "-c", "from src.steps.monitoring_steps import run_evidently_report; run_evidently_report()"]
