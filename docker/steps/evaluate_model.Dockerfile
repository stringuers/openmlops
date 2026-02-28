# ── Step: evaluate_model ───────────────────────────────────────────────────
# Confusion matrix, classification report, test accuracy; logs to MLflow.
# Extra deps: scikit-learn, matplotlib, seaborn
FROM openmlops-base:latest

LABEL org.opencontainers.image.title="openmlops-step-evaluate"
LABEL org.opencontainers.image.description="ZenML step: evaluate_model — confusion matrix + classification report"

COPY requirements/train.txt /tmp/requirements_train.txt
RUN pip install --no-cache-dir -r /tmp/requirements_train.txt

CMD ["python", "-c", "from src.steps.training_steps import evaluate_model; evaluate_model()"]
