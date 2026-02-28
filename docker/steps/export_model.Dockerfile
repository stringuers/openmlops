# ── Step: export_model ─────────────────────────────────────────────────────
# Exports trained model to TorchScript + ONNX serving-ready formats.
# Extra deps: onnx, onnxruntime
FROM openmlops-base:latest

LABEL org.opencontainers.image.title="openmlops-step-export"
LABEL org.opencontainers.image.description="ZenML step: export_model — TorchScript + ONNX export"

COPY requirements/export.txt /tmp/requirements_export.txt
RUN pip install --no-cache-dir -r /tmp/requirements_export.txt

CMD ["python", "-c", "from src.steps.training_steps import export_model; export_model()"]
