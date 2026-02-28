"""ZenML monitoring pipeline for drift detection."""
from zenml import pipeline
from zenml.logger import get_logger

from src.steps.monitoring_steps import (
    collect_inference_data,
    run_evidently_report,
    trigger_decision,
    store_monitoring_artifacts,
)

logger = get_logger(__name__)


@pipeline(name="cifar10_monitoring_pipeline", enable_cache=False)
def monitoring_pipeline():
    """Monitoring pipeline: collect → drift detection → decision → store artifacts."""
    inference_info = collect_inference_data()
    drift_result = run_evidently_report(inference_info)
    decision = trigger_decision(drift_result)
    artifact_summary = store_monitoring_artifacts(decision)
    return artifact_summary


if __name__ == "__main__":
    monitoring_pipeline()
