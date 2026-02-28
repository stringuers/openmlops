"""Monitoring pipeline steps with Evidently drift detection.
Each step is decorated with DockerSettings so ZenML's local_docker step
operator runs it in its own dedicated container image.
"""
import os
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Annotated
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import mlflow
from zenml import step
from zenml.config import DockerSettings
from zenml.logger import get_logger

logger = get_logger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MONITORING_DIR = Path("/app/monitoring")
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.15"))

# ─── Per-step Docker image settings ─────────────────────────────────────────

_docker_collect = DockerSettings(
    parent_image="openmlops-step-collect:latest",
    skip_build=True,
    replicate_local_python_environment=None,
)
_docker_evidently = DockerSettings(
    parent_image="openmlops-step-evidently:latest",
    skip_build=True,
    replicate_local_python_environment=None,
)
_docker_trigger = DockerSettings(
    parent_image="openmlops-step-trigger:latest",
    skip_build=True,
    replicate_local_python_environment=None,
)
_docker_store = DockerSettings(
    parent_image="openmlops-step-store:latest",
    skip_build=True,
    replicate_local_python_environment=None,
)


def _get_feature_stats(loader, model, device, num_batches=20):
    """Extract softmax probability features for drift detection."""
    import torch.nn.functional as F
    all_probs = []
    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(loader):
            if i >= num_batches:
                break
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.vstack(all_probs)


# ─────────────────────────── Step 1: collect_inference_data ─────────────────

@step(settings={"docker": _docker_collect})
def collect_inference_data() -> Annotated[Dict[str, Any], "inference_info"]:
    """Simulate collecting inference data logs (use test set + noise as drift signal)."""
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_set = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=transform)
    ref_subset = torch.utils.data.Subset(test_set, range(0, 2000))
    ref_loader = torch.utils.data.DataLoader(ref_subset, batch_size=256, shuffle=False)

    drift_mode = os.getenv("INJECT_DRIFT", "false").lower() == "true"

    if drift_mode:
        noisy_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.5),
        ])
        cur_set = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=noisy_transform)
    else:
        cur_set = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=transform)

    cur_subset = torch.utils.data.Subset(cur_set, range(2000, 4000))
    cur_loader = torch.utils.data.DataLoader(cur_subset, batch_size=256, shuffle=False)

    device = torch.device("cpu")
    model = _load_production_model(device)

    if model is not None:
        ref_probs = _get_feature_stats(ref_loader, model, device)
        cur_probs = _get_feature_stats(cur_loader, model, device)
    else:
        logger.warning("No production model found, generating synthetic probabilities")
        ref_probs = np.random.dirichlet(np.ones(10), size=500)
        if drift_mode:
            cur_probs = np.random.dirichlet(np.ones(10) * 0.3, size=500)
        else:
            cur_probs = np.random.dirichlet(np.ones(10), size=500)

    class_names = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    ref_df = pd.DataFrame(ref_probs, columns=[f"prob_{c}" for c in class_names])
    ref_df["predicted_class"] = ref_probs.argmax(axis=1)
    ref_df["max_confidence"] = ref_probs.max(axis=1)

    cur_df = pd.DataFrame(cur_probs, columns=[f"prob_{c}" for c in class_names])
    cur_df["predicted_class"] = cur_probs.argmax(axis=1)
    cur_df["max_confidence"] = cur_probs.max(axis=1)

    ref_path = str(MONITORING_DIR / "reference_data.parquet")
    cur_path = str(MONITORING_DIR / "current_data.parquet")
    ref_df.to_parquet(ref_path, index=False)
    cur_df.to_parquet(cur_path, index=False)

    logger.info(f"Inference data collected: {len(ref_df)} reference, {len(cur_df)} current samples")

    return {
        "reference_path": ref_path,
        "current_path": cur_path,
        "drift_injected": drift_mode,
        "timestamp": datetime.now().isoformat(),
        "n_reference": len(ref_df),
        "n_current": len(cur_df),
    }


def _load_production_model(device):
    """Load the best available model from checkpoint."""
    from src.models.cnn_model import get_model

    checkpoint_path = Path("/app/artifacts/model_checkpoint.pth")
    if checkpoint_path.exists():
        try:
            model = get_model(10, 0.5).to(device)
            model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
            model.eval()
            return model
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
    return None


# ─────────────────────────── Step 2: run_evidently_report ───────────────────

@step(settings={"docker": _docker_evidently})
def run_evidently_report(inference_info: Dict[str, Any]) -> Annotated[Dict[str, Any], "drift_result"]:
    """Run Evidently drift detection and generate HTML report."""
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset

    ref_df = pd.read_parquet(inference_info["reference_path"])
    cur_df = pd.read_parquet(inference_info["current_path"])

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    report.run(reference_data=ref_df, current_data=cur_df)

    report_path = str(MONITORING_DIR / f"evidently_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    report.save_html(report_path)

    report_dict = report.as_dict()
    drift_metrics = {}

    try:
        drift_result_data = report_dict["metrics"][0]["result"]
        drift_metrics = {
            "drift_share": drift_result_data.get("drift_share", 0.0),
            "n_drifted_features": drift_result_data.get("number_of_drifted_columns", 0),
            "n_total_features": drift_result_data.get("number_of_columns", 0),
            "dataset_drift_detected": drift_result_data.get("dataset_drift", False),
        }
    except (KeyError, IndexError) as e:
        logger.warning(f"Could not extract drift metrics: {e}")
        drift_metrics = {"drift_share": 0.0, "dataset_drift_detected": False}

    logger.info(f"Evidently report generated: {report_path}")
    logger.info(f"Drift metrics: {drift_metrics}")

    return {
        **inference_info,
        "report_path": report_path,
        **drift_metrics,
    }


# ─────────────────────────── Step 3: trigger_decision ───────────────────────

@step(settings={"docker": _docker_trigger})
def trigger_decision(drift_result: Dict[str, Any]) -> Annotated[Dict[str, Any], "decision"]:
    """Decide whether to retrain based on drift detection."""
    drift_share = drift_result.get("drift_share", 0.0)
    dataset_drift = drift_result.get("dataset_drift_detected", False)
    drift_injected = drift_result.get("drift_injected", False)

    should_retrain = dataset_drift or (drift_share >= DRIFT_THRESHOLD)

    decision = {
        **drift_result,
        "should_retrain": should_retrain,
        "drift_share": drift_share,
        "drift_threshold": DRIFT_THRESHOLD,
        "reason": (
            f"Dataset drift detected (share={drift_share:.3f} >= threshold={DRIFT_THRESHOLD})"
            if should_retrain
            else f"No significant drift (share={drift_share:.3f} < threshold={DRIFT_THRESHOLD})"
        ),
    }

    if should_retrain:
        logger.warning(f"⚠️  RETRAIN TRIGGERED: {decision['reason']}")
    else:
        logger.info(f"✅ No retrain needed: {decision['reason']}")

    if should_retrain and os.getenv("AUTO_RETRAIN", "false").lower() == "true":
        logger.info("AUTO_RETRAIN=true → launching training pipeline...")
        subprocess.Popen(["python", "/app/src/pipelines/run_training.py"])

    return decision


# ─────────────────────────── Step 4: store_monitoring_artifacts ─────────────

@step(settings={"docker": _docker_store})
def store_monitoring_artifacts(decision: Dict[str, Any]) -> Annotated[str, "artifact_summary"]:
    """Save monitoring artifacts to MLflow and disk."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("cifar10-monitoring")

    summary = {
        "timestamp": decision["timestamp"],
        "drift_share": decision.get("drift_share", 0.0),
        "dataset_drift_detected": decision.get("dataset_drift_detected", False),
        "should_retrain": decision["should_retrain"],
        "reason": decision["reason"],
        "n_reference": decision["n_reference"],
        "n_current": decision["n_current"],
        "drift_threshold": decision["drift_threshold"],
    }

    summary_path = str(MONITORING_DIR / "monitoring_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    with mlflow.start_run(run_name=f"monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_metrics({
            "drift_share": decision.get("drift_share", 0.0),
            "n_drifted_features": decision.get("n_drifted_features", 0),
            "should_retrain": int(decision["should_retrain"]),
        })
        mlflow.log_param("drift_threshold", DRIFT_THRESHOLD)
        mlflow.log_artifact(summary_path, "monitoring")

        report_path = decision.get("report_path", "")
        if report_path and Path(report_path).exists():
            mlflow.log_artifact(report_path, "monitoring")

    logger.info(f"Monitoring artifacts stored. Summary: {summary}")
    return json.dumps(summary)
