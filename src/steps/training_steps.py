"""Training pipeline steps for CIFAR-10 CNN classifier.

Each step is decorated with DockerSettings so ZenML's local_docker step
operator runs it in its own dedicated container image.
"""
import os
import json
import subprocess
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Annotated

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms
import mlflow
import mlflow.pytorch
from zenml import step
from zenml.config import DockerSettings
from zenml.logger import get_logger

logger = get_logger(__name__)

CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# ─── Per-step Docker image settings ─────────────────────────────────────────
# Each step runs in its own container image built from docker/steps/<step>.Dockerfile
# Build all images first with: bash scripts/build_step_images.sh

_docker_ingest = DockerSettings(
    parent_image="openmlops-step-ingest:latest",
    skip_build=True,
    replicate_local_python_environment=None,
)
_docker_validate = DockerSettings(
    parent_image="openmlops-step-validate:latest",
    skip_build=True,
    replicate_local_python_environment=None,
)
_docker_split = DockerSettings(
    parent_image="openmlops-step-split:latest",
    skip_build=True,
    replicate_local_python_environment=None,
)
_docker_preprocess = DockerSettings(
    parent_image="openmlops-step-preprocess:latest",
    skip_build=True,
    replicate_local_python_environment=None,
)
_docker_train = DockerSettings(
    parent_image="openmlops-step-train:latest",
    skip_build=True,
    replicate_local_python_environment=None,
)
_docker_evaluate = DockerSettings(
    parent_image="openmlops-step-evaluate:latest",
    skip_build=True,
    replicate_local_python_environment=None,
)
_docker_register = DockerSettings(
    parent_image="openmlops-step-register:latest",
    skip_build=True,
    replicate_local_python_environment=None,
)
_docker_export = DockerSettings(
    parent_image="openmlops-step-export:latest",
    skip_build=True,
    replicate_local_python_environment=None,
)


# ─────────────────────────── Step 1: ingest_data ────────────────────────────

@step(settings={"docker": _docker_ingest})
def ingest_data() -> Annotated[str, "data_path"]:
    """Pull data via DVC or download CIFAR-10 directly."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cifar_path = DATA_DIR / "cifar-10-batches-py"

    if not cifar_path.exists():
        logger.info("Attempting DVC pull...")
        try:
            result = subprocess.run(
                ["dvc", "pull", "data/cifar-10-batches-py.dvc"],
                capture_output=True, text=True, cwd="/app"
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr)
            logger.info("DVC pull succeeded.")
        except Exception as e:
            logger.warning(f"DVC pull failed ({e}), downloading directly...")
            torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True)
            torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True)
    else:
        logger.info("Data already present, skipping download.")

    return str(DATA_DIR)


# ─────────────────────────── Step 2: validate_data ──────────────────────────

@step(settings={"docker": _docker_validate})
def validate_data(data_path: str) -> Annotated[bool, "is_valid"]:
    """Validate that dataset can be loaded and has expected shape."""
    try:
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=transform)

        assert len(train_set) == 50000, f"Expected 50000 train samples, got {len(train_set)}"
        assert len(test_set) == 10000, f"Expected 10000 test samples, got {len(test_set)}"

        sample_img, label = train_set[0]
        assert sample_img.shape == (3, 32, 32), f"Unexpected image shape: {sample_img.shape}"
        assert 0 <= label <= 9, f"Invalid label: {label}"

        logger.info(f"Data validation passed: {len(train_set)} train, {len(test_set)} test samples.")
        return True
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        raise RuntimeError(f"Data validation failed: {e}") from e


# ─────────────────────────── Step 3: split_data ─────────────────────────────

@step(settings={"docker": _docker_split})
def split_data(data_path: str) -> Annotated[Dict[str, Any], "split_info"]:
    """Split dataset into train/val/test."""
    transform = transforms.Compose([transforms.ToTensor()])
    full_train = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False, transform=transform)

    train_size = int(0.8 * len(full_train))
    val_size = len(full_train) - train_size
    train_set, val_set = random_split(full_train, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(42))

    split_info = {
        "data_path": data_path,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": 10000,
        "num_classes": 10,
    }
    logger.info(f"Split: {train_size} train / {val_size} val / 10000 test")
    return split_info


# ─────────────────────────── Step 4: preprocess ─────────────────────────────

@step(settings={"docker": _docker_preprocess})
def preprocess(split_info: Dict[str, Any]) -> Annotated[Dict[str, Any], "preprocessed_info"]:
    """Define preprocessing transforms and save normalization stats."""
    # CIFAR-10 normalization stats (pre-computed)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    preprocessed_info = {
        **split_info,
        "mean": list(mean),
        "std": list(std),
        "augmentation": True,
    }

    stats_path = DATA_DIR / "normalization_stats.json"
    with open(stats_path, "w") as f:
        json.dump({"mean": list(mean), "std": list(std)}, f)

    logger.info(f"Preprocessing configured. Stats saved to {stats_path}")
    return preprocessed_info


# ─────────────────────────── Step 5: train_model ────────────────────────────

@step(settings={"docker": _docker_train})
def train_model(preprocessed_info: Dict[str, Any]) -> Annotated[Dict[str, Any], "training_result"]:
    """Train CNN on CIFAR-10 with MLflow tracking."""
    from src.models.cnn_model import get_model

    data_path = preprocessed_info["data_path"]
    mean = tuple(preprocessed_info["mean"])
    std = tuple(preprocessed_info["std"])

    # Hyperparameters
    params = {
        "epochs": int(os.getenv("TRAIN_EPOCHS", "20")),
        "batch_size": int(os.getenv("BATCH_SIZE", "128")),
        "learning_rate": float(os.getenv("LEARNING_RATE", "0.001")),
        "dropout_rate": float(os.getenv("DROPOUT_RATE", "0.5")),
        "weight_decay": 1e-4,
        "num_classes": 10,
    }

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_train = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False, transform=train_transform)
    train_size = int(0.8 * len(full_train))
    val_size = len(full_train) - train_size
    train_set, val_set = random_split(full_train, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=params["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=params["batch_size"], shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    model = get_model(params["num_classes"], params["dropout_rate"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["epochs"])

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("cifar10-cnn")

    best_val_acc = 0.0
    model_path = Path("/app/artifacts/model_checkpoint.pth")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run() as run:
        mlflow.log_params(params)

        for epoch in range(params["epochs"]):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                train_correct += predicted.eq(labels).sum().item()
                train_total += inputs.size(0)

            scheduler.step()

            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(labels).sum().item()
                    val_total += inputs.size(0)

            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            epoch_train_loss = train_loss / train_total
            epoch_val_loss = val_loss / val_total

            mlflow.log_metrics({
                "train_loss": epoch_train_loss,
                "train_accuracy": train_acc,
                "val_loss": epoch_val_loss,
                "val_accuracy": val_acc,
                "learning_rate": scheduler.get_last_lr()[0],
            }, step=epoch)

            logger.info(f"Epoch {epoch+1}/{params['epochs']} | "
                        f"Train Loss: {epoch_train_loss:.4f} Acc: {train_acc:.4f} | "
                        f"Val Loss: {epoch_val_loss:.4f} Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), str(model_path))
                logger.info(f"  → New best model saved (val_acc={val_acc:.4f})")

        run_id = run.info.run_id

    logger.info(f"Training complete. Best val_acc: {best_val_acc:.4f}")

    return {
        **preprocessed_info,
        "run_id": run_id,
        "best_val_acc": best_val_acc,
        "model_path": str(model_path),
        "params": params,
    }


# ─────────────────────────── Step 6: evaluate_model ─────────────────────────

@step(settings={"docker": _docker_evaluate})
def evaluate_model(training_result: Dict[str, Any]) -> Annotated[Dict[str, Any], "eval_result"]:
    """Evaluate model on test set and log artifacts."""
    from src.models.cnn_model import get_model
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns

    data_path = training_result["data_path"]
    mean = tuple(training_result["mean"])
    std = tuple(training_result["std"])
    model_path = training_result["model_path"]
    run_id = training_result["run_id"]

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_set = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(training_result["params"]["num_classes"],
                      training_result["params"]["dropout_rate"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    test_loss, test_correct, test_total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            test_correct += predicted.eq(labels).sum().item()
            test_total += inputs.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = test_correct / test_total
    test_loss_avg = test_loss / test_total

    artifacts_dir = Path("/app/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (Test Acc: {test_acc:.4f})")
    cm_path = str(artifacts_dir / "confusion_matrix.png")
    fig.savefig(cm_path, bbox_inches="tight", dpi=100)
    plt.close(fig)

    report = classification_report(all_labels, all_preds, target_names=CLASSES, output_dict=True)
    report_path = str(artifacts_dir / "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    mlflow.set_tracking_uri(MLFLOW_URI)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({"test_loss": test_loss_avg, "test_accuracy": test_acc})
        mlflow.log_artifact(cm_path, "plots")
        mlflow.log_artifact(report_path, "reports")

    logger.info(f"Evaluation complete. Test accuracy: {test_acc:.4f}")

    return {
        **training_result,
        "test_acc": test_acc,
        "test_loss": test_loss_avg,
        "cm_path": cm_path,
        "report_path": report_path,
    }


# ─────────────────────────── Step 7: register_model ─────────────────────────

@step(settings={"docker": _docker_register})
def register_model(eval_result: Dict[str, Any]) -> Annotated[str, "model_version"]:
    """Register model in MLflow Model Registry."""
    from src.models.cnn_model import get_model

    run_id = eval_result["run_id"]
    model_path = eval_result["model_path"]
    test_acc = eval_result["test_acc"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(eval_result["params"]["num_classes"],
                      eval_result["params"]["dropout_rate"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    mlflow.set_tracking_uri(MLFLOW_URI)

    with mlflow.start_run(run_id=run_id):
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="cifar10_cnn",
            registered_model_name="cifar10-cnn-classifier",
        )

    client = mlflow.MlflowClient()
    versions = client.search_model_versions("name='cifar10-cnn-classifier'")
    latest_version = max(versions, key=lambda v: int(v.version))

    if test_acc >= float(os.getenv("MIN_ACCURACY_THRESHOLD", "0.70")):
        client.transition_model_version_stage(
            name="cifar10-cnn-classifier",
            version=latest_version.version,
            stage="Staging",
        )
        logger.info(f"Model v{latest_version.version} promoted to Staging (acc={test_acc:.4f})")
    else:
        logger.warning(f"Model accuracy {test_acc:.4f} below threshold, staying in None stage")

    return str(latest_version.version)


# ─────────────────────────── Step 8: export_model ───────────────────────────

@step(settings={"docker": _docker_export})
def export_model(eval_result: Dict[str, Any], model_version: str) -> Annotated[str, "export_path"]:
    """Export model to TorchScript + ONNX for serving."""
    from src.models.cnn_model import get_model

    device = torch.device("cpu")
    model = get_model(eval_result["params"]["num_classes"],
                      eval_result["params"]["dropout_rate"]).to(device)
    model.load_state_dict(torch.load(eval_result["model_path"], map_location=device))
    model.eval()

    export_dir = Path("/app/artifacts/serving")
    export_dir.mkdir(parents=True, exist_ok=True)

    # TorchScript
    example_input = torch.randn(1, 3, 32, 32)
    traced_model = torch.jit.trace(model, example_input)
    ts_path = str(export_dir / f"cifar10_cnn_v{model_version}.pt")
    traced_model.save(ts_path)

    # ONNX
    onnx_path = str(export_dir / f"cifar10_cnn_v{model_version}.onnx")
    torch.onnx.export(
        model, example_input, onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11,
    )

    metadata = {
        "model_version": model_version,
        "run_id": eval_result["run_id"],
        "test_accuracy": eval_result["test_acc"],
        "input_shape": [1, 3, 32, 32],
        "output_shape": [1, 10],
        "classes": list(CLASSES),
        "mean": eval_result["mean"],
        "std": eval_result["std"],
        "torchscript_path": ts_path,
        "onnx_path": onnx_path,
    }
    meta_path = str(export_dir / "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    mlflow.set_tracking_uri(MLFLOW_URI)
    with mlflow.start_run(run_id=eval_result["run_id"]):
        mlflow.log_artifact(ts_path, "serving")
        mlflow.log_artifact(onnx_path, "serving")
        mlflow.log_artifact(meta_path, "serving")

    logger.info(f"Model exported: TorchScript → {ts_path}, ONNX → {onnx_path}")
    return ts_path
