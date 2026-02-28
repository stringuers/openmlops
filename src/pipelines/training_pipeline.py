"""ZenML training pipeline for CIFAR-10 CNN."""
from zenml import pipeline
from zenml.logger import get_logger

from src.steps.training_steps import (
    ingest_data,
    validate_data,
    split_data,
    preprocess,
    train_model,
    evaluate_model,
    register_model,
    export_model,
)

logger = get_logger(__name__)


@pipeline(name="cifar10_training_pipeline", enable_cache=False)
def training_pipeline():
    """Full training pipeline: ingest → validate → split → preprocess → train → evaluate → register → export."""
    data_path = ingest_data()
    is_valid = validate_data(data_path)
    split_info = split_data(data_path)
    preprocessed_info = preprocess(split_info)
    training_result = train_model(preprocessed_info)
    eval_result = evaluate_model(training_result)
    model_version = register_model(eval_result)
    export_path = export_model(eval_result, model_version)
    return export_path


if __name__ == "__main__":
    training_pipeline()
