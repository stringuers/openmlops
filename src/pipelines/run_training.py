#!/usr/bin/env python3
"""Entrypoint script to run the training pipeline."""
import sys
import os

sys.path.insert(0, "/app")

from src.pipelines.training_pipeline import training_pipeline

if __name__ == "__main__":
    print("🚀 Starting CIFAR-10 Training Pipeline...")
    training_pipeline()
    print("✅ Training pipeline completed!")
