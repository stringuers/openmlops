#!/usr/bin/env python3
"""Entrypoint script to run the monitoring pipeline."""
import sys

sys.path.insert(0, "/app")

from src.pipelines.monitoring_pipeline import monitoring_pipeline

if __name__ == "__main__":
    print("📊 Starting CIFAR-10 Monitoring Pipeline...")
    monitoring_pipeline()
    print("✅ Monitoring pipeline completed!")
