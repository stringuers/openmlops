#!/bin/bash
# setup_zenml_stack.sh
# Configures the ZenML mlflow_stack with local_docker step operator.
# Run as: docker exec -it openmlops-zenml-setup bash /app/scripts/setup_zenml_stack.sh
# Or via: make setup

set -e

MINIO_USER="${MINIO_ROOT_USER:-minioadmin}"
MINIO_PASS="${MINIO_ROOT_PASSWORD:-minioadmin123}"

echo "╔══════════════════════════════════════════════════════╗"
echo "║        ZenML Stack Configuration                     ║"
echo "║  (local_docker step operator + mlflow tracker)       ║"
echo "╚══════════════════════════════════════════════════════╝"

echo "📦 Installing ZenML integrations..."
zenml integration install mlflow s3 docker evidently -y || true

# 1) MLflow experiment tracker
zenml experiment-tracker register mlflow_tracker \
    --flavor=mlflow \
    --tracking_uri=http://mlflow:5000 \
    --tracking_token="dummy-token" || echo "  mlflow_tracker already registered"

# 2) MinIO secret
zenml secret create minio_zenml_secret \
    --aws_access_key_id="${MINIO_USER}" \
    --aws_secret_access_key="${MINIO_PASS}" || echo "  minio_zenml_secret already exists"

# 3) S3 artifact store → MinIO
zenml artifact-store register minio_artifacts \
    --flavor=s3 \
    --path="s3://mlflow-artifacts/zenml" \
    --authentication_secret=minio_zenml_secret \
    --client_kwargs='{"endpoint_url": "http://minio:9000", "region_name": "us-east-1"}' || echo "  minio_artifacts already registered"

# 4) Local orchestrator
zenml orchestrator register local_orch --flavor=local || echo "  local_orch already registered"

# 5) Local Docker step operator
zenml step-operator register local_docker \
    --flavor=local_docker || echo "  local_docker step operator already registered"

# 6) Register the full mlflow_stack with docker step operator
zenml stack register mlflow_stack \
    -o local_orch \
    -a minio_artifacts \
    -e mlflow_tracker \
    -s local_docker || echo "  mlflow_stack already registered"

# 7) Set as default
zenml stack set mlflow_stack

echo ""
echo "✅ ZenML stack configured!"
echo ""
zenml stack describe
echo ""
echo "📌 Next step: build all step Docker images"
echo "   bash scripts/build_step_images.sh"
