#!/bin/bash
# docker_entrypoint.sh — runtime setup for ZenML + DVC before running pipeline

set -e

ZENML_STORE_URL="${ZENML_STORE_URL:-http://zenml:8080}"
ZENML_USERNAME="${ZENML_STORE_USERNAME:-admin}"
ZENML_PASSWORD="${ZENML_STORE_PASSWORD:-Admin123#}"

echo "╔══════════════════════════════════════════╗"
echo "║  OpenMLOps — Container Startup           ║"
echo "╚══════════════════════════════════════════╝"

# ─── DVC configuration ───────────────────────────────────────────────────────
echo "🔧 Configuring DVC remote..."
dvc remote modify minio endpointurl "${AWS_S3_ENDPOINT:-http://minio:9000}" 2>/dev/null || true
git config --global user.email "openmlops@ci" 2>/dev/null || true
git config --global user.name "OpenMLOps" 2>/dev/null || true

# ─── ZenML initialization ────────────────────────────────────────────────────
echo "🔗 Connecting to ZenML server at ${ZENML_STORE_URL}..."

# Initialize local ZenML repo if needed
if [ ! -d "/app/.zen" ]; then
    zenml init 2>/dev/null || true
fi

# Connect to remote ZenML server
zenml connect \
    --url "${ZENML_STORE_URL}" \
    --username "${ZENML_USERNAME}" \
    --password "${ZENML_PASSWORD}" \
    --no-verify-ssl 2>/dev/null || {
    echo "⚠️  Could not connect to ZenML server, will try running locally..."
}

# Wait for ZenML server to be ready (up to 60 s)
MAX_WAIT=60
WAIT=0
until zenml stack list &>/dev/null || [ $WAIT -ge $MAX_WAIT ]; do
    echo "⏳ Waiting for ZenML server ($WAIT/$MAX_WAIT s)..."
    sleep 5
    WAIT=$((WAIT+5))
done

# Set stack if mlflow_stack exists, else use default
if zenml stack list 2>/dev/null | grep -q "mlflow_stack"; then
    zenml stack set mlflow_stack 2>/dev/null || true
    echo "✅ ZenML stack: mlflow_stack"
else
    echo "⚠️  mlflow_stack not found, using default stack"
fi

echo "🚀 Starting pipeline..."
exec "$@"
