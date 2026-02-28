#!/bin/bash
# build_step_images.sh
# Builds the base image and all 12 per-step Docker images.
# Run from the project root: bash scripts/build_step_images.sh

set -e

cd "$(dirname "$0")/.."

REGISTRY="${REGISTRY:-}"   # e.g. "myregistry.io/" if pushing to remote

echo "╔══════════════════════════════════════════════════════╗"
echo "║     OpenMLOps — Build All Step Docker Images         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

tag() { echo "${REGISTRY}$1"; }

# ─── 1. Base image ──────────────────────────────────────────────────────────
echo "📦 [0/13] Building base image..."
docker build -f docker/base.Dockerfile -t "$(tag openmlops-base:latest)" .
echo "✅  openmlops-base:latest"
echo ""

# ─── 2. Training pipeline steps ─────────────────────────────────────────────
TRAINING_STEPS=(
  "ingest_data:openmlops-step-ingest"
  "validate_data:openmlops-step-validate"
  "split_data:openmlops-step-split"
  "preprocess:openmlops-step-preprocess"
  "train_model:openmlops-step-train"
  "evaluate_model:openmlops-step-evaluate"
  "register_model:openmlops-step-register"
  "export_model:openmlops-step-export"
)

STEP_NUM=1
for entry in "${TRAINING_STEPS[@]}"; do
  STEP="${entry%%:*}"
  IMAGE="${entry##*:}"
  echo "🔨 [$STEP_NUM/12] Building ${IMAGE}:latest (step: ${STEP})..."
  docker build \
    -f "docker/steps/${STEP}.Dockerfile" \
    -t "$(tag ${IMAGE}:latest)" \
    .
  echo "✅  ${IMAGE}:latest"
  STEP_NUM=$((STEP_NUM + 1))
done

echo ""

# ─── 3. Monitoring pipeline steps ───────────────────────────────────────────
MONITORING_STEPS=(
  "collect_inference_data:openmlops-step-collect"
  "run_evidently_report:openmlops-step-evidently"
  "trigger_decision:openmlops-step-trigger"
  "store_monitoring_artifacts:openmlops-step-store"
)

for entry in "${MONITORING_STEPS[@]}"; do
  STEP="${entry%%:*}"
  IMAGE="${entry##*:}"
  echo "🔨 [$STEP_NUM/12] Building ${IMAGE}:latest (step: ${STEP})..."
  docker build \
    -f "docker/steps/${STEP}.Dockerfile" \
    -t "$(tag ${IMAGE}:latest)" \
    .
  echo "✅  ${IMAGE}:latest"
  STEP_NUM=$((STEP_NUM + 1))
done

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ✅  All 13 images built successfully!              ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Built images:"
docker images --filter "reference=openmlops-*" --format "  {{.Repository}}:{{.Tag}}  ({{.Size}})"

if [ -n "$REGISTRY" ]; then
  echo ""
  echo "📤 Pushing to registry: ${REGISTRY}..."
  docker push "$(tag openmlops-base:latest)"
  for entry in "${TRAINING_STEPS[@]}" "${MONITORING_STEPS[@]}"; do
    IMAGE="${entry##*:}"
    docker push "$(tag ${IMAGE}:latest)"
  done
  echo "✅ All images pushed!"
fi
