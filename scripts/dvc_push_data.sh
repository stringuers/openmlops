#!/bin/bash
# dvc_push_data.sh — Download CIFAR-10 and push to MinIO via DVC
set -e

DATA_DIR="${DATA_DIR:-data}"

echo "╔══════════════════════════════════════════╗"
echo "║  DVC — Push CIFAR-10 to MinIO            ║"
echo "╚══════════════════════════════════════════╝"

echo "📦 Downloading CIFAR-10 dataset..."
python3 -c "
import torchvision, os
data_dir = '${DATA_DIR}'
os.makedirs(data_dir, exist_ok=True)
torchvision.datasets.CIFAR10(root=data_dir, train=True,  download=True)
torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)
print('✅ CIFAR-10 downloaded successfully')
"

CIFAR_PATH="${DATA_DIR}/cifar-10-batches-py"
if [ ! -d "$CIFAR_PATH" ]; then
    echo "❌ Expected $CIFAR_PATH to exist after download. Aborting."
    exit 1
fi

echo "📝 Adding data to DVC tracking..."
git config --global --add safe.directory /app
git init
dvc init || true
dvc add "${CIFAR_PATH}"


echo "⬆️  Pushing data to MinIO remote..."
dvc push

echo ""
echo "✅ Data versioned and pushed to MinIO dvc-store bucket!"
echo ""
echo "📌 Next — commit the DVC pointer file:"
echo "   git add data/cifar-10-batches-py.dvc data/.gitignore"
echo "   git commit -m 'feat: track CIFAR-10 dataset with DVC'"
