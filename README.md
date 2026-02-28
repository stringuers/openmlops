# OpenMLOps Challenge — CIFAR-10 CNN Classifier

> **Fully open-source MLOps workflow** — Git · DVC · MLflow · ZenML · Evidently · Docker Compose
> **Fully Containerized** — Each ZenML `@step` runs in its own dedicated Docker container.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                       Docker Compose Stack                            │
│                                                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌────────────────┐     │
│  │  MinIO   │  │ Postgres │  │    MLflow    │  │ ZenML Server   │     │
│  │ S3/DVC   │  │ MLflow DB│  │  Tracking    │  │ Orchestrator   │     │
│  │ :9000    │  │ :5432    │  │  :5000       │  │ :8080          │     │
│  └──────────┘  └──────────┘  └──────────────┘  └────────────────┘     │
│                                                                       │
│  ┌─────────────────────────────────┐ ┌──────────────────────────────┐ │
│  │     training_pipeline           │ │     monitoring_pipeline      │ │
│  │                                 │ │                              │ │
│  │ [🐳 step-ingest] ingest_data    │ │ [🐳 step-collect]   collect  │ │
│  │ [🐳 step-validate] validate_data│ │ [🐳 step-evidently] evidently│ │
│  │ [🐳 step-split]  split_data     │ │ [🐳 step-trigger]   trigger  │ │
│  │ [🐳 step-preprocess] preprocess │ │ [🐳 step-store]     store    │ │
│  │ [🐳 step-train]  train_model    │ │                              │ │
│  │ [🐳 step-evaluate] evaluate     │ │                              │ │
│  │ [🐳 step-register] register     │ │                              │ │
│  │ [🐳 step-export] export_model   │ │                              │ │
│  └─────────────────────────────────┘ └──────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────┘
```
**Per-Step Docker Execution:**  
We use the ZenML `local_docker` step operator. When a pipeline runs, ZenML spins up a *distinct* Docker container for each step, tailored with only the dependencies that step needs (e.g., the `evaluate_model` container has `scikit-learn` and `seaborn`, while `train_model` does not).

---

## Prerequisites

- Docker ≥ 24.0 with Docker Compose plugin
- Git

---

## Quickstart

### 1. Clone and configure

```bash
git clone <your-repo-url>
cd openmlops
cp .env .env.local   # optional: override credentials
```

### 2. Start infrastructure

```bash
make infra-up
```

Wait ~30 s, then verify:
- **MLflow UI**: http://localhost:5001
- **ZenML UI**: http://localhost:8080 (admin / Admin123#)
- **MinIO UI**: http://localhost:9001 (minioadmin / minioadmin123)

### 3. Configure ZenML stack

```bash
make setup
```

This registers the **mlflow_stack** inside ZenML: local orchestrator + MinIO artifact store + MLflow experiment tracker + **local_docker step operator**.

### 4. Build per-step Docker images

Because each pipeline step runs in its own container, you must build the images:

```bash
make build-steps
```
*(This builds 1 base image and 12 lightweight per-step images).*

### 5. Version data with DVC

```bash
make dvc-push          # downloads CIFAR-10, pushes to MinIO

git add data/cifar-10-batches-py.dvc data/.gitignore
git commit -m "feat: track CIFAR-10 dataset with DVC"
```

### 6. Run training pipeline

```bash
make train
```

Runs the full ZenML training pipeline. **Watch the logs as ZenML spins up a Docker container for each step:**

| Step | Container Image |
|------|-----------------|
| `ingest_data` | `openmlops-step-ingest` |
| `validate_data` | `openmlops-step-validate` |
| `split_data` | `openmlops-step-split` |
| `preprocess` | `openmlops-step-preprocess` |
| `train_model` | `openmlops-step-train` |
| `evaluate_model` | `openmlops-step-evaluate` |
| `register_model` | `openmlops-step-register` |
| `export_model` | `openmlops-step-export` |

### 7. View results

```
MLflow experiments:   http://localhost:5001  → cifar10-cnn
MLflow model registry: http://localhost:5001 → cifar10-cnn-classifier
ZenML pipeline runs:  http://localhost:8080
```

### 8. Run monitoring pipeline

```bash
# Monitoring without drift (baseline check)
make monitor

# Monitoring WITH drift injection → triggers retrain flag
make monitor-drift

# Enable automatic retraining on drift
AUTO_RETRAIN=true make monitor-drift
```

The monitoring pipeline:

| Step | Container Image |
|------|-----------------|
| `collect_inference_data` | `openmlops-step-collect` |
| `run_evidently_report` | `openmlops-step-evidently` |
| `trigger_decision` | `openmlops-step-trigger` |
| `store_monitoring_artifacts`| `openmlops-step-store` |

**Or do everything in one command:** `make quickstart`

---

## Project Structure

```
openmlops/
├── docker/
│   ├── base.Dockerfile             # Base image with torch/zenml
│   └── steps/                      # 12 per-step Dockerfiles
├── requirements/
│   ├── base.txt                    # Core ML deps
│   ├── train.txt                   # Train/eval deps
│   └── monitoring.txt              # Evidently deps
├── src/
│   ├── models/cnn_model.py         
│   ├── steps/
│   │   ├── training_steps.py       
│   │   └── monitoring_steps.py     
│   ├── pipelines/
│   │   ├── training_pipeline.py    
│   │   └── monitoring_pipeline.py  
├── data/
│   └── cifar10.dvc                 # DVC pointer
├── scripts/
│   ├── build_step_images.sh        # Builds all 13 step images
│   ├── setup_zenml_stack.sh        # Registers docker step operator
│   └── dvc_push_data.sh            
├── .dvc/config                     # DVC remote s3://dvc-store
├── docker-compose.yml              
├── .env                            
└── Makefile                        
```

---

## MLOps Stack

| Tool | Role | Port |
|------|------|------|
| **Git** | Source code versioning | — |
| **DVC** | Data versioning (remote: MinIO S3 `dvc-store`) | — |
| **MLflow** | Experiment tracking, Model Registry, Artifact store | 5001 (host) / 5000 (net) |
| **ZenML** | Pipeline orchestration & artifact lineage | 8080 |
| **Evidently** | Drift detection, HTML reports | — |
| **MinIO** | S3-compatible storage for DVC + MLflow artifacts | 9000/9001 |
| **PostgreSQL** | MLflow backend metadata store | 5432 |
| **Docker** | Containerized, reproducible execution (per-step) | — |

---

## All Commands

```bash
make help            # Show all commands
make infra-up        # Start MinIO, Postgres, MLflow, ZenML
make infra-down      # Stop all services
make infra-clean     # Stop + remove all volumes
make setup           # Configure ZenML mlflow_stack (run once)
make build-steps     # Build all 12 per-step Docker images
make dvc-push        # Download CIFAR-10 + push to MinIO via DVC
make train           # Run training pipeline
make monitor         # Run monitoring pipeline (no drift)
make monitor-drift   # Run monitoring WITH drift injection
make quickstart      # infra-up + setup + build-steps + train + monitor
```

---

*OpenMLOps Challenge — Ecole Polytechnique de Sousse — 2025*
