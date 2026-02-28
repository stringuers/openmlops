.PHONY: help infra-up infra-down setup build-steps train monitor monitor-drift logs-mlflow logs-train clean dvc-push dvc-pull git-init

COMPOSE = docker compose
COMPOSE_FILE = docker-compose.yml

help: ## Show this help message
	@echo "OpenMLOps Challenge - CIFAR-10 CNN Classifier"
	@echo "=============================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

## ─── Infrastructure ──────────────────────────────────────────────────────
infra-up: ## Start all infrastructure services (MinIO, MLflow, ZenML, PostgreSQL)
	@echo "🚀 Starting infrastructure..."
	$(COMPOSE) -f $(COMPOSE_FILE) up -d minio minio-init postgres mlflow zenml
	@echo "⏳ Waiting for services to be healthy..."
	@sleep 15
	@echo "✅ Infrastructure ready!"
	@echo "  MLflow UI:  http://localhost:5001"
	@echo "  ZenML UI:   http://localhost:8080"
	@echo "  MinIO UI:   http://localhost:9001"

infra-down: ## Stop all infrastructure services
	@echo "🛑 Stopping infrastructure..."
	$(COMPOSE) -f $(COMPOSE_FILE) down
	@echo "✅ Infrastructure stopped"

infra-clean: ## Stop and remove all volumes
	@echo "🗑️  Removing all containers and volumes..."
	$(COMPOSE) -f $(COMPOSE_FILE) down -v --remove-orphans
	@echo "✅ Clean shutdown complete"

## ─── ZenML Setup ───────────────────────────────────────────
setup: ## Configure ZenML stack (mlflow_stack) -- run once after infra-up
	@echo "⚙️  Configuring ZenML mlflow_stack..."
	$(COMPOSE) -f $(COMPOSE_FILE) --profile setup up zenml-setup --build
	@echo "✅ ZenML stack configured! Check http://localhost:8080"

build-steps: ## Build all 12 per-step Docker images
	@echo "🔨 Building all step Docker images..."
	bash scripts/build_step_images.sh

## ─── Data ────────────────────────────────────────────────────────────────
dvc-push: ## Download CIFAR-10 and push to MinIO via DVC
	@echo "📦 Pushing data to DVC remote..."
	@docker run --rm -v "$$(pwd):/app" -w /app --network openmlops_mlops_network -e AWS_S3_ENDPOINT=http://minio:9000 -e DATA_DIR=data openmlops-base:latest bash scripts/dvc_push_data.sh

dvc-pull: ## Pull data from DVC remote
	@echo "📥 Pulling data from DVC remote..."
	dvc pull

## ─── Pipelines ───────────────────────────────────────────────────────────
train: ## Run the full training pipeline
	@echo "🏋️  Running training pipeline..."
	$(COMPOSE) -f $(COMPOSE_FILE) --profile train up trainer --build
	@echo "✅ Training complete! Check MLflow at http://localhost:5001"

monitor: ## Run the monitoring pipeline (no drift)
	@echo "📊 Running monitoring pipeline..."
	$(COMPOSE) -f $(COMPOSE_FILE) --profile monitor up monitor --build
	@echo "✅ Monitoring complete! Check MLflow at http://localhost:5001"

monitor-drift: ## Run monitoring pipeline WITH drift injection (triggers retrain)
	@echo "⚠️  Running monitoring pipeline WITH drift injection..."
	$(COMPOSE) -f $(COMPOSE_FILE) --profile monitor-drift up monitor-drift --build
	@echo "⚡ Drift triggered! Check monitoring results."

## ─── Logs ────────────────────────────────────────────────────────────────
logs-mlflow: ## Tail MLflow server logs
	$(COMPOSE) -f $(COMPOSE_FILE) logs -f mlflow

logs-zenml: ## Tail ZenML server logs
	$(COMPOSE) -f $(COMPOSE_FILE) logs -f zenml

logs-train: ## Tail trainer logs
	$(COMPOSE) -f $(COMPOSE_FILE) logs -f trainer

logs-monitor: ## Tail monitor logs
	$(COMPOSE) -f $(COMPOSE_FILE) logs -f monitor

## ─── Git ─────────────────────────────────────────────────────────────────
git-init: ## Initialize Git repository and make first commit
	git init
	git add .
	git commit -m "feat: initial OpenMLOps project setup"
	@echo "✅ Git repository initialized!"

## ─── Quick start ─────────────────────────────────────────────────────────
quickstart: infra-up ## Full quickstart: start infra + setup + build-steps + train + monitor
	@echo "⏳ Waiting 30s for all services to fully start..."
	@sleep 30
	$(MAKE) setup
	$(MAKE) build-steps
	$(MAKE) train
	$(MAKE) monitor
	@echo ""
	@echo "🎉 OpenMLOps pipeline completed!"
	@echo "  📊 MLflow UI:  http://localhost:5001"
	@echo "  🔄 ZenML UI:   http://localhost:8080"
	@echo "  🪣 MinIO UI:   http://localhost:9001"
