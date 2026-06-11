.PHONY: help install lint format test ci act clean download-data train-tokenizer train train-base train-ddp-2 train-ddp-4 train-ddp-8 train-ddp-short-2 train-ddp-short-4 train-ddp-short-8 smoke-ddp-2 smoke-ddp-preflight smoke-ddp-cpu-preflight average-base smoke train-mlflow mlflow-ui evaluate evaluate-base evaluate-single

EVAL_CONFIG ?= evaluate_base

help:
	@echo "Targets:"
	@echo "  install  Install project dependencies in editable mode"
	@echo "  download-data  Download raw WMT14 de-en data"
	@echo "  train-tokenizer  Train the shared BPE tokenizer"
	@echo "  train    Run the default training config"
	@echo "  train-base  Run Base EN-DE training config (Phase 4)"
	@echo "  smoke-ddp-2  Run smoke config via torchrun on 2 GPUs"
	@echo "  train-ddp-2  Run default training via torchrun on 2 GPUs"
	@echo "  train-ddp-4  Run default training via torchrun on 4 GPUs"
	@echo "  train-ddp-8  Run default training via torchrun on 8 GPUs"
	@echo "  train-ddp-short-2  Run short DDP training validation on 2 GPUs"
	@echo "  train-ddp-short-4  Run short DDP training validation on 4 GPUs"
	@echo "  train-ddp-short-8  Run short DDP training validation on 8 GPUs"
	@echo "  smoke-ddp-preflight  Run smoke with torchrun nproc=1 (DDP path preflight)"
	@echo "  smoke-ddp-cpu-preflight  Run smoke with torchrun nproc=2 on CPU/Gloo"
	@echo "  average-base  Average last N base checkpoints into averaged.pt"
	@echo "  smoke    Run the smoke-test config"
	@echo "  evaluate  Run evaluation using EVAL_CONFIG"
	@echo "  evaluate-base  Run the averaged base checkpoint evaluation"
	@echo "  evaluate-single  Run a representative single-checkpoint evaluation"
	@echo "  mlflow-ui  Start the local MLflow UI on port 5000"
	@echo "  lint     Run ruff"
	@echo "  format   Run black"
	@echo "  test     Run pytest"
	@echo "  ci       Run lint and tests"
	@echo "  act      Run the GitHub Actions workflow locally with act"
	@echo "  clean    Remove common cache/build artifacts"


download-data:
	@echo "Downloading WMT14 de-en raw data..."
	python scripts/download_data.py --output-dir data/raw/wmt14_de_en

train-tokenizer:
	@echo "Training shared BPE tokenizer (default 32k vocab, sample 10k lines)..."
	python -m scripts.train_tokenizer --raw-dir data/raw/wmt14_de_en --output-dir data/processed/bpe --vocab-size 32000 --sample-size 0
 
install:
	python -m pip install -e '.[dev]'

train:
	python train.py --config-name train

train-base:
	python train.py --config-name=base_en_de

smoke-ddp-2:
	torchrun --standalone --nproc_per_node=2 train.py --config-name smoke distributed.enabled=true training.device=cuda smoke.steps=20 mlflow.enabled=false

train-ddp-2:
	torchrun --standalone --nproc_per_node=2 train.py distributed.enabled=true training.device=cuda

train-ddp-4:
	torchrun --standalone --nproc_per_node=4 train.py distributed.enabled=true training.device=cuda

train-ddp-8:
	torchrun --standalone --nproc_per_node=8 train.py distributed.enabled=true training.device=cuda

train-ddp-short-2:
	torchrun --standalone --nproc_per_node=2 train.py distributed.enabled=true training.device=cuda mlflow.enabled=false training.auto_resume_latest=false training.max_steps=100 training.log_every=10 training.validate_every=1000 training.save_every=1000 data.max_train_samples=2000 data.batch_size=4

train-ddp-short-4:
	torchrun --standalone --nproc_per_node=4 train.py distributed.enabled=true training.device=cuda mlflow.enabled=false training.auto_resume_latest=false training.max_steps=100 training.log_every=10 training.validate_every=1000 training.save_every=1000 data.max_train_samples=2000 data.batch_size=2

train-ddp-short-8:
	torchrun --standalone --nproc_per_node=8 train.py distributed.enabled=true training.device=cuda mlflow.enabled=false training.auto_resume_latest=false training.max_steps=100 training.log_every=10 training.validate_every=1000 training.save_every=1000 data.max_train_samples=2000 data.batch_size=1

average-base:
# 	make average-base CHECKPOINT_DIR=checkpoints/check0dir
# 	python scripts/average_checkpoints.py --checkpoint-dir checkpoints/base_en_de --n-last 5 --output checkpoints/base_en_de/averaged.pt
	python scripts/average_checkpoints.py --checkpoint-dir $(CHECKPOINT_DIR) --n-last 5 --output checkpoints/base_en_de/averaged.pt

smoke:
	python train.py --config-name smoke

smoke-ddp-preflight:
	torchrun --standalone --nproc_per_node=1 train.py --config-name smoke distributed.enabled=true training.device=cuda

smoke-ddp-cpu-preflight:
	torchrun --standalone --nproc_per_node=2 train.py --config-name smoke distributed.enabled=true distributed.backend=gloo training.device=cpu

evaluate:
	python -m src.evaluation.generate --config-name $(EVAL_CONFIG)

evaluate-base:
	python -m src.evaluation.generate --config-name evaluate_base

evaluate-single:
	python -m src.evaluation.generate --config-name evaluate_single

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///runs/mlflow/mlflow.db --port 5000

lint:
	python -m ruff check src tests

format:
	python -m black .

test:
	python -m pytest

ci: lint test

act:
	act pull_request

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info