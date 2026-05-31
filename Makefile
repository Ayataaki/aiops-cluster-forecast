# Makefile – AIOps Predictive Observability
# Usage: make <target>

.PHONY: help setup data diagnostics train compare api test clean

PYTHON := python
SRC    := src

help:
	@echo "Available targets:"
	@echo "  make setup        Install Python dependencies"
	@echo "  make data         Generate synthetic dataset"
	@echo "  make diagnostics  Run stationarity tests + decomposition"
	@echo "  make train        Train XGBoost (fast, ~2 min)"
	@echo "  make train-all    Train XGBoost + Informer + Autoformer"
	@echo "  make compare      Compare all models and produce results/"
	@echo "  make api          Start FastAPI server on port 8000"
	@echo "  make notebook     Launch Jupyter Lab"
	@echo "  make test         Run pytest suite"
	@echo "  make clean        Remove generated artefacts"

setup:
	pip install -r requirements.txt

data:
	$(PYTHON) scripts/generate_synthetic.py

diagnostics: data
	$(PYTHON) $(SRC)/analytics/diagnostics.py

train: data
	$(PYTHON) $(SRC)/models/model_comparison.py --no-transformers

train-all: data
	$(PYTHON) $(SRC)/models/model_comparison.py

compare: train
	@echo "Results saved to results/model_comparison.csv"

api:
	uvicorn $(SRC).api.main:app --host 0.0.0.0 --port 8000 --reload

notebook:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=''

test:
	pytest tests/ -v --tb=short

clean:
	rm -rf results/ models_saved/ __pycache__ src/**/__pycache__
	find . -name "*.pyc" -delete
