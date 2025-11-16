# ============================================================
# Supply Chain Analytics — Makefile
# ============================================================
.PHONY: install install-dev generate-data run-analysis create-report \
        streamlit test test-cov lint format typecheck clean help

PYTHON := python
PIP    := pip
APP    := app/streamlit_app.py

# ------------------------------------------------------------
help:           ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' | sort

# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------
install:        ## Install production dependencies
	$(PIP) install -r requirements.txt

install-dev:    ## Install all dependencies including dev tools
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov pytest-mock black isort flake8 mypy

# ------------------------------------------------------------
# Data & Analysis
# ------------------------------------------------------------
generate-data:  ## Generate synthetic procurement dataset (50K POs)
	$(PYTHON) scripts/generate_data.py --n-pos 50000 --output-dir data/sample

generate-data-small:  ## Generate small dataset for testing (5K POs)
	$(PYTHON) scripts/generate_data.py --n-pos 5000 --output-dir data/sample

run-analysis:   ## Run full analytics pipeline and print summary
	$(PYTHON) scripts/run_analysis.py --output-dir results

run-analysis-fast:  ## Run analysis without demand forecast (faster)
	$(PYTHON) scripts/run_analysis.py --output-dir results --skip-forecast

create-report:  ## Generate PDF executive report
	$(PYTHON) scripts/create_report.py --output reports/procurement_executive_report.pdf

# ------------------------------------------------------------
# Dashboard
# ------------------------------------------------------------
streamlit:      ## Launch Streamlit dashboard (opens in browser)
	streamlit run $(APP) --server.port 8501 --server.headless false

streamlit-headless:  ## Launch Streamlit in headless mode (CI/server)
	streamlit run $(APP) --server.port 8501 --server.headless true

# ------------------------------------------------------------
# Testing
# ------------------------------------------------------------
test:           ## Run all unit tests
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov:       ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-fast:      ## Run tests, stop at first failure
	$(PYTHON) -m pytest tests/ -x -v --tb=short

# ------------------------------------------------------------
# Code Quality
# ------------------------------------------------------------
lint:           ## Run flake8 linter
	flake8 src/ app/ scripts/ tests/ --max-line-length=120 --ignore=E203,W503

format:         ## Auto-format with black + isort
	black src/ app/ scripts/ tests/ config/ --line-length=120
	isort src/ app/ scripts/ tests/ config/ --profile=black

format-check:   ## Check formatting without modifying files
	black src/ app/ scripts/ tests/ config/ --line-length=120 --check
	isort src/ app/ scripts/ tests/ config/ --profile=black --check-only

typecheck:      ## Run mypy static type checks
	mypy src/ --ignore-missing-imports --no-strict-optional

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
clean:          ## Remove generated data, reports, and Python artifacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache/ .mypy_cache/ htmlcov/ .coverage 2>/dev/null || true
	@echo "Clean complete."

clean-data:     ## Remove generated data files (keeps directory structure)
	rm -f data/sample/*.csv data/sample/*.parquet
	rm -f data/processed/*.parquet data/processed/*.csv
	@echo "Data files removed."

dirs:           ## Create all required directories
	mkdir -p data/{raw,processed,sample} reports results logs

# ------------------------------------------------------------
# All-in-one
# ------------------------------------------------------------
all: install generate-data run-analysis create-report  ## Full pipeline: install → generate → analyze → report
	@echo "All steps complete!"
