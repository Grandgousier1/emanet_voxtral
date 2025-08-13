# Makefile for the Emanet Subtitling Pipeline

# -----------------------------------------------------------------------------
# Variables
# -----------------------------------------------------------------------------

# Python interpreter
PYTHON = python3
# Virtual environment directory
VENV_DIR = venv
# Activate script for the virtual environment
VENV_ACTIVATE = . $(VENV_DIR)/bin/activate;
# Requirements file for standard (GPU) setup
REQUIREMENTS = requirements.txt
# Requirements file for CPU-only setup
REQUIREMENTS_CPU = requirements-cpu.txt
# All python source files
PY_SOURCES = main.py utils/*.py

# -----------------------------------------------------------------------------
# Default Target
# -----------------------------------------------------------------------------

.PHONY: help
help:
	@echo "--- Emanet Subtitling Pipeline ---"
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install          - Creates a virtual environment and installs GPU dependencies from $(REQUIREMENTS)"
	@echo "  install-cpu      - Creates a virtual environment and installs CPU-only dependencies from $(REQUIREMENTS_CPU)"
	@echo "  run              - Runs the main script with default arguments (requires a --batch-list)"
	@echo "  batch            - Example: Runs the script with a specific batch list (e.g., make batch BATCH_FILE=videos.txt)"
	@echo "  debug            - Runs the main script with a sample debug command"
	@echo "  format           - Formats the code using 'black'"
	@echo "  clean            - Removes the virtual environment and __pycache__ directories"
	@echo "------------------------------------"

# -----------------------------------------------------------------------------
# Environment & Installation
# -----------------------------------------------------------------------------

$(VENV_DIR):
	$(PYTHON) -m venv $(VENV_DIR)

.PHONY: install
install: $(VENV_DIR)
	@echo "Activating virtual environment and installing dependencies from $(REQUIREMENTS)..."
	$(VENV_ACTIVATE) \
	pip install --upgrade pip; \
	pip install -r $(REQUIREMENTS);
	@echo "Installation complete."

.PHONY: install-cpu
install-cpu: $(VENV_DIR)
	@echo "Activating virtual environment and installing CPU-only dependencies from $(REQUIREMENTS_CPU)..."
	$(VENV_ACTIVATE) \
	pip install --upgrade pip; \
	pip install -r $(REQUIREMENTS_CPU);
	@echo "CPU-only installation complete."

# -----------------------------------------------------------------------------
# Execution Targets
# -----------------------------------------------------------------------------

.PHONY: run
run: $(VENV_DIR)
	@if [ -z "$(BATCH_FILE)" ]; then \
		echo "Error: BATCH_FILE is not set. Usage: make run BATCH_FILE=/path/to/your/list.txt"; \
		exit 1; \
	fi
	@echo "Running pipeline with batch file: $(BATCH_FILE)..."
	$(VENV_ACTIVATE) \
	$(PYTHON) main.py --batch-list $(BATCH_FILE)

# Alias for 'run' for more intuitive use
.PHONY: batch
batch: run

.PHONY: debug
debug: $(VENV_DIR)
	@echo "Running pipeline in debug mode with a sample batch file..."
	$(VENV_ACTIVATE) \
	$(PYTHON) main.py --batch-list examples/sample_videos.txt --download-video=false
	@echo "Debug run finished."

# -----------------------------------------------------------------------------
# Code Quality & Cleanup
# -----------------------------------------------------------------------------

.PHONY: format
format: $(VENV_DIR)
	@echo "Formatting code with black..."
	$(VENV_ACTIVATE) \
	pip install black; \
	black $(PY_SOURCES);
	@echo "Formatting complete."

.PHONY: clean
clean:
	@echo "Cleaning up project..."
	rm -rf $(VENV_DIR)
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm -rf output/*
	@echo "Cleanup complete."
