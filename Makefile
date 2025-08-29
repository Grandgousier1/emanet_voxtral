# Makefile - Automatisation tâches développement et déploiement B200

# Variables configuration
PYTHON := python3
PIP := pip3
PYTEST := pytest
PROJECT_NAME := voxtral-b200
VENV_DIR := .venv

# Couleurs pour output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Aide par défaut
.PHONY: help
help: ## Affiche cette aide
	@echo "$(BLUE)🚀 VOXTRAL B200 - Commandes disponibles$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# =============================================================================
# ENVIRONNEMENT DE DÉVELOPPEMENT
# =============================================================================

.PHONY: setup
setup: ## 🔥 Configuration de l'environnement de développement complet
	@echo "$(BLUE)🔧 Installation des dépendances de développement...$(NC)"
	$(MAKE) install-dev
	@echo "$(BLUE)🪝 Installation des pre-commit hooks...$(NC)"
	$(MAKE) install-hooks
	@echo "$(BLUE)✅ Validation de la configuration...$(NC)"
	$(MAKE) validate-setup
	@echo "$(GREEN)✅ Environnement prêt !$(NC)"

.PHONY: install
install: ## 📦 Installation des dépendances de production
	@echo "$(BLUE)📦 Installation des dépendances de production...$(NC)"
	$(PIP) install -e .

.PHONY: install-dev
install-dev: ## 🛠️ Installation des dépendances de développement (dev, docs, benchmark)
	@echo "$(BLUE)🛠️ Installation des dépendances de développement...$(NC)"
	$(PIP) install -e ".[dev,docs,benchmark,nlp]"

.PHONY: install-vllm
install-vllm: ## 🚀 Installation de vLLM pour les optimisations B200
	@echo "$(BLUE)🚀 Installation de vLLM...$(NC)"
	$(PIP) install -e ".[vllm]"

.PHONY: install-hooks
install-hooks: ## 🪝 Installation des pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

.PHONY: validate-setup
validate-setup: ## ✅ Validation de la configuration de l'environnement
	@echo "$(BLUE)✅ Validation de la configuration...$(NC)"
	@echo "$(YELLOW)Vérification des dépendances système...$(NC)"
	@if ! command -v ffmpeg > /dev/null; then \
		echo "$(RED)❌ Dépendance système manquante: ffmpeg n'est pas installé.$(NC)"; \
		echo "$(YELLOW)   Veuillez l'installer avec 'sudo apt-get install ffmpeg' ou 'conda install ffmpeg'.$(NC)"; \
		exit 1; \
	else \
		echo "$(GREEN)✅ ffmpeg est installé.$(NC)"; \
	fi
	@echo "$(YELLOW)Vérification des dépendances Python...$(NC)"
	@$(PYTHON) -c "import torch; print('PyTorch:', torch.__version__)"
	@$(PYTHON) -c "import torch; print('CUDA disponible:', torch.cuda.is_available())"
	@$(PYTHON) -c "import transformers; print('Transformers:', transformers.__version__)"
	@if command -v nvidia-smi > /dev/null; then \
		echo "$(GREEN)GPU détecté:$(NC)"; \
		nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; \
	else \
		echo "$(YELLOW)⚠️ Aucun GPU NVIDIA détecté$(NC)"; \
	fi

.PHONY: check-deps
check-deps: ## ⛑️ Vérifie la cohérence des dépendances installées
	@echo "$(BLUE)🔎 Vérification des conflits de dépendances...$(NC)"
	$(PIP) check



# =============================================================================
# QUALITÉ CODE
# =============================================================================

.PHONY: format
format: ## Formatage code (black + isort)
	@echo "$(BLUE)🎨 Formatage code...$(NC)"
	black .
	isort .

.PHONY: lint
lint: ## Linting code (flake8 + mypy + bandit)
	@echo "$(BLUE)🔍 Linting code...$(NC)"
	flake8 .
	mypy . --ignore-missing-imports
	bandit -c pyproject.toml -r .

.PHONY: check
check: format lint ## Formatage + linting complet

.PHONY: quality-report
quality-report: ## Génère rapport qualité code
	@echo "$(BLUE)📊 Génération rapport qualité...$(NC)"
	$(PYTHON) code_quality_analyzer.py --output code_quality_report.md --json
	@echo "$(GREEN)✅ Rapport généré: code_quality_report.md$(NC)"

# =============================================================================
# TESTS
# =============================================================================

.PHONY: test
test: ## Lance tous les tests avec une sortie améliorée
	@echo "$(BLUE)🧪 Lancement des tests (avec pytest-sugar)...$(NC)"
	$(PYTEST) -vv --showlocals

.PHONY: test-fast
test-fast: ## Tests rapides (sans GPU, sans slow)
	@echo "$(BLUE)⚡ Tests rapides...$(NC)"
	$(PYTEST) -v -m "not slow and not gpu" --maxfail=5

.PHONY: test-gpu
test-gpu: ## Tests nécessitant GPU
	@echo "$(BLUE)🎮 Tests GPU...$(NC)"
	$(PYTEST) -v -m "gpu" -s

.PHONY: test-b200
test-b200: ## Tests spécifiques B200
	@echo "$(BLUE)🚀 Tests B200...$(NC)"
	$(PYTEST) -v -m "b200" -s

.PHONY: test-integration
test-integration: ## Tests intégration
	@echo "$(BLUE)🔗 Tests intégration...$(NC)"
	$(PYTEST) -v -m "integration" -s

.PHONY: test-coverage
test-coverage: ## Tests avec couverture
	@echo "$(BLUE)📊 Tests avec couverture...$(NC)"
	$(PYTEST) --cov=. --cov-report=html --cov-report=term-missing

.PHONY: test-all
test-all: test-fast test-integration test-coverage ## Tous les tests + couverture

# =============================================================================
# BENCHMARK ET PERFORMANCE
# =============================================================================

.PHONY: benchmark
benchmark: ## Benchmark performance complet
	@echo "$(BLUE)🏃 Benchmark performance...$(NC)"
	$(PYTHON) benchmark.py --output benchmark_results.json

.PHONY: benchmark-basic
benchmark-basic: ## Benchmark opérations de base
	@echo "$(BLUE)⚡ Benchmark basique...$(NC)"
	$(PYTHON) benchmark.py --basic-only --output basic_benchmark.json

.PHONY: benchmark-b200
benchmark-b200: ## Benchmark optimisations B200
	@echo "$(BLUE)🚀 Benchmark B200...$(NC)"
	$(PYTHON) benchmark.py --b200-only --output b200_benchmark.json

.PHONY: benchmark-model
benchmark-model: ## Benchmark optimisations modèles
	@echo "$(BLUE)🧠 Benchmark modèles...$(NC)"
	$(PYTHON) benchmark.py --model-only --output model_benchmark.json

.PHONY: profile
profile: ## Profiling performance détaillé
	@echo "$(BLUE)🔬 Profiling performance...$(NC)"
	$(PYTHON) -m cProfile -o profile_results.prof main.py --help
	@echo "$(GREEN)✅ Résultats: profile_results.prof$(NC)"

# =============================================================================
# VALIDATION ET DEPLOYMENT
# =============================================================================

# Permet de passer des arguments au make, ex: make check-vram MODEL_ID=google/gemma-7b
MODEL_ID ?= "meta-llama/Meta-Llama-3-8B"

.PHONY: validate
validate: ## Validation complète avant déploiement
	@echo "$(BLUE)🔍 Validation complète...$(NC)"
	$(MAKE) check
	$(MAKE) test-fast
	$(MAKE) quality-report
	$(PYTHON) validator.py
	@echo "$(GREEN)✅ Validation réussie !$(NC)"

.PHONY: validate-deps
validate-deps: ## Valide l'installation des dépendances critiques (vllm, flash-attn...)
	@echo "$(BLUE)🔎 Validation des dépendances critiques...$(NC)"
	$(PYTHON) validate_dependencies.py

.PHONY: test-b200-atomic
test-b200-atomic: ## Teste l'API B200 en isolation
	@echo "$(BLUE)⚛️ Test atomique de l'API B200...$(NC)"
	$(PYTHON) test_b200_api_atomic.py

.PHONY: check-vram
check-vram: ## Vérifie l'empreinte VRAM d'un modèle (par défaut Llama-3-8B)
	@echo "$(BLUE)🧠 Vérification VRAM pour le modèle: $(MODEL_ID)...$(NC)"
	$(PYTHON) check_model_vram.py --model_id $(MODEL_ID)

.PHONY: validate-all-b200
validate-all-b200: ## Lance TOUTES les validations B200 dans l'ordre via l'orchestrateur
	@echo "$(BLUE)🚀🚀🚀 Lancement de la suite de validation complète pour B200...$(NC)"
	$(PYTHON) run_b200_validation.py

.PHONY: validate-b200
validate-b200: ## (OBSOLÈTE, utiliser validate-all-b200) Validation spécifique B200
	@echo "$(YELLOW)⚠️  Cette cible est obsolète. Utilisez 'make validate-all-b200' pour une validation complète. Lancement de la nouvelle cible...$(NC)"
	@$(MAKE) validate-all-b200

.PHONY: clean
clean: ## Nettoyage fichiers temporaires
	@echo "$(BLUE)🧹 Nettoyage...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -f profile_results.prof
	rm -f benchmark_results.json
	rm -f code_quality_report.json

.PHONY: clean-cache
clean-cache: clean ## Nettoyage + cache modèles
	@echo "$(BLUE)🗑️ Nettoyage cache modèles...$(NC)"
	rm -rf ~/.cache/huggingface/
	rm -rf ~/.cache/torch/
	@if command -v nvidia-smi > /dev/null; then \
		echo "$(BLUE)Nettoyage cache GPU...$(NC)"; \
		$(PYTHON) -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"; \
	fi

# =============================================================================
# RUNPOD B200 SPECIFIC
# =============================================================================

.PHONY: setup-runpod
setup-runpod: ## Configuration spécifique RunPod B200
	@echo "$(BLUE)🚀 Configuration RunPod B200...$(NC)"
	@echo "$(YELLOW)Hardware: runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04$(NC)"
	@echo "$(YELLOW)GPU: 1x B200 (180 GB VRAM), RAM: 188 GB, CPU: 28 vCPU, Disk: 80 GB$(NC)"
	chmod +x setup_runpod.sh
	./setup_runpod.sh
	$(MAKE) validate-b200

.PHONY: monitor-b200
monitor-b200: ## Monitoring B200 en temps réel
	@echo "$(BLUE)📊 Monitoring B200...$(NC)"
	$(PYTHON) monitor.py --b200-mode

.PHONY: optimize-b200
optimize-b200: ## Optimisation modèle pour B200
	@echo "$(BLUE)⚡ Optimisation B200...$(NC)"
	$(PYTHON) -c "from utils.b200_optimizer import get_b200_optimizer; opt = get_b200_optimizer(); print('B200 optimizer configuré')"

# =============================================================================
# DOCUMENTATION
# =============================================================================

.PHONY: docs
docs: ## Generate Sphinx documentation
	@echo "$(BLUE)📚 Génération documentation Sphinx...$(NC)"
	@if ! command -v sphinx-build > /dev/null; then \
		echo "$(YELLOW)Installation Sphinx...$(NC)"; \
		$(PIP) install sphinx furo myst-parser; \
	fi
	sphinx-apidoc -o docs/api/ . --separate --force
	sphinx-build -b html docs/ docs/_build/html/
	@echo "$(GREEN)✅ Documentation générée: docs/_build/html/index.html$(NC)"

.PHONY: docs-clean
docs-clean: ## Clean documentation build
	@echo "$(BLUE)🧹 Nettoyage documentation...$(NC)"
	rm -rf docs/_build/
	rm -rf docs/api/*.rst

.PHONY: docs-serve
docs-serve: docs ## Serve documentation locally
	@echo "$(BLUE)🌐 Serveur documentation local...$(NC)"
	@echo "$(YELLOW)Ouvrez: http://localhost:8000$(NC)"
	cd docs/_build/html && python -m http.server 8000

# =============================================================================
# UTILITAIRES
# =============================================================================

.PHONY: status
status: ## Statut projet
	@echo "$(BLUE)📋 Statut projet$(NC)"
	@echo "$(YELLOW)Fichiers Python:$(NC) $$(find . -name '*.py' -not -path './.venv/*' | wc -l)"
	@echo "$(YELLOW)Lignes de code:$(NC) $$(find . -name '*.py' -not -path './.venv/*' -exec wc -l {} + | tail -1 | awk '{print $$1}')"
	@echo "$(YELLOW)Tests:$(NC) $$(find tests/ -name 'test_*.py' | wc -l) fichiers"
	@echo "$(YELLOW)Dernière validation:$(NC)"
	@if [ -f "code_quality_report.md" ]; then \
		echo "  Rapport qualité: $$(stat -c %y code_quality_report.md)"; \
	else \
		echo "  $(RED)Aucun rapport qualité$(NC)"; \
	fi
	@if [ -f "benchmark_results.json" ]; then \
		echo "  Benchmark: $$(stat -c %y benchmark_results.json)"; \
	else \
		echo "  $(RED)Aucun benchmark$(NC)"; \
	fi

.PHONY: info
info: ## Informations environnement
	@echo "$(BLUE)ℹ️ Informations environnement$(NC)"
	@echo "$(YELLOW)Python:$(NC) $$($(PYTHON) --version)"
	@echo "$(YELLOW)Pip:$(NC) $$($(PIP) --version)"
	@echo "$(YELLOW)Répertoire:$(NC) $$(pwd)"
	@if command -v nvidia-smi > /dev/null; then \
		echo "$(YELLOW)GPU:$(NC)"; \
		nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; \
	fi
	@if [ -f "pyproject.toml" ]; then \
		echo "$(YELLOW)Version projet:$(NC) $$(grep '^version' pyproject.toml | cut -d'"' -f2)"; \
	fi

# =============================================================================
# UTILISATION
# =============================================================================

.PHONY: start
start: ## 🚀 Démarrage guidé. Utilisation: make start URL="..." OUTPUT="..."
	@if [ -z "$(URL)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "$(RED)❌ Erreur: URL et OUTPUT sont requis pour le démarrage guidé."(NC)"; \
		echo "$(YELLOW)👉 Utilisation: make start URL=\"votre_url\" OUTPUT=\"votre_fichier.srt\"$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)🚀 Lancement du traitement guidé...$(NC)"
	$(MAKE) process URL="$(URL)" OUTPUT="$(OUTPUT)"

.PHONY: run
run: start ## Alias pour 'start' - Démarrage guidé

.PHONY: setup-token
setup-token: ## 🔑 Configuration interactive du token HuggingFace
	@echo "$(BLUE)🔑 Configuration Token HuggingFace$(NC)"
	@$(PYTHON) start_simple.py

.PHONY: diagnose
diagnose: ## 🔍 Diagnostic complet du projet
	@echo "$(BLUE)🔍 Diagnostic Complet$(NC)"
	@$(PYTHON) diagnose_all.py


# =============================================================================
# COMMANDES UTILISATEUR FINALES
# =============================================================================

.PHONY: process
process: ## 🎯 Traitement direct (ex: make process URL="..." OUTPUT="...")
	@if [ -z "$(URL)" ]; then \
		echo "$(RED)❌ Erreur: URL manquante$(NC)"; \
		echo "$(YELLOW)Utilisation: make process URL=\"https://youtube.com/...\" OUTPUT=\"sous_titres.srt\"$(NC)"; \
		echo "$(YELLOW)Ou utilisez: make start (interface guidée)$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)🎯 Traitement Direct$(NC)"
	@$(PYTHON) main.py --url "$(URL)" --output "$(OUTPUT)" $(ARGS)

.PHONY: batch
batch: ## 📦 Traitement en lot (ex: make batch LIST="batch.txt" DIR="output")
	@if [ -z "$(LIST)" ]; then \
		echo "$(RED)❌ Erreur: Liste manquante$(NC)"; \
		echo "$(YELLOW)Utilisation: make batch LIST=\"batch.txt\" DIR=\"output\"$(NC)"; \
		echo "$(YELLOW)Ou utilisez: make start (interface guidée)$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)📦 Traitement en Lot$(NC)"
	@$(PYTHON) main.py --batch-list "$(LIST)" --output-dir "$(DIR)" $(ARGS)

# Cible d'aide étendue
.PHONY: help-extended
help-extended: ## 📖 Aide détaillée avec exemples
	@echo "$(BLUE)🚀 EMANET VOXTRAL - Aide Complète$(NC)"
	@echo ""
	@echo "$(GREEN)📋 COMMANDES PRINCIPALES (Utilisateurs finaux):$(NC)"
	@echo "  $(YELLOW)make start$(NC)              - 🚀 Interface guidée interactive (RECOMMANDÉ)"
	@echo "  $(YELLOW)make wizard$(NC)             - 🧙‍♂️ Assistant configuration avancé"
	@echo "  $(YELLOW)make setup-interactive$(NC)  - ⚙️ Configuration système"
	@echo "  $(YELLOW)make validate-interactive$(NC) - 🏥 Diagnostic système"
	@echo "  $(YELLOW)make tutorial$(NC)           - 📚 Guide d'utilisation"
	@echo ""
	@echo "$(GREEN)⚡ COMMANDES DIRECTES (Utilisateurs avancés):$(NC)"
	@echo "  $(YELLOW)make process URL=\"...\" OUTPUT=\"...\"$(NC) - Traitement direct"
	@echo "  $(YELLOW)make batch LIST=\"...\" DIR=\"...\"$(NC)     - Traitement en lot"
	@echo "  $(YELLOW)make demo$(NC)               - 🎬 Démonstration"
	@echo ""
	@echo "$(GREEN)🛠️ COMMANDES DÉVELOPPEUR:$(NC)"
	@echo "  $(YELLOW)make setup$(NC)              - Configuration environnement"
	@echo "  $(YELLOW)make validate$(NC)           - Validation complète"
	@echo "  $(YELLOW)make test$(NC)               - Tests complets"
	@echo "  $(YELLOW)make benchmark$(NC)          - Tests performance"
	@echo ""
	@echo "$(GREEN)📚 EXEMPLES:$(NC)"
	@echo "  $(BLUE)# Interface guidée (débutants)$(NC)"
	@echo "  make start"
	@echo ""
	@echo "  $(BLUE)# Traitement YouTube direct$(NC)"
	@echo "  make process URL=\"https://youtube.com/watch?v=...\" OUTPUT=\"sous_titres.srt\""
	@echo ""
	@echo "  $(BLUE)# Traitement en lot$(NC)"
	@echo "  make batch LIST=\"videos.txt\" DIR=\"resultats\""
	@echo ""
	@echo "  $(BLUE)# Configuration système$(NC)"
	@echo "  make setup-interactive"
	@echo ""
	@echo "$(YELLOW)💡 Pour une première utilisation, commencez par: make start$(NC)"

# Cible par défaut mise à jour
.DEFAULT_GOAL := help-extended