.PHONY: help install list run-all clean
.DEFAULT_GOAL := help

# Colors for output
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
BLUE := \033[34m
RESET := \033[0m

help: ## Show this help message
	@echo "$(BLUE)BTL_NLP - Modular Fine-tuning Framework$(RESET)"
	@echo "=========================================="
	@echo ""
	@echo "$(GREEN)Available commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup
install: ## Install dependencies
	@echo "$(YELLOW)Installing dependencies...$(RESET)"
	pip install -r requirements.txt
	@echo "$(GREEN)✅ Dependencies installed successfully!$(RESET)"

##@ Training Scripts
list: ## List all available experiments
	@echo "$(YELLOW)Listing available experiments...$(RESET)"
	cd scripts && python run_all_experiments.py --list

run-all: ## Run all experiments
	@echo "$(YELLOW)Running all experiments...$(RESET)"
	cd scripts && python run_all_experiments.py --all

##@ Individual Model Experiments
llama3: ## Run all Llama3-8B experiments
	@echo "$(YELLOW)Running Llama3-8B experiments...$(RESET)"
	cd scripts && python run_all_experiments.py --model llama3

llama3-1b: ## Run all Llama3-1B experiments  
	@echo "$(YELLOW)Running Llama3-1B experiments...$(RESET)"
	cd scripts && python run_all_experiments.py --model llama3_1b

phi4: ## Run all Phi-4 experiments
	@echo "$(YELLOW)Running Phi-4 experiments...$(RESET)"
	cd scripts && python run_all_experiments.py --model phi4

##@ Task-specific Experiments
summarization: ## Run all summarization experiments
	@echo "$(YELLOW)Running summarization experiments...$(RESET)"
	cd scripts && python run_all_experiments.py --task summarization

translation: ## Run all translation experiments
	@echo "$(YELLOW)Running translation experiments...$(RESET)"
	cd scripts && python run_all_experiments.py --task translation

classification: ## Run all classification experiments
	@echo "$(YELLOW)Running classification experiments...$(RESET)"
	cd scripts && python run_all_experiments.py --task classification

##@ Individual Experiments
llama3-cnn: ## Train Llama3-8B on CNN/DailyMail
	@echo "$(YELLOW)Running Llama3-8B CNN/DailyMail experiment...$(RESET)"
	cd scripts && python train_llama3_cnn_dailymail.py

llama3-wmt14: ## Train Llama3-8B on WMT14
	@echo "$(YELLOW)Running Llama3-8B WMT14 experiment...$(RESET)"
	cd scripts && python train_llama3_wmt14.py

llama3-sst2: ## Train Llama3-8B on SST2
	@echo "$(YELLOW)Running Llama3-8B SST2 experiment...$(RESET)"
	cd scripts && python train_llama3_sst2.py

phi4-cnn: ## Train Phi-4 on CNN/DailyMail
	@echo "$(YELLOW)Running Phi-4 CNN/DailyMail experiment...$(RESET)"
	cd scripts && python train_phi4_cnn_dailymail.py

phi4-wmt14: ## Train Phi-4 on WMT14
	@echo "$(YELLOW)Running Phi-4 WMT14 experiment...$(RESET)"
	cd scripts && python train_phi4_wmt14.py

phi4-sst2: ## Train Phi-4 on SST2
	@echo "$(YELLOW)Running Phi-4 SST2 experiment...$(RESET)"
	cd scripts && python train_phi4_sst2.py

##@ Utilities
check-gpu: ## Check GPU status and memory
	@echo "$(YELLOW)Checking GPU status...$(RESET)"
	@python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory // 1024**3}GB)') for i in range(torch.cuda.device_count())]"

check-env: ## Check environment setup
	@echo "$(YELLOW)Checking environment...$(RESET)"
	@python -c "import sys; print(f'Python: {sys.version}'); import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda if torch.cuda.is_available() else \"Not available\"}')"

clean: ## Clean up results and cache
	@echo "$(YELLOW)Cleaning up...$(RESET)"
	rm -rf results/*/
	rm -rf logs/
	rm -rf cache/
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)✅ Cleanup completed!$(RESET)"

##@ Development
format: ## Format code with black
	@echo "$(YELLOW)Formatting code...$(RESET)"
	black src/ scripts/ examples/
	@echo "$(GREEN)✅ Code formatted!$(RESET)"

lint: ## Lint code with flake8
	@echo "$(YELLOW)Linting code...$(RESET)"
	flake8 src/ scripts/ examples/ --max-line-length=100
	@echo "$(GREEN)✅ Code linted!$(RESET)"

test: ## Run tests
	@echo "$(YELLOW)Running tests...$(RESET)"
	pytest tests/ -v
	@echo "$(GREEN)✅ Tests completed!$(RESET)"

##@ Examples
quick-test: ## Run a quick test with Llama3-1B on SST2 (fastest experiment)
	@echo "$(YELLOW)Running quick test (Llama3-1B SST2)...$(RESET)"
	cd scripts && python run_all_experiments.py llama3_1b_sst2

demo: ## Run demo experiments (one from each category)
	@echo "$(YELLOW)Running demo experiments...$(RESET)"
	cd scripts && python run_all_experiments.py llama3_1b_sst2 phi4_sst2 llama3_sst2 