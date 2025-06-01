# NLP Assignment (CO3086) - Transformer Models and Fine-Tuning

A comprehensive modular framework for fine-tuning language models on various NLP tasks including translation, summarization, and classification.

## 🌟 Features

- **Modular Design**: Clean separation of concerns with dedicated modules for models, data, training, and evaluation
- **Multiple Model Support**: Llama3, Llama3.1 (8B/1B), Phi-4 Mini with optimized configurations
- **Multiple Tasks**: Translation (WMT14), Summarization (CNN/DailyMail), Classification (SST2)
- **Memory Efficient**: LoRA/DoRA adapters with 4-bit quantization using Unsloth
- **Comprehensive Evaluation**: ROUGE, BLEU, METEOR, accuracy metrics with visualization
- **Easy Configuration**: YAML-based configuration with sensible defaults

## 📁 Project Structure

```
BTL_NLP/
├── src/                          # Main source code
│   ├── models/                   # Model loading and configuration
│   │   ├── __init__.py
│   │   ├── model_loader.py       # Unified model loader
│   │   ├── llama_models.py       # Llama-specific functions
│   │   ├── phi_models.py         # Phi-specific functions
│   │   └── lora_config.py        # LoRA/DoRA configuration
│   ├── data/                     # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── dataset_loader.py     # Unified dataset loader
│   │   ├── task_datasets.py      # Task-specific datasets
│   │   ├── processors.py         # Data processors
│   │   └── formatters.py         # Data formatters
│   ├── training/                 # Training utilities
│   │   ├── __init__.py
│   │   ├── training_config.py    # Training configurations
│   │   ├── trainer.py            # Training interface
│   │   ├── callbacks.py          # Training callbacks
│   │   └── utils.py              # Training utilities
│   ├── evaluation/               # Evaluation and metrics
│   │   ├── __init__.py
│   │   ├── metrics.py            # Metrics calculation
│   │   ├── evaluator.py          # Evaluation interface
│   │   └── visualization.py      # Result visualization
│   └── utils/                    # Common utilities
│       ├── __init__.py
│       ├── environment.py        # Environment setup
│       ├── helpers.py            # Helper functions
│       └── logging.py            # Logging utilities
├── examples/                     # Example scripts
│   ├── llama3_cnn_dailymail_example.py
│   ├── phi4_wmt14_example.py
│   └── llama3_sst2_example.py
├── notebooks/                    # Original Jupyter notebooks
├── config.yaml                   # Main configuration file
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd BTL_NLP

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

```python
from src.models import load_llama3_model
from src.data import load_cnn_dailymail
from src.training import TrainingConfig, get_training_args, create_trainer

# Load model
model, tokenizer = load_llama3_model(use_dora=True)

# Load data
train_dataset = load_cnn_dailymail(split="train", max_samples=1000)

# Configure training
config = TrainingConfig(
    output_dir="./results/my_model",
    num_train_epochs=2,
    learning_rate=2e-4
)

# Train
trainer = create_trainer(model, tokenizer, get_training_args(config), train_dataset)
trainer.train()
```

### 3. Run Examples

```bash
# Fine-tune Llama3 on CNN/DailyMail
python examples/llama3_cnn_dailymail_example.py

# Fine-tune Phi-4 on WMT14
python examples/phi4_wmt14_example.py

# Fine-tune Llama3 on SST2
python examples/llama3_sst2_example.py
```

## 🎯 Supported Tasks

### 📝 Summarization (CNN/DailyMail)
- **Models**: Llama3-8B, Llama3-1B, Phi-4
- **Metrics**: ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum
- **Dataset**: CNN/DailyMail v3.0.0

### 🌐 Translation (WMT14)
- **Models**: Llama3-8B, Llama3-1B, Phi-4
- **Metrics**: BLEU, METEOR, chrF
- **Dataset**: WMT14 German-English

### 😊 Classification (SST2)
- **Models**: Llama3-8B, Llama3-1B, Phi-4
- **Metrics**: Accuracy, F1, Precision, Recall
- **Dataset**: Stanford Sentiment Treebank v2

## ⚙️ Configuration

The framework uses a hierarchical configuration system with `config.yaml` as the main configuration file. You can override settings programmatically or through environment variables.

### Model Configuration
```yaml
models:
  llama3_8b:
    name: "unsloth/llama-3-8b-Instruct-bnb-4bit"
    max_seq_length: 2048
    lora:
      r: 16
      alpha: 16
      use_dora: true
```

### Training Configuration
```yaml
training:
  default:
    num_train_epochs: 3
    per_device_train_batch_size: 2
    learning_rate: 2.0e-4
    bf16: true
```

## 📊 Evaluation & Visualization

The framework provides comprehensive evaluation capabilities:

- **Automated Metrics**: Task-specific metrics calculation
- **Training Curves**: Loss and learning rate visualization
- **Model Comparison**: Side-by-side comparison of different models
- **Export Results**: Save results in various formats (JSON, CSV, plots)

## 🔧 Advanced Features

### Memory Optimization
- 4-bit quantization with BitsAndBytes
- LoRA/DoRA adapters for parameter-efficient fine-tuning
- Gradient checkpointing for reduced memory usage
- Automatic batch size optimization

### Flexible Training
- Multiple optimization strategies
- Custom learning rate schedules
- Early stopping with validation monitoring
- Distributed training support

### Extensibility
- Easy addition of new models
- Custom dataset processors
- Pluggable evaluation metrics
- Custom training callbacks

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- [Hugging Face](https://huggingface.co/) for transformers and datasets
- [Meta](https://ai.meta.com/) for Llama models
- [Microsoft](https://www.microsoft.com/) for Phi models

---

**Built with ❤️ by the BTL_NLP Team**
