# Migration Guide: From Notebooks to Modular Framework

This guide helps you transition from using individual Jupyter notebooks to the new modular framework structure.

## 📖 Overview

The modular framework reorganizes your existing notebook code into reusable, maintainable modules while preserving all functionality. Instead of duplicated code across notebooks, you now have:

- **Unified interfaces** for model loading, data processing, and training
- **Consistent configuration** across all experiments
- **Reusable components** that reduce code duplication
- **Better organization** with clear separation of concerns

## 🔄 Key Changes

### Before (Notebooks)
```python
# Each notebook had its own setup code
%%capture
import os
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
# ... more installation code

from unsloth import FastLanguageModel
import torch

# Model loading with duplicated parameters
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
```

### After (Modular)
```python
# Simple imports
from src.models import load_llama3_model
from src.data import load_cnn_dailymail
from src.training import TrainingConfig, create_trainer

# One-line model loading with optimized defaults
model, tokenizer = load_llama3_model(use_dora=True)
```

## 🗂️ Module Mapping

### 1. Model Loading (`src/models/`)

**Old way (in each notebook):**
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_dora = True,
)
```

**New way:**
```python
from src.models import load_llama3_model

# For Llama3 8B
model, tokenizer = load_llama3_model(use_dora=True)

# For Llama3 1B
model, tokenizer = load_llama3_1b_model()

# For Phi-4
from src.models import load_phi4_model
model, tokenizer = load_phi4_model()
```

### 2. Data Loading (`src/data/`)

**Old way:**
```python
from datasets import load_dataset

dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")

def formatting_prompts_func(examples):
    # Custom formatting code...
    pass

dataset = dataset.map(formatting_prompts_func, batched=True)
```

**New way:**
```python
from src.data import load_cnn_dailymail, load_wmt14, load_sst2

# Automatically preprocessed and formatted
train_dataset = load_cnn_dailymail(split="train", max_samples=1000)
eval_dataset = load_wmt14(split="validation", language_pair="de-en")
```

### 3. Training (`src/training/`)

**Old way:**
```python
from transformers import TrainingArguments
from trl import SFTTrainer

training_args = TrainingArguments(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    max_steps = 60,
    learning_rate = 2e-4,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 1,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    output_dir = "outputs",
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = training_args,
)
```

**New way:**
```python
from src.training import TrainingConfig, get_training_args, create_trainer

config = TrainingConfig(
    output_dir="./results/my_model",
    num_train_epochs=2,
    learning_rate=2e-4
)

trainer = create_trainer(
    model=model,
    tokenizer=tokenizer,
    args=get_training_args(config),
    train_dataset=train_dataset
)
```

### 4. Evaluation (`src/evaluation/`)

**Old way:**
```python
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt

# Manual evaluation code...
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
```

**New way:**
```python
from src.evaluation import compute_rouge_scores, plot_training_curves

# Automatic metrics computation
rouge_scores = compute_rouge_scores(predictions, references)
plot_training_curves(trainer.state.log_history, save_path="curves.png")
```

## 📝 Migration Steps

### Step 1: Install the Framework
```bash
pip install -r requirements.txt
```

### Step 2: Identify Your Notebook Pattern

**For Summarization (CNN/DailyMail):**
- Use `src.models.load_llama3_model()` or `src.models.load_phi4_model()`
- Use `src.data.load_cnn_dailymail()`
- Use summarization-specific configuration

**For Translation (WMT14):**
- Use `src.data.load_wmt14()`
- Use translation-specific configuration

**For Classification (SST2):**
- Use `src.data.load_sst2()` 
- Use classification-specific configuration

### Step 3: Convert Your Notebook

1. **Replace imports:**
   ```python
   # Old
   from unsloth import FastLanguageModel
   from datasets import load_dataset
   from transformers import TrainingArguments
   
   # New
   from src.models import load_llama3_model
   from src.data import load_cnn_dailymail
   from src.training import TrainingConfig, create_trainer
   ```

2. **Simplify model loading:**
   ```python
   # Replace 20+ lines of model setup with:
   model, tokenizer = load_llama3_model(use_dora=True)
   ```

3. **Use dataset loaders:**
   ```python
   # Replace dataset loading and preprocessing with:
   train_dataset = load_cnn_dailymail(split="train", max_samples=1000)
   ```

4. **Streamline training:**
   ```python
   # Replace training arguments setup with:
   config = TrainingConfig(output_dir="./results", num_train_epochs=2)
   trainer = create_trainer(model, tokenizer, get_training_args(config), train_dataset)
   ```

### Step 4: Run Your Converted Code

```python
# Your complete training script is now ~10 lines instead of 100+
from src.models import load_llama3_model
from src.data import load_cnn_dailymail
from src.training import TrainingConfig, get_training_args, create_trainer

model, tokenizer = load_llama3_model(use_dora=True)
train_dataset = load_cnn_dailymail(split="train", max_samples=1000)

config = TrainingConfig(output_dir="./results", num_train_epochs=2)
trainer = create_trainer(model, tokenizer, get_training_args(config), train_dataset)

trainer.train()
```

## 🎯 Benefits of Migration

### 1. **Reduced Code Duplication**
- Model loading: **50+ lines** → **1 line**
- Data preprocessing: **30+ lines** → **1 line** 
- Training setup: **20+ lines** → **3 lines**

### 2. **Better Configuration Management**
```yaml
# config.yaml - One place for all settings
models:
  llama3_8b:
    lora:
      r: 16
      use_dora: true

training:
  default:
    learning_rate: 2e-4
    num_train_epochs: 3
```

### 3. **Consistent Results**
- Same preprocessing across all experiments
- Standardized evaluation metrics
- Reproducible training configurations

### 4. **Easier Experimentation**
```python
# Try different models with same data/training:
for model_loader in [load_llama3_model, load_phi4_model]:
    model, tokenizer = model_loader()
    trainer = create_trainer(model, tokenizer, args, dataset)
    trainer.train()
```

## 🔍 Example Conversions

### CNN/DailyMail Notebook → Module

**Before (notebook):** 100+ lines of setup code

**After (module):**
```python
from src.models import load_llama3_model
from src.data import load_cnn_dailymail
from src.training import TrainingConfig, get_training_args, create_trainer

model, tokenizer = load_llama3_model()
dataset = load_cnn_dailymail(max_samples=1000)
config = TrainingConfig(output_dir="./results/cnn")
trainer = create_trainer(model, tokenizer, get_training_args(config), dataset)
trainer.train()
```

### WMT14 Translation → Module

```python
from src.models import load_phi4_model
from src.data import load_wmt14
from src.training import get_recommended_config

model, tokenizer = load_phi4_model()
dataset = load_wmt14(language_pair="de-en")
config = get_recommended_config("translation", "medium", "large")
trainer = create_trainer(model, tokenizer, get_training_args(config), dataset)
trainer.train()
```

## 🚀 Advanced Usage

### Custom Configurations
```python
from src.training import TrainingConfig

# Memory-optimized for limited GPU
config = get_memory_optimized_config(available_vram_gb=8)

# Task-specific optimization
config = get_recommended_config(
    task_type="translation",
    model_size="large", 
    dataset_size="medium"
)
```

### Multiple Experiments
```python
# Run experiments across different models and tasks
experiments = [
    ("llama3", "cnn_dailymail", "summarization"),
    ("phi4", "wmt14", "translation"),
    ("llama3_1b", "sst2", "classification")
]

for model_name, dataset_name, task in experiments:
    model, tokenizer = load_model_by_name(model_name)
    dataset = load_dataset_by_name(dataset_name)
    config = get_recommended_config(task)
    
    trainer = create_trainer(model, tokenizer, get_training_args(config), dataset)
    trainer.train()
```

## 📞 Support

If you encounter issues during migration:

1. **Check the examples** in `examples/` directory
2. **Review configuration** in `config.yaml`
3. **Use recommended configs** for your specific use case
4. **Refer to module documentation** in each `__init__.py` file

The modular framework preserves all the functionality of your original notebooks while making the code more maintainable, reusable, and easier to experiment with! 