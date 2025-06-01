"""
Training Configuration Module
============================

Configuration classes and utilities for training setup.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from transformers import TrainingArguments


@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    output_dir: str = "./results"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    
    # Optimization
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    warmup_steps: int = 50
    
    # Evaluation and saving
    evaluation_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 3
    
    # Logging
    logging_strategy: str = "steps"
    logging_steps: int = 10
    report_to: Optional[str] = None
    
    # Memory optimization
    fp16: bool = False
    bf16: bool = True
    dataloader_pin_memory: bool = False
    remove_unused_columns: bool = False
    
    # Advanced settings
    max_grad_norm: float = 0.3
    group_by_length: bool = True
    optim: str = "adamw_8bit"
    seed: int = 3407


def get_training_args(config: TrainingConfig) -> TrainingArguments:
    """
    Convert TrainingConfig to Transformers TrainingArguments.
    
    Args:
        config: TrainingConfig object
    
    Returns:
        TrainingArguments object
    """
    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_steps,
        
        evaluation_strategy=config.evaluation_strategy,
        eval_steps=config.eval_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        
        logging_strategy=config.logging_strategy,
        logging_steps=config.logging_steps,
        report_to=config.report_to,
        
        fp16=config.fp16,
        bf16=config.bf16,
        dataloader_pin_memory=config.dataloader_pin_memory,
        remove_unused_columns=config.remove_unused_columns,
        
        max_grad_norm=config.max_grad_norm,
        group_by_length=config.group_by_length,
        optim=config.optim,
        seed=config.seed,
    )


def get_recommended_config(
    task_type: str = "general",
    model_size: str = "medium",
    dataset_size: str = "medium"
) -> TrainingConfig:
    """
    Get recommended training configuration for different scenarios.
    
    Args:
        task_type: Type of task ("translation", "summarization", "classification")
        model_size: Size of model ("small", "medium", "large")
        dataset_size: Size of dataset ("small", "medium", "large")
    
    Returns:
        TrainingConfig object with recommended settings
    """
    # Base configuration
    config = TrainingConfig()
    
    # Task-specific adjustments
    if task_type == "translation":
        config.num_train_epochs = 3
        config.learning_rate = 2e-4
        config.warmup_steps = 100
    elif task_type == "summarization":
        config.num_train_epochs = 2
        config.learning_rate = 1e-4
        config.warmup_steps = 50
    elif task_type == "classification":
        config.num_train_epochs = 5
        config.learning_rate = 3e-4
        config.warmup_steps = 30
    
    # Model size adjustments
    if model_size == "small":
        config.per_device_train_batch_size = 4
        config.gradient_accumulation_steps = 2
    elif model_size == "large":
        config.per_device_train_batch_size = 1
        config.gradient_accumulation_steps = 8
    
    # Dataset size adjustments
    if dataset_size == "small":
        config.eval_steps = 25
        config.save_steps = 25
        config.logging_steps = 5
    elif dataset_size == "large":
        config.eval_steps = 100
        config.save_steps = 100
        config.logging_steps = 20
    
    return config


def get_memory_optimized_config(available_vram_gb: int = 16) -> TrainingConfig:
    """
    Get memory-optimized training configuration based on available VRAM.
    
    Args:
        available_vram_gb: Available VRAM in GB
    
    Returns:
        TrainingConfig object optimized for memory usage
    """
    config = TrainingConfig()
    
    if available_vram_gb <= 8:
        # Very memory-constrained
        config.per_device_train_batch_size = 1
        config.gradient_accumulation_steps = 8
        config.fp16 = True
        config.bf16 = False
        config.dataloader_pin_memory = False
    elif available_vram_gb <= 16:
        # Moderately memory-constrained
        config.per_device_train_batch_size = 2
        config.gradient_accumulation_steps = 4
        config.bf16 = True
        config.dataloader_pin_memory = False
    else:
        # Less memory-constrained
        config.per_device_train_batch_size = 4
        config.gradient_accumulation_steps = 2
        config.bf16 = True
        config.dataloader_pin_memory = True
    
    return config


def create_colab_config() -> TrainingConfig:
    """
    Create optimized configuration for Google Colab environment.
    
    Returns:
        TrainingConfig object optimized for Colab
    """
    config = TrainingConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        bf16=True,
        fp16=False,
        dataloader_pin_memory=False,
        logging_steps=10,
        eval_steps=50,
        save_steps=50,
        optim="adamw_8bit",
        max_grad_norm=0.3,
        warmup_steps=50,
        learning_rate=2e-4,
        weight_decay=0.01,
        lr_scheduler_type="linear",
    )
    
    return config 