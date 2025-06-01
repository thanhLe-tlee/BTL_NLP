"""
LoRA Configuration Module
========================

Utilities for configuring LoRA and DoRA adapters.
"""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class LoRAConfig:
    """Configuration for LoRA/DoRA adapters."""
    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: List[str] = None
    use_dora: bool = False
    use_rslora: bool = False
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


def setup_lora_config(
    r: int = 16,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: List[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Setup LoRA configuration.
    
    Args:
        r: LoRA rank
        alpha: LoRA alpha scaling factor
        dropout: LoRA dropout rate
        target_modules: Target modules for LoRA
        **kwargs: Additional parameters
    
    Returns:
        Dictionary of LoRA configuration
    """
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    
    config = {
        "r": r,
        "target_modules": target_modules,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": 3407,
        "use_dora": False,
        "use_rslora": False,
        "loftq_config": None,
    }
    
    config.update(kwargs)
    return config


def setup_dora_config(
    r: int = 16,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: List[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Setup DoRA configuration.
    
    Args:
        r: DoRA rank
        alpha: DoRA alpha scaling factor
        dropout: DoRA dropout rate
        target_modules: Target modules for DoRA
        **kwargs: Additional parameters
    
    Returns:
        Dictionary of DoRA configuration
    """
    config = setup_lora_config(r, alpha, dropout, target_modules, **kwargs)
    config["use_dora"] = True
    return config


def get_recommended_config(model_name: str, task_type: str = "general") -> LoRAConfig:
    """
    Get recommended LoRA/DoRA configuration for specific models and tasks.
    
    Args:
        model_name: Model name (e.g., "llama3-8b", "phi4-mini")
        task_type: Task type ("translation", "summarization", "classification")
    
    Returns:
        LoRAConfig object with recommended settings
    """
    # Base configurations for different models
    base_configs = {
        "llama3-8b": {
            "r": 16,
            "alpha": 16,
            "use_dora": True,
        },
        "llama3-1b": {
            "r": 16,
            "alpha": 16,
            "use_dora": False,
        },
        "phi4-mini": {
            "r": 16,
            "alpha": 16,
            "use_dora": False,
        }
    }
    
    # Task-specific adjustments
    task_adjustments = {
        "translation": {"r": 32, "alpha": 32},
        "summarization": {"r": 16, "alpha": 16},
        "classification": {"r": 8, "alpha": 16},
    }
    
    # Get base config
    if model_name not in base_configs:
        raise ValueError(f"Unknown model: {model_name}")
    
    config = base_configs[model_name].copy()
    
    # Apply task adjustments
    if task_type in task_adjustments:
        config.update(task_adjustments[task_type])
    
    return LoRAConfig(**config) 