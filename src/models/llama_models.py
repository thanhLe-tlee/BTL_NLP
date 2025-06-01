"""
Llama Models Module
==================

Specialized functions for loading Llama model variants.
"""

from typing import Tuple, Any
from .model_loader import ModelLoader, ModelConfig


def load_llama3_model(
    max_seq_length: int = 2048,
    lora_r: int = 16,
    use_dora: bool = True,
    **kwargs
) -> Tuple[Any, Any]:
    """
    Load Llama-3-8B-Instruct model with optimized settings.
    
    Args:
        max_seq_length: Maximum sequence length
        lora_r: LoRA rank
        use_dora: Whether to use DoRA instead of LoRA
        **kwargs: Additional configuration parameters
    
    Returns:
        Tuple of (model, tokenizer)
    """
    config = ModelConfig(
        model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",
        max_seq_length=max_seq_length,
        lora_r=lora_r,
        use_dora=use_dora,
        **kwargs
    )
    
    loader = ModelLoader(config)
    return loader.load_complete_model()


def load_llama3_1b_model(
    max_seq_length: int = 2048,
    lora_r: int = 16,
    use_dora: bool = False,
    **kwargs
) -> Tuple[Any, Any]:
    """
    Load Llama-3.1-1B model with optimized settings.
    
    Args:
        max_seq_length: Maximum sequence length
        lora_r: LoRA rank
        use_dora: Whether to use DoRA instead of LoRA
        **kwargs: Additional configuration parameters
    
    Returns:
        Tuple of (model, tokenizer)
    """
    config = ModelConfig(
        model_name="unsloth/Llama-3.1-1B-Instruct-bnb-4bit",
        max_seq_length=max_seq_length,
        lora_r=lora_r,
        use_dora=use_dora,
        **kwargs
    )
    
    loader = ModelLoader(config)
    return loader.load_complete_model()


def get_llama_config(model_variant: str = "llama3-8b") -> ModelConfig:
    """
    Get predefined configuration for Llama model variants.
    
    Args:
        model_variant: Model variant ("llama3-8b" or "llama3-1b")
    
    Returns:
        ModelConfig object
    """
    configs = {
        "llama3-8b": ModelConfig(
            model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",
            max_seq_length=2048,
            lora_r=16,
            use_dora=True,
        ),
        "llama3-1b": ModelConfig(
            model_name="unsloth/Llama-3.1-1B-Instruct-bnb-4bit",
            max_seq_length=2048,
            lora_r=16,
            use_dora=False,
        )
    }
    
    if model_variant not in configs:
        raise ValueError(f"Unknown model variant: {model_variant}. Available: {list(configs.keys())}")
    
    return configs[model_variant] 