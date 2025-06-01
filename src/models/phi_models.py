"""
Phi Models Module
================

Specialized functions for loading Phi model variants.
"""

from typing import Tuple, Any
from .model_loader import ModelLoader, ModelConfig


def load_phi4_model(
    max_seq_length: int = 2048,
    lora_r: int = 16,
    use_dora: bool = False,
    **kwargs
) -> Tuple[Any, Any]:
    """
    Load Phi-4 model with optimized settings.
    
    Args:
        max_seq_length: Maximum sequence length
        lora_r: LoRA rank
        use_dora: Whether to use DoRA instead of LoRA
        **kwargs: Additional configuration parameters
    
    Returns:
        Tuple of (model, tokenizer)
    """
    config = ModelConfig(
        model_name="unsloth/phi-4-mini-bnb-4bit",
        max_seq_length=max_seq_length,
        lora_r=lora_r,
        use_dora=use_dora,
        # Phi-4 specific target modules
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        **kwargs
    )
    
    loader = ModelLoader(config)
    return loader.load_complete_model()


def get_phi_config(model_variant: str = "phi4-mini") -> ModelConfig:
    """
    Get predefined configuration for Phi model variants.
    
    Args:
        model_variant: Model variant ("phi4-mini")
    
    Returns:
        ModelConfig object
    """
    configs = {
        "phi4-mini": ModelConfig(
            model_name="unsloth/phi-4-mini-bnb-4bit",
            max_seq_length=2048,
            lora_r=16,
            use_dora=False,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
        )
    }
    
    if model_variant not in configs:
        raise ValueError(f"Unknown model variant: {model_variant}. Available: {list(configs.keys())}")
    
    return configs[model_variant] 