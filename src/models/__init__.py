"""
Models Module
=============

Handles model loading, configuration, and setup for different architectures.

Classes:
- ModelConfig: Configuration class for models
- ModelLoader: Unified model loading interface

Functions:
- load_llama3_model: Load Llama3 models
- load_phi4_model: Load Phi-4 models
- setup_lora_config: Configure LoRA/DoRA adapters
"""

from .model_loader import ModelLoader, ModelConfig
from .llama_models import load_llama3_model, load_llama3_1b_model
from .phi_models import load_phi4_model
from .lora_config import setup_lora_config, setup_dora_config

__all__ = [
    "ModelLoader",
    "ModelConfig", 
    "load_llama3_model",
    "load_llama3_1b_model",
    "load_phi4_model",
    "setup_lora_config",
    "setup_dora_config"
] 