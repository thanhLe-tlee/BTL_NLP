"""
Model Loader Module
===================

Unified interface for loading and configuring different model architectures.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from unsloth import FastLanguageModel
import torch


@dataclass
class ModelConfig:
    """Configuration class for model parameters."""
    model_name: str
    max_seq_length: int = 2048
    dtype: Optional[torch.dtype] = None
    load_in_4bit: bool = True
    
    # LoRA/DoRA parameters
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: list = None
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


class ModelLoader:
    """Unified model loader for different architectures."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def load_base_model(self) -> Tuple[Any, Any]:
        """Load the base model and tokenizer."""
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model_name,
                max_seq_length=self.config.max_seq_length,
                dtype=self.config.dtype,
                load_in_4bit=self.config.load_in_4bit,
            )
            self.model = model
            self.tokenizer = tokenizer
            return model, tokenizer
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.config.model_name}: {str(e)}")
    
    def setup_peft(self) -> Any:
        """Setup PEFT (LoRA/DoRA) configuration."""
        if self.model is None:
            raise ValueError("Base model must be loaded first. Call load_base_model().")
        
        try:
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.config.lora_r,
                target_modules=self.config.target_modules,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                use_gradient_checkpointing=self.config.use_gradient_checkpointing,
                random_state=self.config.random_state,
                use_dora=self.config.use_dora,
                use_rslora=self.config.use_rslora,
                loftq_config=None,
            )
            return self.model
        except Exception as e:
            raise RuntimeError(f"Failed to setup PEFT: {str(e)}")
    
    def load_complete_model(self) -> Tuple[Any, Any]:
        """Load base model and setup PEFT in one call."""
        self.load_base_model()
        self.setup_peft()
        return self.model, self.tokenizer
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and parameters."""
        if self.model is None:
            return {"status": "No model loaded"}
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            "model_name": self.config.model_name,
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
            "trainable_percentage": 100.0 * trainable_params / total_params,
            "lora_config": {
                "r": self.config.lora_r,
                "alpha": self.config.lora_alpha,
                "dropout": self.config.lora_dropout,
                "use_dora": self.config.use_dora,
            }
        } 