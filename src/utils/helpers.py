"""
Helper Utilities
================

Common utility functions used across the framework.
"""

import os
import json
import yaml
import random
import numpy as np
import torch
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union


def set_seed(seed: int = 3407):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🎲 Random seed set to: {seed}")


def create_output_dir(output_dir: str) -> Path:
    """Create output directory if it doesn't exist."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory created: {path.absolute()}")
    return path


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def estimate_training_time(
    num_samples: int, 
    batch_size: int, 
    num_epochs: int,
    seconds_per_batch: float = 5.0
) -> str:
    """Estimate training time based on dataset size and configuration."""
    batches_per_epoch = num_samples // batch_size
    total_batches = batches_per_epoch * num_epochs
    total_seconds = total_batches * seconds_per_batch
    
    return format_time(total_seconds)


def save_config(config: Dict[str, Any], filepath: str):
    """Save configuration to file."""
    filepath = Path(filepath)
    
    # Create directory if needed
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save based on file extension
    if filepath.suffix.lower() in ['.yaml', '.yml']:
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    elif filepath.suffix.lower() == '.json':
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported config format: {filepath.suffix}")
    
    print(f"💾 Configuration saved to: {filepath}")


def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from file."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    # Load based on file extension
    if filepath.suffix.lower() in ['.yaml', '.yml']:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
    elif filepath.suffix.lower() == '.json':
        with open(filepath, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {filepath.suffix}")
    
    return config or {}


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries."""
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage."""
    import psutil
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "ram_used_gb": memory_info.rss / 1024**3,
        "ram_available_gb": psutil.virtual_memory().available / 1024**3,
        "ram_total_gb": psutil.virtual_memory().total / 1024**3,
    }


def format_number(num: Union[int, float], precision: int = 2) -> str:
    """Format large numbers with appropriate suffixes."""
    if num >= 1e9:
        return f"{num/1e9:.{precision}f}B"
    elif num >= 1e6:
        return f"{num/1e6:.{precision}f}M"
    elif num >= 1e3:
        return f"{num/1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def create_run_name(model_name: str, dataset_name: str, timestamp: bool = True) -> str:
    """Create a unique run name for the experiment."""
    model_short = model_name.split("/")[-1] if "/" in model_name else model_name
    model_short = model_short.replace("-", "_").replace(".", "_")
    
    dataset_short = dataset_name.replace("-", "_").replace(".", "_")
    
    run_name = f"{model_short}_{dataset_short}"
    
    if timestamp:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name += f"_{now}"
    
    return run_name


def clean_filename(filename: str) -> str:
    """Clean filename by removing invalid characters."""
    import re
    # Remove invalid characters
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    return cleaned


def get_model_size(model) -> Dict[str, Any]:
    """Get model size information."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "trainable_percentage": 100.0 * trainable_params / total_params if total_params > 0 else 0,
        "trainable_formatted": format_number(trainable_params),
        "total_formatted": format_number(total_params),
    }


def calculate_batch_size_for_memory(
    available_memory_gb: float,
    sequence_length: int,
    model_size_params: int,
    safety_factor: float = 0.8
) -> int:
    """Calculate optimal batch size based on available memory."""
    # Rough estimation formula
    memory_per_token = model_size_params * 4  # 4 bytes per parameter (fp32)
    memory_per_sequence = memory_per_token * sequence_length
    memory_per_batch_item = memory_per_sequence * 4  # Forward + backward + gradients + optimizer
    
    available_bytes = available_memory_gb * 1024**3 * safety_factor
    optimal_batch_size = int(available_bytes / memory_per_batch_item)
    
    return max(1, optimal_batch_size)


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """Validate that config contains all required keys."""
    missing_keys = []
    
    for key in required_keys:
        if '.' in key:
            # Handle nested keys
            keys = key.split('.')
            current = config
            for k in keys:
                if k not in current:
                    missing_keys.append(key)
                    break
                current = current[k]
        else:
            if key not in config:
                missing_keys.append(key)
    
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")
    
    return True 