"""
Training Module
===============

Handles model training, configuration, and optimization.

Classes:
- TrainingConfig: Configuration for training parameters
- Trainer: Unified training interface

Functions:
- setup_training_args: Configure training arguments
- create_trainer: Create trainer instance
- train_model: High-level training function
"""

from .training_config import TrainingConfig, get_training_args
from .trainer import Trainer, create_trainer
from .callbacks import EarlyStoppingCallback, LoggingCallback
from .utils import (
    setup_training_environment,
    save_model,
    load_model,
    estimate_training_time
)

__all__ = [
    "TrainingConfig",
    "get_training_args", 
    "Trainer",
    "create_trainer",
    "EarlyStoppingCallback",
    "LoggingCallback",
    "setup_training_environment",
    "save_model",
    "load_model",
    "estimate_training_time"
] 