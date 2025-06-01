"""
BTL_NLP - Modular Fine-tuning Framework
========================================

A modular framework for fine-tuning language models on various NLP tasks.

Modules:
- models: Model loading and configuration
- data: Data loading and preprocessing  
- training: Training utilities and loops
- evaluation: Evaluation metrics and visualization
- utils: Helper functions and utilities
"""

__version__ = "1.0.0"
__author__ = "BTL_NLP Team"

from . import models
from . import data
from . import training
from . import evaluation
from . import utils

__all__ = ["models", "data", "training", "evaluation", "utils"] 