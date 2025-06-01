"""
Evaluation Module
================

Handles model evaluation, metrics computation, and result visualization.

Classes:
- MetricsCalculator: Unified metrics calculation
- Evaluator: Model evaluation interface

Functions:
- compute_rouge_scores: ROUGE metrics for summarization
- compute_bleu_scores: BLEU metrics for translation  
- compute_accuracy: Accuracy metrics for classification
- plot_training_curves: Visualize training progress
"""

from .metrics import (
    MetricsCalculator,
    compute_rouge_scores,
    compute_bleu_scores,
    compute_accuracy,
    compute_meteor_scores
)
from .evaluator import Evaluator, EvaluationConfig
from .visualization import (
    plot_training_curves,
    plot_loss_curves,
    plot_metrics_comparison,
    create_evaluation_report
)

__all__ = [
    "MetricsCalculator",
    "compute_rouge_scores",
    "compute_bleu_scores", 
    "compute_accuracy",
    "compute_meteor_scores",
    "Evaluator",
    "EvaluationConfig",
    "plot_training_curves",
    "plot_loss_curves",
    "plot_metrics_comparison",
    "create_evaluation_report"
] 