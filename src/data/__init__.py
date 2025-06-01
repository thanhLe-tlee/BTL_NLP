"""
Data Module
===========

Handles dataset loading, preprocessing, and formatting for different NLP tasks.

Classes:
- DatasetConfig: Configuration for datasets
- DataProcessor: Unified data processing interface

Functions:
- load_cnn_dailymail: Load CNN/DailyMail summarization dataset
- load_wmt14: Load WMT14 translation dataset
- load_sst2: Load SST2 sentiment analysis dataset
- format_chat_template: Format data for chat templates
"""

from .dataset_loader import DatasetLoader, DatasetConfig
from .processors import DataProcessor
from .formatters import ChatTemplateFormatter, format_chat_template
from .task_datasets import (
    load_cnn_dailymail,
    load_wmt14,
    load_sst2,
    create_translation_prompt,
    create_summarization_prompt,
    create_classification_prompt
)

__all__ = [
    "DatasetLoader",
    "DatasetConfig",
    "DataProcessor", 
    "ChatTemplateFormatter",
    "format_chat_template",
    "load_cnn_dailymail",
    "load_wmt14", 
    "load_sst2",
    "create_translation_prompt",
    "create_summarization_prompt",
    "create_classification_prompt"
] 