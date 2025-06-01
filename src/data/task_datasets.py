"""
Task-Specific Dataset Module
===========================

Functions for loading and preprocessing task-specific datasets.
"""

from typing import Dict, Any, Optional
from datasets import Dataset
from .dataset_loader import DatasetLoader, DatasetConfig


def load_cnn_dailymail(
    split: str = "train",
    max_samples: Optional[int] = None,
    version: str = "3.0.0"
) -> Dataset:
    """
    Load CNN/DailyMail summarization dataset.
    
    Args:
        split: Dataset split ("train", "validation", "test")
        max_samples: Maximum number of samples to load
        version: Dataset version
    
    Returns:
        Processed dataset
    """
    config = DatasetConfig(
        name="cnn_dailymail",
        subset=version,
        split=split,
        max_samples=max_samples,
        source_column="article",
        target_column="highlights"
    )
    
    loader = DatasetLoader(config)
    
    def preprocess_summarization(example):
        return {
            "text": create_summarization_prompt(
                article=example["article"],
                summary=example["highlights"]
            )
        }
    
    return loader.load_and_preprocess(preprocess_summarization)


def load_wmt14(
    split: str = "train",
    max_samples: Optional[int] = None,
    language_pair: str = "de-en"
) -> Dataset:
    """
    Load WMT14 translation dataset.
    
    Args:
        split: Dataset split ("train", "validation", "test")
        max_samples: Maximum number of samples to load
        language_pair: Language pair (e.g., "de-en")
    
    Returns:
        Processed dataset
    """
    config = DatasetConfig(
        name="wmt14",
        subset=language_pair,
        split=split,
        max_samples=max_samples
    )
    
    loader = DatasetLoader(config)
    
    def preprocess_translation(example):
        source_lang, target_lang = language_pair.split("-")
        source_text = example["translation"][source_lang]
        target_text = example["translation"][target_lang]
        
        return {
            "text": create_translation_prompt(
                source_text=source_text,
                target_text=target_text,
                source_lang=source_lang,
                target_lang=target_lang
            )
        }
    
    return loader.load_and_preprocess(preprocess_translation)


def load_sst2(
    split: str = "train",
    max_samples: Optional[int] = None
) -> Dataset:
    """
    Load SST2 sentiment analysis dataset.
    
    Args:
        split: Dataset split ("train", "validation", "test")
        max_samples: Maximum number of samples to load
    
    Returns:
        Processed dataset
    """
    config = DatasetConfig(
        name="glue",
        subset="sst2",
        split=split,
        max_samples=max_samples,
        text_column="sentence",
        label_column="label"
    )
    
    loader = DatasetLoader(config)
    
    def preprocess_classification(example):
        label_map = {0: "negative", 1: "positive"}
        sentiment = label_map[example["label"]]
        
        return {
            "text": create_classification_prompt(
                text=example["sentence"],
                label=sentiment
            )
        }
    
    return loader.load_and_preprocess(preprocess_classification)


def create_summarization_prompt(article: str, summary: str) -> str:
    """Create prompt for summarization task."""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that summarizes news articles. Provide a concise and accurate summary of the given article.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Please summarize the following article:

{article}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{summary}<|eot_id|>"""


def create_translation_prompt(
    source_text: str, 
    target_text: str,
    source_lang: str = "de",
    target_lang: str = "en"
) -> str:
    """Create prompt for translation task."""
    lang_names = {
        "de": "German",
        "en": "English", 
        "fr": "French",
        "es": "Spanish"
    }
    
    source_name = lang_names.get(source_lang, source_lang)
    target_name = lang_names.get(target_lang, target_lang)
    
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a professional translator. Translate the given {source_name} text to {target_name} accurately and naturally.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Translate the following {source_name} text to {target_name}:

{source_text}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{target_text}<|eot_id|>"""


def create_classification_prompt(text: str, label: str) -> str:
    """Create prompt for sentiment classification task."""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a sentiment analysis expert. Analyze the sentiment of the given text and classify it as either "positive" or "negative".

<|eot_id|><|start_header_id|>user<|end_header_id|>

Analyze the sentiment of the following text:

{text}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{label}<|eot_id|>"""


def get_dataset_config(dataset_name: str, **kwargs) -> DatasetConfig:
    """
    Get predefined configuration for common datasets.
    
    Args:
        dataset_name: Name of the dataset
        **kwargs: Additional configuration parameters
    
    Returns:
        DatasetConfig object
    """
    configs = {
        "cnn_dailymail": DatasetConfig(
            name="cnn_dailymail",
            subset="3.0.0",
            source_column="article",
            target_column="highlights"
        ),
        "wmt14_de_en": DatasetConfig(
            name="wmt14",
            subset="de-en"
        ),
        "sst2": DatasetConfig(
            name="glue",
            subset="sst2",
            text_column="sentence",
            label_column="label"
        )
    }
    
    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(configs.keys())}")
    
    config = configs[dataset_name]
    
    # Update with provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config 