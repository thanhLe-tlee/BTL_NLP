"""
Data Processors Module
======================

Unified data processing utilities.
"""

from typing import Dict, Any, List, Optional
from datasets import Dataset


class DataProcessor:
    """Unified data processor for all tasks."""
    
    def __init__(self, task_type: str):
        self.task_type = task_type
    
    def process_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single example based on task type."""
        if self.task_type == "summarization":
            return self._process_summarization(example)
        elif self.task_type == "translation":
            return self._process_translation(example)
        elif self.task_type == "classification":
            return self._process_classification(example)
        else:
            return example
    
    def _process_summarization(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Process summarization example."""
        article = example.get("article", "")
        highlights = example.get("highlights", "")
        
        # Clean and format text
        article = self._clean_text(article)
        highlights = self._clean_text(highlights)
        
        return {
            "article": article,
            "highlights": highlights,
            "source_text": article,
            "target_text": highlights
        }
    
    def _process_translation(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Process translation example."""
        if "translation" in example:
            translation_dict = example["translation"]
            # Assuming de-en translation
            source_text = translation_dict.get("de", "")
            target_text = translation_dict.get("en", "")
        else:
            source_text = example.get("source_text", "")
            target_text = example.get("target_text", "")
        
        return {
            "source_text": self._clean_text(source_text),
            "target_text": self._clean_text(target_text),
            "source_lang": "de",
            "target_lang": "en"
        }
    
    def _process_classification(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Process classification example."""
        text = example.get("sentence", example.get("text", ""))
        label = example.get("label", 0)
        
        # Convert numeric label to text
        if isinstance(label, (int, float)):
            label_map = {0: "negative", 1: "positive"}
            label_text = label_map.get(int(label), "unknown")
        else:
            label_text = str(label).lower()
        
        return {
            "text": self._clean_text(text),
            "sentence": self._clean_text(text),
            "label": int(label) if isinstance(label, (int, float)) else 0,
            "label_text": label_text
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            text = str(text)
        
        # Basic cleaning
        text = text.strip()
        text = " ".join(text.split())  # Normalize whitespace
        
        return text
    
    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Process entire dataset."""
        return dataset.map(
            self.process_example,
            batched=False,
            desc=f"Processing {self.task_type} dataset"
        )


def create_processor(task_type: str) -> DataProcessor:
    """Create a data processor for specific task type."""
    return DataProcessor(task_type) 