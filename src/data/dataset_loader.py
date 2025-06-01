"""
Dataset Loader Module
====================

Unified interface for loading and preprocessing datasets.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
from datasets import load_dataset, Dataset
import random


@dataclass
class DatasetConfig:
    """Configuration class for dataset parameters."""
    name: str
    subset: Optional[str] = None
    split: str = "train"
    streaming: bool = False
    trust_remote_code: bool = False
    
    # Preprocessing parameters
    max_samples: Optional[int] = None
    shuffle: bool = True
    seed: int = 42
    
    # Task-specific parameters
    source_column: Optional[str] = None
    target_column: Optional[str] = None
    text_column: Optional[str] = None
    label_column: Optional[str] = None


class DatasetLoader:
    """Unified dataset loader for different NLP tasks."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.dataset = None
        self.processed_dataset = None
    
    def load_raw_dataset(self) -> Dataset:
        """Load the raw dataset from Hugging Face."""
        try:
            if self.config.subset:
                dataset = load_dataset(
                    self.config.name,
                    self.config.subset,
                    split=self.config.split,
                    streaming=self.config.streaming,
                    trust_remote_code=self.config.trust_remote_code
                )
            else:
                dataset = load_dataset(
                    self.config.name,
                    split=self.config.split,
                    streaming=self.config.streaming,
                    trust_remote_code=self.config.trust_remote_code
                )
            
            self.dataset = dataset
            return dataset
            
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {self.config.name}: {str(e)}")
    
    def sample_dataset(self, dataset: Dataset) -> Dataset:
        """Sample a subset of the dataset if specified."""
        if self.config.max_samples is None:
            return dataset
        
        # Convert to list if streaming
        if self.config.streaming:
            dataset = Dataset.from_list(list(dataset.take(self.config.max_samples)))
        else:
            # Shuffle if requested
            if self.config.shuffle:
                dataset = dataset.shuffle(seed=self.config.seed)
            
            # Take subset
            if len(dataset) > self.config.max_samples:
                dataset = dataset.select(range(self.config.max_samples))
        
        return dataset
    
    def apply_preprocessing(self, 
                          dataset: Dataset, 
                          preprocessing_fn: Callable) -> Dataset:
        """Apply preprocessing function to the dataset."""
        try:
            processed = dataset.map(
                preprocessing_fn,
                batched=False,
                remove_columns=dataset.column_names
            )
            self.processed_dataset = processed
            return processed
        except Exception as e:
            raise RuntimeError(f"Failed to preprocess dataset: {str(e)}")
    
    def load_and_preprocess(self, preprocessing_fn: Callable) -> Dataset:
        """Load dataset and apply preprocessing in one call."""
        dataset = self.load_raw_dataset()
        dataset = self.sample_dataset(dataset)
        return self.apply_preprocessing(dataset, preprocessing_fn)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the loaded dataset."""
        if self.dataset is None:
            return {"status": "No dataset loaded"}
        
        info = {
            "name": self.config.name,
            "subset": self.config.subset,
            "split": self.config.split,
            "columns": self.dataset.column_names,
        }
        
        # Add size information
        if not self.config.streaming:
            info["size"] = len(self.dataset)
            if self.processed_dataset:
                info["processed_size"] = len(self.processed_dataset)
        
        return info
    
    def preview_samples(self, n: int = 3) -> Dict[str, Any]:
        """Preview sample data from the dataset."""
        if self.dataset is None:
            return {"error": "No dataset loaded"}
        
        if self.config.streaming:
            samples = list(self.dataset.take(n))
        else:
            samples = self.dataset.select(range(min(n, len(self.dataset))))
            samples = [samples[i] for i in range(len(samples))]
        
        return {
            "samples": samples,
            "count": len(samples)
        } 