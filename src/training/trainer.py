"""
Trainer Module
==============

Unified training interface using TRL SFTTrainer.
"""

from typing import Any, Optional, Union
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel


class Trainer:
    """Unified trainer wrapper for SFTTrainer."""
    
    def __init__(self, 
                 model: Any,
                 tokenizer: Any,
                 args: TrainingArguments,
                 train_dataset: Dataset,
                 eval_dataset: Optional[Dataset] = None,
                 max_seq_length: int = 2048,
                 dataset_text_field: str = "text",
                 dataset_num_proc: int = 2,
                 packing: bool = False):
        """
        Initialize trainer.
        
        Args:
            model: The model to train
            tokenizer: The tokenizer
            args: Training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            max_seq_length: Maximum sequence length
            dataset_text_field: Text field name in dataset
            dataset_num_proc: Number of processes for dataset processing
            packing: Whether to use packing
        """
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Create SFTTrainer
        self.trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field=dataset_text_field,
            max_seq_length=max_seq_length,
            dataset_num_proc=dataset_num_proc,
            packing=packing,
            args=args,
        )
    
    def train(self):
        """Start training."""
        return self.trainer.train()
    
    def evaluate(self, eval_dataset: Optional[Dataset] = None):
        """Evaluate the model."""
        if eval_dataset is not None:
            return self.trainer.evaluate(eval_dataset)
        return self.trainer.evaluate()
    
    def save_model(self, output_dir: str):
        """Save the trained model."""
        self.trainer.save_model(output_dir)
    
    def predict(self, test_dataset: Dataset):
        """Make predictions on test dataset."""
        return self.trainer.predict(test_dataset)
    
    @property
    def state(self):
        """Get trainer state for logging history."""
        return self.trainer.state


def create_trainer(
    model: Any,
    tokenizer: Any,
    args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    **kwargs
) -> Trainer:
    """
    Create a trainer instance.
    
    Args:
        model: The model to train
        tokenizer: The tokenizer
        args: Training arguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        **kwargs: Additional arguments for trainer
    
    Returns:
        Trainer instance
    """
    return Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        **kwargs
    ) 