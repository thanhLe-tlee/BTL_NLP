#!/usr/bin/env python3
"""
Training Script: Llama3 8B Fine-tuning on CNN/DailyMail Summarization
=====================================================================

This script fine-tunes Llama3-8B on CNN/DailyMail dataset for summarization.
Equivalent to llama3_fine_tuned_cnn_dailymail.ipynb notebook.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import load_llama3_model
from src.data import load_cnn_dailymail
from src.training import TrainingConfig, get_training_args, create_trainer
from src.evaluation import compute_rouge_scores
from src.utils import setup_logging, set_seed

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def main():
    """Main training function for Llama3 CNN/DailyMail."""
    
    print("🦥 BTL_NLP - Llama3 8B Fine-tuning on CNN/DailyMail")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(3407)
    
    # 1. Setup and configuration
    print("📋 Setting up configuration...")
    output_dir = "./results/llama3_8b_cnn_dailymail"
    
    # Training configuration optimized for summarization
    config = TrainingConfig(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,  # Lower LR for summarization
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=50,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        logging_strategy="steps",
        logging_steps=10,
        bf16=True,
        fp16=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        max_grad_norm=0.3,
        group_by_length=True,
        optim="adamw_8bit",
        seed=3407,
    )
    
    print("✅ Configuration ready!")
    
    # 2. Load model
    print("\n🤖 Loading Llama3-8B model...")
    try:
        model, tokenizer = load_llama3_model(
            max_seq_length=2048,
            lora_r=16,
            use_dora=True  # Use DoRA for better performance
        )
        print("✅ Model loaded successfully!")
        
        # Print model info
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Trainable parameters: {trainable_params:,}")
        print(f"📊 Total parameters: {total_params:,}")
        print(f"📊 Trainable %: {100 * trainable_params / total_params:.2f}%")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # 3. Load datasets
    print("\n📚 Loading CNN/DailyMail dataset...")
    try:
        # Load training data (reduced for demo)
        train_dataset = load_cnn_dailymail(
            split="train",
            max_samples=5000,  # Increase for full training
            version="3.0.0"
        )
        
        # Load validation data
        eval_dataset = load_cnn_dailymail(
            split="validation",
            max_samples=500,
            version="3.0.0"
        )
        
        print(f"✅ Loaded {len(train_dataset)} training samples")
        print(f"✅ Loaded {len(eval_dataset)} validation samples")
        
        # Preview sample
        print("\n📖 Sample data:")
        sample = train_dataset[0]
        print(f"Text length: {len(sample['text'])} characters")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # 4. Setup trainer
    print("\n🏋️ Setting up trainer...")
    try:
        training_args = get_training_args(config)
        trainer = create_trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            max_seq_length=2048,
            dataset_text_field="text",
            dataset_num_proc=2,
            packing=False,
        )
        print("✅ Trainer setup complete!")
        
    except Exception as e:
        logger.error(f"Failed to setup trainer: {e}")
        return
    
    # 5. Start training
    print("\n🚀 Starting training...")
    print(f"💾 Model will be saved to: {output_dir}")
    
    try:
        # Train the model
        result = trainer.train()
        print("✅ Training completed!")
        
        # Print training results
        print(f"\n📈 Training Results:")
        print(f"Final train loss: {result.training_loss:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return
    
    # 6. Save model
    print("\n💾 Saving model...")
    try:
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"✅ Model saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
    
    # 7. Final evaluation
    print("\n📊 Running final evaluation...")
    try:
        eval_results = trainer.evaluate()
        print("✅ Evaluation completed!")
        
        print(f"\n📈 Evaluation Results:")
        for key, value in eval_results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
                
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
    
    print("\n🎉 Llama3-8B CNN/DailyMail fine-tuning completed!")
    print(f"📁 Results saved in: {output_dir}")

if __name__ == "__main__":
    main() 