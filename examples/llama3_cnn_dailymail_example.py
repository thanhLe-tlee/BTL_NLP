"""
Example: Fine-tuning Llama3 on CNN/DailyMail Summarization
==========================================================

This example demonstrates how to use the modular framework to fine-tune
Llama3 on the CNN/DailyMail summarization dataset.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import load_llama3_model
from src.data import load_cnn_dailymail
from src.training import TrainingConfig, get_training_args, create_trainer
from src.evaluation import compute_rouge_scores, plot_training_curves
from src.utils import setup_environment, get_device_info, create_output_dir

def main():
    """Main training function."""
    print("🦥 BTL_NLP Modular Framework - Llama3 CNN/DailyMail Example")
    print("=" * 60)
    
    # 1. Setup environment
    print("📋 Setting up environment...")
    setup_environment()
    device_info = get_device_info()
    print(f"💻 Device: {device_info}")
    
    # 2. Load model
    print("\n🤖 Loading Llama3 model...")
    model, tokenizer = load_llama3_model(
        max_seq_length=2048,
        lora_r=16,
        use_dora=True
    )
    print("✅ Model loaded successfully!")
    
    # 3. Load and preprocess data
    print("\n📚 Loading CNN/DailyMail dataset...")
    train_dataset = load_cnn_dailymail(
        split="train",
        max_samples=1000  # Reduce for faster training
    )
    eval_dataset = load_cnn_dailymail(
        split="validation", 
        max_samples=100
    )
    print(f"✅ Loaded {len(train_dataset)} training samples and {len(eval_dataset)} eval samples")
    
    # 4. Setup training configuration
    print("\n⚙️ Setting up training configuration...")
    training_config = TrainingConfig(
        output_dir="./results/llama3_cnn_dailymail",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=50,
        eval_steps=50,
        save_steps=50,
        logging_steps=10,
        evaluation_strategy="steps",
        save_strategy="steps"
    )
    
    training_args = get_training_args(training_config)
    print("✅ Training configuration ready!")
    
    # 5. Create trainer
    print("\n🏋️ Creating trainer...")
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    print("✅ Trainer created!")
    
    # 6. Start training
    print("\n🚀 Starting training...")
    trainer.train()
    print("✅ Training completed!")
    
    # 7. Save model
    print("\n💾 Saving model...")
    output_dir = training_config.output_dir
    create_output_dir(output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model saved to {output_dir}")
    
    # 8. Evaluation
    print("\n📊 Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    # 9. Plot training curves
    print("\n📈 Creating training visualizations...")
    plot_training_curves(
        trainer.state.log_history,
        save_path=f"{output_dir}/training_curves.png"
    )
    print("✅ Visualizations saved!")
    
    print("\n🎉 Training pipeline completed successfully!")

if __name__ == "__main__":
    main() 