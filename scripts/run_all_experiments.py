#!/usr/bin/env python3
"""
Master Training Script: Run All Experiments
===========================================

This script runs all training experiments equivalent to the original notebooks.
You can run all experiments or select specific ones.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Define all available experiments
EXPERIMENTS = {
    # Llama3 8B experiments
    "llama3_cnn": {
        "script": "train_llama3_cnn_dailymail.py",
        "description": "Llama3-8B fine-tuning on CNN/DailyMail summarization",
        "estimated_time": "2-3 hours",
        "memory_required": "16GB VRAM"
    },
    "llama3_wmt14": {
        "script": "train_llama3_wmt14.py", 
        "description": "Llama3-8B fine-tuning on WMT14 translation",
        "estimated_time": "3-4 hours",
        "memory_required": "16GB VRAM"
    },
    "llama3_sst2": {
        "script": "train_llama3_sst2.py",
        "description": "Llama3-8B fine-tuning on SST2 classification", 
        "estimated_time": "1-2 hours",
        "memory_required": "12GB VRAM"
    },
    
    # Phi-4 experiments
    "phi4_cnn": {
        "script": "train_phi4_cnn_dailymail.py",
        "description": "Phi-4 Mini fine-tuning on CNN/DailyMail summarization",
        "estimated_time": "1.5-2.5 hours",
        "memory_required": "12GB VRAM"
    },
    "phi4_wmt14": {
        "script": "train_phi4_wmt14.py",
        "description": "Phi-4 Mini fine-tuning on WMT14 translation", 
        "estimated_time": "2-3 hours",
        "memory_required": "12GB VRAM"
    },
    "phi4_sst2": {
        "script": "train_phi4_sst2.py",
        "description": "Phi-4 Mini fine-tuning on SST2 classification",
        "estimated_time": "1-1.5 hours", 
        "memory_required": "8GB VRAM"
    },
    
    # Llama3 1B experiments  
    "llama3_1b_wmt14": {
        "script": "train_llama3_1b_wmt14.py",
        "description": "Llama3-1B fine-tuning on WMT14 translation",
        "estimated_time": "1-2 hours",
        "memory_required": "8GB VRAM"
    },
    "llama3_1b_cnn": {
        "script": "train_llama3_1b_cnn_dailymail.py", 
        "description": "Llama3-1B fine-tuning on CNN/DailyMail summarization",
        "estimated_time": "1-1.5 hours",
        "memory_required": "8GB VRAM"
    },
    "llama3_1b_sst2": {
        "script": "train_llama3_1b_sst2.py",
        "description": "Llama3-1B fine-tuning on SST2 classification",
        "estimated_time": "0.5-1 hour",
        "memory_required": "6GB VRAM"
    }
}


def print_banner():
    """Print welcome banner."""
    print("=" * 80)
    print("🦥 BTL_NLP - Modular Fine-tuning Framework")
    print("Master Experiment Runner")
    print("=" * 80)


def list_experiments():
    """List all available experiments."""
    print("\n📋 Available Experiments:")
    print("-" * 80)
    
    for exp_id, exp_info in EXPERIMENTS.items():
        print(f"🔹 {exp_id}")
        print(f"   Description: {exp_info['description']}")
        print(f"   Estimated time: {exp_info['estimated_time']}")
        print(f"   Memory required: {exp_info['memory_required']}")
        print()


def run_experiment(exp_id: str, script_dir: Path) -> bool:
    """Run a single experiment."""
    if exp_id not in EXPERIMENTS:
        logger.error(f"Unknown experiment: {exp_id}")
        return False
    
    exp_info = EXPERIMENTS[exp_id]
    script_path = script_dir / exp_info["script"]
    
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False
    
    print(f"\n🚀 Starting experiment: {exp_id}")
    print(f"📄 Script: {exp_info['script']}")
    print(f"📝 Description: {exp_info['description']}")
    print(f"⏱️ Estimated time: {exp_info['estimated_time']}")
    print("-" * 60)
    
    try:
        # Run the script
        result = subprocess.run([
            sys.executable, str(script_path)
        ], check=True, cwd=script_dir.parent)
        
        print(f"✅ Experiment {exp_id} completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Experiment {exp_id} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        logger.warning(f"⚠️ Experiment {exp_id} interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error in experiment {exp_id}: {e}")
        return False


def run_experiments(exp_ids: List[str], script_dir: Path) -> Dict[str, bool]:
    """Run multiple experiments."""
    results = {}
    
    print(f"\n🎯 Running {len(exp_ids)} experiment(s)...")
    
    for i, exp_id in enumerate(exp_ids, 1):
        print(f"\n{'='*20} Experiment {i}/{len(exp_ids)} {'='*20}")
        results[exp_id] = run_experiment(exp_id, script_dir)
        
        if not results[exp_id]:
            print(f"\n❌ Stopping due to failed experiment: {exp_id}")
            break
    
    return results


def print_summary(results: Dict[str, bool]):
    """Print experiment results summary."""
    print("\n" + "=" * 80)
    print("📊 EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    
    successful = [exp_id for exp_id, success in results.items() if success]
    failed = [exp_id for exp_id, success in results.items() if not success]
    
    print(f"\n✅ Successful experiments ({len(successful)}):")
    for exp_id in successful:
        print(f"   🟢 {exp_id}: {EXPERIMENTS[exp_id]['description']}")
    
    if failed:
        print(f"\n❌ Failed experiments ({len(failed)}):")
        for exp_id in failed:
            print(f"   🔴 {exp_id}: {EXPERIMENTS[exp_id]['description']}")
    
    print(f"\n📈 Success rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run BTL_NLP training experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_experiments.py --list                    # List all experiments
  python run_all_experiments.py --all                     # Run all experiments
  python run_all_experiments.py llama3_cnn llama3_sst2    # Run specific experiments
  python run_all_experiments.py --model llama3            # Run all Llama3 experiments
  python run_all_experiments.py --task classification     # Run all classification experiments
        """
    )
    
    parser.add_argument("experiments", nargs="*", help="Experiment IDs to run")
    parser.add_argument("--list", action="store_true", help="List all available experiments")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--model", choices=["llama3", "phi4", "llama3_1b"], help="Run experiments for specific model")
    parser.add_argument("--task", choices=["summarization", "translation", "classification"], help="Run experiments for specific task")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue running even if some experiments fail")
    
    args = parser.parse_args()
    
    print_banner()
    
    # List experiments if requested
    if args.list:
        list_experiments()
        return
    
    # Determine which experiments to run
    exp_ids = []
    
    if args.all:
        exp_ids = list(EXPERIMENTS.keys())
    elif args.model:
        if args.model == "llama3":
            exp_ids = [k for k in EXPERIMENTS.keys() if k.startswith("llama3") and not k.startswith("llama3_1b")]
        elif args.model == "llama3_1b":
            exp_ids = [k for k in EXPERIMENTS.keys() if k.startswith("llama3_1b")]
        elif args.model == "phi4":
            exp_ids = [k for k in EXPERIMENTS.keys() if k.startswith("phi4")]
    elif args.task:
        if args.task == "summarization":
            exp_ids = [k for k in EXPERIMENTS.keys() if "cnn" in k]
        elif args.task == "translation":
            exp_ids = [k for k in EXPERIMENTS.keys() if "wmt14" in k]
        elif args.task == "classification":
            exp_ids = [k for k in EXPERIMENTS.keys() if "sst2" in k]
    elif args.experiments:
        exp_ids = args.experiments
    else:
        print("❌ No experiments specified. Use --list to see available experiments.")
        return
    
    # Validate experiment IDs
    invalid_ids = [exp_id for exp_id in exp_ids if exp_id not in EXPERIMENTS]
    if invalid_ids:
        print(f"❌ Invalid experiment IDs: {invalid_ids}")
        print("Use --list to see available experiments.")
        return
    
    if not exp_ids:
        print("❌ No experiments to run.")
        return
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Run experiments
    results = run_experiments(exp_ids, script_dir)
    
    # Print summary
    print_summary(results)
    
    # Exit with appropriate code
    failed_count = len([r for r in results.values() if not r])
    if failed_count > 0:
        print(f"\n⚠️ {failed_count} experiment(s) failed.")
        sys.exit(1)
    else:
        print(f"\n🎉 All {len(results)} experiment(s) completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main() 