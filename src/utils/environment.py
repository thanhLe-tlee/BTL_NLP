"""
Environment Utilities
====================

Functions for setting up training environment and managing devices.
"""

import os
import subprocess
import sys
import torch
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def setup_environment():
    """Setup the training environment."""
    print("🔧 Setting up training environment...")
    
    # Install dependencies if needed
    install_dependencies()
    
    # Setup CUDA if available
    setup_cuda()
    
    # Print environment info
    print_environment_info()


def install_dependencies():
    """Install required dependencies."""
    try:
        # Check if in Colab
        if "COLAB_" in "".join(os.environ.keys()):
            print("📦 Installing Colab dependencies...")
            # Install Unsloth and dependencies for Colab
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "--no-deps", "bitsandbytes", "accelerate", 
                "xformers==0.0.29.post3", "peft", "trl==0.15.2", 
                "triton", "cut_cross_entropy", "unsloth_zoo"
            ], check=True, capture_output=True)
            
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "sentencepiece", "protobuf", "datasets>=3.4.1", 
                "huggingface_hub", "hf_transfer"
            ], check=True, capture_output=True)
            
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "--no-deps", "unsloth"
            ], check=True, capture_output=True)
        else:
            print("📦 Installing standard dependencies...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "unsloth"
            ], check=True, capture_output=True)
            
        print("✅ Dependencies installed successfully!")
        
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to install some dependencies: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during dependency installation: {e}")


def setup_cuda():
    """Setup CUDA environment."""
    if torch.cuda.is_available():
        print(f"🚀 CUDA available! Using GPU: {torch.cuda.get_device_name()}")
        print(f"🔧 CUDA version: {torch.version.cuda}")
        
        # Set memory fraction to avoid OOM
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.95)
            
        # Clear cache
        torch.cuda.empty_cache()
    else:
        print("⚠️ CUDA not available, using CPU")


def get_device_info() -> Dict[str, Any]:
    """Get device and system information."""
    info = {
        "platform": sys.platform,
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(),
        })
        
        # Memory info
        for i in range(torch.cuda.device_count()):
            memory_info = torch.cuda.get_device_properties(i)
            info[f"gpu_{i}_memory_total"] = f"{memory_info.total_memory / 1024**3:.2f} GB"
    
    return info


def check_gpu_memory() -> Dict[str, float]:
    """Check GPU memory usage."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    memory_info = {}
    for i in range(torch.cuda.device_count()):
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
        memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        
        memory_info[f"gpu_{i}"] = {
            "allocated_gb": round(memory_allocated, 2),
            "reserved_gb": round(memory_reserved, 2),
            "total_gb": round(memory_total, 2),
            "free_gb": round(memory_total - memory_reserved, 2),
            "utilization": round(memory_reserved / memory_total * 100, 1)
        }
    
    return memory_info


def print_environment_info():
    """Print environment information."""
    info = get_device_info()
    
    print("\n💻 Environment Information:")
    print(f"Platform: {info['platform']}")
    print(f"Python: {info['python_version'].split()[0]}")
    print(f"PyTorch: {info['pytorch_version']}")
    
    if info["cuda_available"]:
        print(f"CUDA: {info['cuda_version']}")
        print(f"GPU: {info['device_name']}")
        
        # Memory info
        memory_info = check_gpu_memory()
        for gpu_id, mem_info in memory_info.items():
            if isinstance(mem_info, dict):
                print(f"{gpu_id.upper()}: {mem_info['free_gb']:.1f}GB free / {mem_info['total_gb']:.1f}GB total")
    else:
        print("CUDA: Not available")


def set_environment_variables():
    """Set useful environment variables."""
    # Set HF transfer for faster downloads
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    # Set tokenizers parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Set CUDA launch blocking for debugging
    if os.getenv("DEBUG", "").lower() in ("1", "true"):
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def optimize_for_memory():
    """Optimize settings for memory usage."""
    if torch.cuda.is_available():
        # Set memory growth
        torch.cuda.empty_cache()
        
        # Set deterministic algorithms if needed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 