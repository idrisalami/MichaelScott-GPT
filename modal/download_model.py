"""
Download Model from Modal Volume
=================================
Helper script to download trained model weights from Modal to your local machine.

Usage:
    python download_model.py
"""

import subprocess
import os
import json
from pathlib import Path


def run_command(cmd):
    """Run a shell command and return output"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(result.stdout)
    return True


def download_from_modal():
    """Download model files from Modal Volume"""
    
    # Create local models directory
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("📥 DOWNLOADING MODEL FROM MODAL")
    print("=" * 80)
    
    # List available files
    print("\n1. Checking available files on Modal Volume...")
    run_command("modal volume ls model-checkpoints checkpoints/")
    
    # Download best model weights
    print("\n2. Downloading best model weights...")
    success = run_command(
        f"modal volume get model-checkpoints checkpoints/best_model.pt {models_dir}/weights_tiktoken.pt"
    )
    
    if success:
        print("✅ Downloaded best model weights")
    
    # Download config
    print("\n3. Downloading model configuration...")
    success = run_command(
        f"modal volume get model-checkpoints checkpoints/config.json {models_dir}/config_tiktoken.json"
    )
    
    if success:
        print("✅ Downloaded model configuration")
        
        # Display config
        config_path = models_dir / "config_tiktoken.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            print("\n📋 Model Configuration:")
            print(json.dumps(config, indent=2))
    
    # Download final model (optional)
    print("\n4. Downloading final model (last epoch)...")
    run_command(
        f"modal volume get model-checkpoints checkpoints/final_model.pt {models_dir}/final_model.pt"
    )
    
    print("\n" + "=" * 80)
    print("✅ DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"\nModel files saved to: {models_dir.absolute()}")
    print("\nYou can now use these files for inference!")
    print("\nNext steps:")
    print("  1. Load the model in your notebook")
    print("  2. Use the generate_reply() function for inference")
    print("  3. Enjoy your Michael Scott chatbot! 🎉")


if __name__ == "__main__":
    download_from_modal()

