"""
Modal GPU Training Script
==============================================================
This script uses the existing transformer architecture and training code
to train on Modal's remote GPUs.

Usage:
    modal run train_modal_integrated.py
"""

import modal
import json
from pathlib import Path
from utils.data_loader import get_local_data

# Create a Modal app
app = modal.App("michael-scott-transformer-v2")

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.0",
        "tiktoken==0.5.1",
        "tqdm==4.66.1",
        "numpy==1.24.3",
    )

    # Add your local modules to the image
    .add_local_python_source("gpt_tokenizers")
    .add_local_python_source("training")
    .add_local_python_source("utils")
)

# Create a Modal Volume to store model checkpoints
volume = modal.Volume.from_name("model-checkpoints", create_if_missing=True)

# Mount paths
VOLUME_PATH = "/vol"
CHECKPOINT_DIR = f"{VOLUME_PATH}/checkpoints"

SRC_PATH = "data/src.txt"
TGT_PATH = "data/tgt.txt"

@app.function(
    image=image,
    gpu="T4",  # Options: "T4", "A10G", "A100-40GB", "A100-80GB", "H100"
    volumes={VOLUME_PATH: volume},
    timeout=3600 * 6,  # 6 hours timeout
)
def train_model_remote(
    src_lines: list[str],
    tgt_lines: list[str],
    epochs: int = 50,
    batch_size: int = 32,
    d_model: int = 512,
    n_layers: int = 6,
    n_heads: int = 8,
    d_ff: int = 2048,
    dropout: float = 0.1,
    learning_rate: float = 1e-4,
    max_src_len: int = 256,
    max_tgt_len: int = 128,
):
    """
    Train the transformer model on Modal's GPU with my existing architecture.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from tqdm import tqdm
    import math
    import os
    import numpy as np
    
    print("=" * 80)
    print("🚀 STARTING TRAINING ON MODAL GPU")
    print("=" * 80)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_name}")
    print(f"Training samples: {len(src_lines)}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print(f"Model config: d_model={d_model}, layers={n_layers}, heads={n_heads}")
    print("=" * 80 + "\n")
    
    # ============================================================================
    # TOKENIZER
    # ============================================================================
    
    from gpt_tokenizers.tiktoken import TiktokenTokenizer
    from gpt_tokenizers.tiktoken import VocabInfo

    vocab = VocabInfo(pad_id=0, bos_id=1, eos_id=2, shift=3)

    tok = TiktokenTokenizer(vocab=vocab)
    enc_src = tok.encode_batch(src_lines)
    enc_tgt = tok.encode_batch(tgt_lines)
    vocab_size = tok.vocab_size
    
    print(f"✅ Tokenizer initialized - Vocab size: {vocab_size}")
    
    # Analyze sequence lengths
    src_lens = [len(x) for x in enc_src]
    tgt_lens = [len(x) for x in enc_tgt]
    print(f"Source lengths - Mean: {np.mean(src_lens):.1f}, P95: {np.percentile(src_lens, 95):.0f}")
    print(f"Target lengths - Mean: {np.mean(tgt_lens):.1f}, P95: {np.percentile(tgt_lens, 95):.0f}\n")
    
    # ============================================================================
    # DATASET
    # ============================================================================
    
    from utils.data_loader import OfficeSeq2Seq

    # Split data
    N = len(enc_src)
    split = int(0.9 * N)
    
    train_ds = OfficeSeq2Seq(enc_src[:split], enc_tgt[:split], max_src_len, max_tgt_len, vocab.pad_id)
    val_ds = OfficeSeq2Seq(enc_src[split:], enc_tgt[split:], max_src_len, max_tgt_len, vocab.pad_id)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    
    print(f"✅ Data loaded - Train: {len(train_ds)}, Val: {len(val_ds)}\n")
    
    # ============================================================================
    # MODEL
    # ============================================================================
    
    from utils.transformer import build_transformer
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        src_seq_len=max_src_len,
        tgt_seq_len=max_tgt_len - 1,
        d_model=d_model,
        N=n_layers,
        h=n_heads,
        dropout=dropout,
        d_ff=d_ff
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model built - Total params: {total_params:,} (Trainable: {trainable_params:,})\n")
    
    # ============================================================================
    # TRAINING SETUP (Using TrainConfig and Trainer from train.py)
    # ============================================================================
    
    from dataclasses import dataclass
    from typing import Optional
    from utils.masks import Masks
    from training.train import TrainConfig, Trainer

    # Configure trainer with Modal-specific settings
    cfg = TrainConfig(
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        clip_grad_norm=1.0,
        pad_id=vocab.pad_id,
        epochs=epochs,
        amp=False,
        log_progress=True,
        label_smoothing=0.1,
        use_scheduler=True,
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_frequency=10,
        volume_commit_fn=lambda: volume.commit()
    )
    
    # ============================================================================
    # TRAINING (Using Trainer class)
    # ============================================================================
    
    print("🎯 Starting training...\n")
    
    # Initialize trainer
    trainer = Trainer(model, device=device, cfg=cfg)
    
    # Train the model
    history, best_val_loss = trainer.fit(train_dl, val_dl)
    
    # Save final model and config
    print("\n💾 Saving final model and configuration...")
    
    final_checkpoint_path = f"{CHECKPOINT_DIR}/final_model.pt"
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.opt.state_dict(),
        'train_loss': history[-1]['train'],
        'val_loss': history[-1]['val'],
        'perplexity': history[-1]['ppl'],
    }, final_checkpoint_path)
    
    config = {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "dropout": dropout,
        "max_src_len": max_src_len,
        "max_tgt_len": max_tgt_len,
        "src_seq_len": max_src_len,
        "tgt_seq_len": max_tgt_len - 1,
        "specials": {
            "PAD": vocab.pad_id,
            "BOS": vocab.bos_id,
            "EOS": vocab.eos_id,
            "SHIFT": vocab.shift
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "best_val_loss": best_val_loss,
        }
    }
    
    config_path = f"{CHECKPOINT_DIR}/config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    volume.commit()
    
    print("=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final perplexity: {history[-1]['ppl']:.2f}")
    print(f"Total epochs: {epochs}")
    print("=" * 80 + "\n")
    
    return {
        "history": history,
        "config": config,
        "best_val_loss": best_val_loss,
        "final_perplexity": history[-1]['ppl'],
    }


@app.local_entrypoint()
def main(
    epochs: int = 50,
    batch_size: int = 32,
    d_model: int = 512,
    n_layers: int = 6,
    n_heads: int = 8,
    d_ff: int = 2048,
    learning_rate: float = 1e-4,
):
    """
    Main entry point - trains model on Modal GPU and downloads weights.
    
    Usage:
        # Train with default parameters
        modal run train_modal_integrated.py
        
        # Train with custom parameters
        modal run train_modal_integrated.py --epochs 100 --batch-size 64 --d-model 768 --n-layers 8
        
        # Use larger GPU
        # Edit the @app.function decorator to use gpu="A100-40GB" or gpu="H100"
    """
    import os
    
    print("=" * 80)
    print("🚀 MICHAEL SCOTT TRANSFORMER - MODAL GPU TRAINING")
    print("=" * 80)
    print("\n📂 Loading local data files...")
    
    # Load local data
    src_lines, tgt_lines = get_local_data(SRC_PATH, TGT_PATH)
    print(f"✅ Loaded {len(src_lines)} dialogue pairs\n")
    
    print("📤 Uploading data to Modal and starting training...")
    print(f"Configuration:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Model: d_model={d_model}, layers={n_layers}, heads={n_heads}, d_ff={d_ff}")
    print(f"  - Learning rate: {learning_rate}")
    print("=" * 80 + "\n")
    
    # Start training on remote GPU
    result = train_model_remote.remote(
        src_lines=src_lines,
        tgt_lines=tgt_lines,
        epochs=epochs,
        batch_size=batch_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        learning_rate=learning_rate,
    )
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Best validation loss: {result['best_val_loss']:.4f}")
    print(f"Final perplexity: {result['final_perplexity']:.2f}")
    print(f"Total epochs: {len(result['history'])}")
    print("=" * 80 + "\n")
    
    # Instructions for downloading
    print("📥 TO DOWNLOAD YOUR TRAINED MODEL:")
    print("=" * 80)
    print("Run these commands to download the model files:\n")
    print("  # Download best model weights")
    print("  modal volume get model-checkpoints checkpoints/best_model.pt ./models/weights_tiktoken.pt\n")
    print("  # Download model configuration")
    print("  modal volume get model-checkpoints checkpoints/config.json ./models/config_tiktoken.json\n")
    print("  # Download final model (last epoch)")
    print("  modal volume get model-checkpoints checkpoints/final_model.pt ./models/final_model.pt\n")
    print("=" * 80)
    print("\n💡 TIP: You can also browse all files with:")
    print("  modal volume ls model-checkpoints checkpoints/")
    print("\n🎉 All done! Your model is ready to use.")