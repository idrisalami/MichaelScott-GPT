import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm.auto import tqdm

from utils.masks import Masks

@dataclass
class TrainConfig:
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.05
    clip_grad_norm: float = 1.0
    pad_id: int = 0
    epochs: int = 10
    amp: bool = False  # set True on CUDA if you want mixed precision
    log_progress: bool = True
    label_smoothing: float = 0.0  # label smoothing for CrossEntropyLoss (0.0 = disabled, 0.1 = recommended)
    use_scheduler: bool = False  # use cosine annealing LR scheduler
    checkpoint_dir: Optional[str] = None  # directory to save checkpoints (None = no checkpoints)
    checkpoint_frequency: int = 10  # save checkpoint every N epochs
    volume_commit_fn: Optional[callable] = None  # function to call after saving (for Modal volumes)


class Trainer:
    def __init__(self, model: nn.Module, device: Optional[str] = None, cfg: Optional[TrainConfig] = None):
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

        self.cfg = cfg or TrainConfig()
        self.crit = nn.CrossEntropyLoss(ignore_index=self.cfg.pad_id, label_smoothing=self.cfg.label_smoothing)
        self.opt = AdamW(self.model.parameters(),
                         lr=self.cfg.lr,
                         betas=self.cfg.betas,
                         weight_decay=self.cfg.weight_decay)
        self.masks = Masks(pad_id=self.cfg.pad_id)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp)
        
        # Optional: learning rate scheduler
        self.scheduler = None
        if self.cfg.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.cfg.epochs)

    def _run_epoch(self, loader, train: bool) -> float:
        self.model.train(train)
        total, steps = 0.0, 0

        pbar = tqdm(loader, desc=("train epoch" if train else "val epoch"),
                    leave=False, disable=not self.cfg.log_progress)

        for src, dec_in, labels in pbar:
            src, dec_in, labels = src.to(self.device), dec_in.to(self.device), labels.to(self.device)

            # masks
            src_mask = self.masks.encoder(src)
            _, _, tgt_mask = self.masks.decoder(dec_in)

            # forward (+ AMP if enabled)
            with torch.amp.autocast("cuda", enabled=self.cfg.amp):
                enc_out = self.model.encode(src, src_mask)
                dec_out = self.model.decode(enc_out, src_mask, dec_in, tgt_mask)
                logits = self.model.project(dec_out)             # (B, T-1, V)
                loss = self.crit(logits.reshape(-1, logits.size(-1)),
                                 labels.reshape(-1))             # use reshape, not view

            if train:
                self.opt.zero_grad(set_to_none=True)
                if self.cfg.amp:
                    self.scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)
                    self.opt.step()

            total += loss.item()
            steps += 1
            if self.cfg.log_progress:
                pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        pbar.close()
        return total / max(1, steps)

    @torch.no_grad()
    def evaluate(self, loader) -> float:
        return self._run_epoch(loader, train=False)

    def fit(self, train_loader, val_loader=None):
        import os
        history = []
        best_val_loss = float('inf')
        
        for epoch in range(self.cfg.epochs):
            tr = self._run_epoch(train_loader, train=True)
            
            if val_loader is not None:
                va = self.evaluate(val_loader)
                ppl = math.exp(min(va, 10))  # Cap for numerical stability
                
                # Print with learning rate if scheduler is enabled
                if self.scheduler:
                    lr = self.scheduler.get_last_lr()[0]
                    print(f"epoch {epoch+1:02d} | train {tr:.3f} | val {va:.3f} | ppl {ppl:.2f} | lr {lr:.2e}")
                else:
                    print(f"epoch {epoch+1:02d} | train {tr:.3f} | val {va:.3f} | ppl {ppl:.2f}")
                
                history.append({"epoch": epoch+1, "train": tr, "val": va, "ppl": ppl})
                
                # Save best checkpoint if checkpoint_dir is specified
                if self.cfg.checkpoint_dir and va < best_val_loss:
                    best_val_loss = va
                    checkpoint_path = os.path.join(self.cfg.checkpoint_dir, "best_model.pt")
                    os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.opt.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                        'train_loss': tr,
                        'val_loss': va,
                        'perplexity': ppl,
                    }, checkpoint_path)
                    print(f"  ✅ Saved best model (val_loss: {va:.4f})")
                    
                    # Call volume commit function if provided (for Modal)
                    if self.cfg.volume_commit_fn:
                        self.cfg.volume_commit_fn()
                
                # Save periodic checkpoints
                if self.cfg.checkpoint_dir and (epoch + 1) % self.cfg.checkpoint_frequency == 0:
                    checkpoint_path = os.path.join(self.cfg.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.opt.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                        'train_loss': tr,
                        'val_loss': va,
                    }, checkpoint_path)
                    if self.cfg.volume_commit_fn:
                        self.cfg.volume_commit_fn()
            else:
                print(f"epoch {epoch+1:02d} | train {tr:.3f}")
                history.append({"epoch": epoch+1, "train": tr})
            
            # Step scheduler if enabled
            if self.scheduler:
                self.scheduler.step()
        
        return history, best_val_loss if val_loader else None