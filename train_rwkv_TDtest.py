#!/usr/bin/env python3
"""
RWKV-v7 Time-Domain Speech Separation - NO AMP VERSION
=======================================================

This version has:
- ✅ CUDA cache clearing (prevents OOM)
- ❌ NO AMP (for checkpoint compatibility)
- ✅ All NaN debugging
- ✅ head_mode="direct"

Use this when resuming from checkpoints trained without AMP!
"""

import sys
import argparse
import csv
import os
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio

import soundfile as sf
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# --- RWKV v7 CUDA settings ---
os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")
os.environ.setdefault("RWKV_MY_TESTING", "x070")
os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")
os.environ.setdefault("RWKV_HEAD_SIZE_A", "64")

from rwkv_separator_Final import build_rwkv7_separator


# =========================
#   SEPFORMER ENCODER/DECODER (FROZEN)
# =========================

class SepFormerEncoder(nn.Module):
    def __init__(self, kernel_size=16, out_channels=256, in_channels=1):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            bias=False
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv1d(x)
        x = self.relu(x)
        return x


class SepFormerDecoder(nn.Module):
    def __init__(self, in_channels=256, out_channels=1, kernel_size=16, stride=8):
        super().__init__()
        self.transconv1d = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False
        )
    
    def forward(self, x):
        if x.dim() == 4:  # [B, S, 256, T_enc]
            B, S, C, T_enc = x.shape
            x = x.reshape(B * S, C, T_enc)
            x = self.transconv1d(x)
            if x.size(1) == 1:
                x = x.squeeze(1)
            x = x.view(B, S, -1)
        else:  # [B, 256, T_enc]
            x = self.transconv1d(x)
            if x.dim() == 3 and x.size(1) == 1:
                x = x.squeeze(1)
        return x


class SepFormerProcessor(nn.Module):
    def __init__(self, encoder_path, decoder_path):
        super().__init__()
        
        self.encoder = SepFormerEncoder(
            kernel_size=16,
            out_channels=256,
            in_channels=1
        )
        self.decoder = SepFormerDecoder(
            in_channels=256,
            out_channels=1,
            kernel_size=16,
            stride=8
        )
        
        print(f"[INFO] Loading encoder from: {encoder_path}")
        encoder_state = torch.load(encoder_path, map_location='cpu')
        self.encoder.load_state_dict(encoder_state)
        
        print(f"[INFO] Loading decoder from: {decoder_path}")
        decoder_state = torch.load(decoder_path, map_location='cpu')
        self.decoder.load_state_dict(decoder_state)
        
        self.freeze()
        print("[INFO] SepFormer encoder/decoder loaded and frozen ✓")
    
    def freeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self.decoder.eval()
    
    def encode(self, waveform):
        with torch.no_grad():
            return self.encoder(waveform)
    
    def decode(self, features, original_length=None):
        # NO torch.no_grad() - gradients must flow through!
        waveform = self.decoder(features)
        
        if original_length is not None:
            if waveform.dim() == 3:  # [B, S, T]
                T_current = waveform.size(-1)
                if T_current > original_length:
                    waveform = waveform[..., :original_length]
                elif T_current < original_length:
                    pad_size = original_length - T_current
                    waveform = F.pad(waveform, (0, pad_size))
            else:  # [B, T]
                T_current = waveform.size(-1)
                if T_current > original_length:
                    waveform = waveform[..., :original_length]
                elif T_current < original_length:
                    pad_size = original_length - T_current
                    waveform = F.pad(waveform, (0, pad_size))
        
        return waveform


# =========================
#   DATASET
# =========================

class Libri2MixDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        sample_rate: int = 8000,
        segment_seconds: float = 2.0,
        max_samples: int = None,
        subset_frac: float = None,
        subset_seed: int = 42,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_seconds * sample_rate)
        self.rows: List[Dict[str, str]] = []

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("mix_path") or not row.get("s1_path") or not row.get("s2_path"):
                    continue
                self.rows.append(row)

        if not self.rows:
            raise RuntimeError(f"No valid rows found in CSV: {csv_path}")

        import random
        N = len(self.rows)

        if max_samples is not None:
            n_keep = min(max_samples, N)
        elif subset_frac is not None:
            n_keep = max(1, int(round(N * subset_frac)))
        else:
            n_keep = N

        if n_keep < N:
            rng = random.Random(subset_seed)
            indices = list(range(N))
            rng.shuffle(indices)
            keep_idx = indices[:n_keep]
            self.rows = [self.rows[i] for i in keep_idx]
    
    def __len__(self) -> int:
        return len(self.rows)

    def _load_mono(self, path: str) -> torch.Tensor:
        data, sr = sf.read(path, dtype="float32")
        if data.ndim == 1:
            wav = torch.from_numpy(data).unsqueeze(0)
        else:
            wav = torch.from_numpy(data.T)
            wav = wav.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        return wav

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        mix = self._load_mono(row["mix_path"])
        s1 = self._load_mono(row["s1_path"])
        s2 = self._load_mono(row["s2_path"])

        T = min(mix.size(-1), s1.size(-1), s2.size(-1))
        mix = mix[..., :T]
        s1 = s1[..., :T]
        s2 = s2[..., :T]

        seg = self.segment_samples
        if T > seg:
            start = torch.randint(0, T - seg + 1, (1,)).item()
            end = start + seg
            mix = mix[..., start:end]
            s1 = s1[..., start:end]
            s2 = s2[..., start:end]
        elif T < seg:
            pad = seg - T
            mix = F.pad(mix, (0, pad))
            s1 = F.pad(s1, (0, pad))
            s2 = F.pad(s2, (0, pad))

        sources = torch.cat([s1, s2], dim=0)

        return {
            "mix": mix.squeeze(0),
            "sources": sources,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    mix = torch.stack([b["mix"] for b in batch], dim=0)
    sources = torch.stack([b["sources"] for b in batch], dim=0)
    return mix, sources


# =========================
#   ROBUST SI-SDR LOSS
# =========================

def si_sdr(est: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Ultra-robust SI-SDR implementation"""
    est = est - est.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    
    if torch.isnan(est).any() or torch.isnan(target).any():
        print("[WARNING] NaN in SI-SDR input!")
        return torch.zeros(est.size(0), device=est.device)
    
    dot = (est * target).sum(dim=-1, keepdim=True)
    s_target_energy = (target ** 2).sum(dim=-1, keepdim=True).clamp(min=eps)
    alpha = dot / s_target_energy
    
    target_scaled = alpha * target
    noise = est - target_scaled
    
    signal_energy = (target_scaled ** 2).sum(dim=-1).clamp(min=eps)
    noise_energy = (noise ** 2).sum(dim=-1).clamp(min=eps)
    
    ratio = (signal_energy / noise_energy).clamp(min=1e-10, max=1e10)
    si_sdr_value = 10 * torch.log10(ratio)
    si_sdr_value = torch.clamp(si_sdr_value, min=-30, max=30)
    
    if torch.isnan(si_sdr_value).any():
        print("[WARNING] NaN in SI-SDR output! Returning zeros.")
        return torch.zeros_like(si_sdr_value)
    
    return si_sdr_value


def pit_si_sdr_loss(est_sources: torch.Tensor, target_sources: torch.Tensor) -> torch.Tensor:
    """Permutation Invariant Training with robust SI-SDR"""
    B, num_sources, T = est_sources.shape
    
    if num_sources == 2:
        sdr_1 = si_sdr(est_sources[:, 0], target_sources[:, 0])
        sdr_2 = si_sdr(est_sources[:, 1], target_sources[:, 1])
        perm1_sdr = sdr_1 + sdr_2
        
        sdr_1 = si_sdr(est_sources[:, 0], target_sources[:, 1])
        sdr_2 = si_sdr(est_sources[:, 1], target_sources[:, 0])
        perm2_sdr = sdr_1 + sdr_2
        
        best_sdr = torch.maximum(perm1_sdr, perm2_sdr)
        loss = -best_sdr.mean()
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("[ERROR] NaN/Inf loss detected! Returning large value.")
            return torch.tensor(100.0, device=loss.device, requires_grad=True)
        
        return loss
    else:
        raise NotImplementedError("PIT loss only implemented for 2 sources")


# =========================
#   TRAINING LOOP (NO AMP, WITH CACHE CLEARING)
# =========================

def train_one_epoch(
    epoch: int,
    model: nn.Module,
    sepformer_processor: SepFormerProcessor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 5.0,
    writer: SummaryWriter = None
) -> float:
    """Training with FP32 + cache clearing"""
    
    model.train()
    sepformer_processor.eval()
    
    total_loss = 0.0
    num_batches_done = 0
    nan_count = 0
    
    pbar = tqdm(dataloader, desc=f"[TRAIN Epoch {epoch:03d}]", leave=False)
    
    for batch_idx, (mix_wav, src_wav) in enumerate(pbar):
        mix_wav = mix_wav.to(device)
        src_wav = src_wav.to(device)
        
        T_wav = mix_wav.size(-1)
        
        optimizer.zero_grad()
        
        # Regular FP32 forward pass
        mix_features = sepformer_processor.encode(mix_wav)
        mix_features = mix_features.transpose(1, 2)
        
        if torch.isnan(mix_features).any():
            print(f"\n[ERROR] NaN in mix_features at batch {batch_idx}")
            nan_count += 1
            continue
        
        sep_features = model(mix_features)
        
        if torch.isnan(sep_features).any():
            print(f"\n[ERROR] NaN in sep_features at batch {batch_idx}")
            print(f"  Input range: [{mix_features.min():.3f}, {mix_features.max():.3f}]")
            nan_count += 1
            continue
        
        if batch_idx == 0 and epoch == 1:
            print(f"\n{'='*60}")
            print(f"[DEBUG] First batch diagnostics:")
            print(f"  Segment: {T_wav} samples")
            print(f"  RWKV input: {mix_features.shape}")
            print(f"  RWKV output: {sep_features.shape}")
            print(f"  Output range: [{sep_features.min():.3f}, {sep_features.max():.3f}]")
            print(f"  AMP: Disabled (FP32 only)")
            print(f"{'='*60}\n")
        
        sep_features = sep_features.permute(0, 2, 3, 1)
        sep_wav = sepformer_processor.decode(sep_features, original_length=T_wav)
        
        if torch.isnan(sep_wav).any():
            print(f"\n[ERROR] NaN in sep_wav at batch {batch_idx}")
            nan_count += 1
            continue
        
        T_min = min(sep_wav.size(-1), src_wav.size(-1))
        sep_wav = sep_wav[..., :T_min]
        src_wav = src_wav[..., :T_min]
        
        loss = pit_si_sdr_loss(sep_wav, src_wav)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n[ERROR] NaN/Inf loss at batch {batch_idx}")
            nan_count += 1
            continue
        
        # Backward
        loss.backward()
        
        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"\n[ERROR] NaN gradient in {name}")
                has_nan_grad = True
                break
        
        if has_nan_grad:
            nan_count += 1
            continue
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # CACHE CLEARING (prevents memory leak)
        if batch_idx % 50 == 0:  # More frequent!
            torch.cuda.empty_cache()
        
        batch_loss = float(loss.item())
        total_loss += batch_loss
        num_batches_done += 1
        
        pbar.set_postfix(loss=f"{batch_loss:.4f}", nan=nan_count)
        
        if writer is not None:
            global_step = (epoch - 1) * len(dataloader) + batch_idx
            writer.add_scalar("batch/train_loss", batch_loss, global_step)
    
    if nan_count > 0:
        print(f"\n[WARNING] Encountered {nan_count} NaN batches in epoch {epoch}")
    
    if num_batches_done == 0:
        print(f"\n[ERROR] No valid batches in epoch {epoch}!")
        return float('nan')
    
    return total_loss / num_batches_done


def validate(
    epoch: int,
    model: nn.Module,
    sepformer_processor: SepFormerProcessor,
    dataloader: DataLoader,
    device: torch.device
) -> float:
    """Validation"""
    
    model.eval()
    sepformer_processor.eval()
    
    total_loss = 0.0
    num_batches_done = 0
    
    pbar = tqdm(dataloader, desc=f"[VALID Epoch {epoch:03d}]", leave=False)
    
    with torch.no_grad():
        for mix_wav, src_wav in pbar:
            mix_wav = mix_wav.to(device)
            src_wav = src_wav.to(device)
            
            T_wav = mix_wav.size(-1)
            
            mix_features = sepformer_processor.encode(mix_wav)
            mix_features = mix_features.transpose(1, 2)
            sep_features = model(mix_features)
            sep_features = sep_features.permute(0, 2, 3, 1)
            sep_wav = sepformer_processor.decode(sep_features, original_length=T_wav)
            
            T_min = min(sep_wav.size(-1), src_wav.size(-1))
            sep_wav = sep_wav[..., :T_min]
            src_wav = src_wav[..., :T_min]
            
            loss = pit_si_sdr_loss(sep_wav, src_wav)
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                batch_loss = float(loss.item())
                total_loss += batch_loss
                num_batches_done += 1
                pbar.set_postfix(val_loss=total_loss / num_batches_done)
    
    if num_batches_done == 0:
        return float('nan')
    
    return total_loss / num_batches_done


# =========================
#   MAIN
# =========================

def main():
    ap = argparse.ArgumentParser(description="RWKV Time-Domain - NO AMP (Cache Clearing Only)")
    
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--valid_csv", type=str, required=True)
    ap.add_argument("--sample_rate", type=int, default=8000)
    ap.add_argument("--seg_sec", type=float, default=2.0)
    ap.add_argument("--subset_frac", type=float, default=None)
    
    ap.add_argument("--encoder_path", type=str, required=True)
    ap.add_argument("--decoder_path", type=str, required=True)
    
    ap.add_argument("--n_layer", type=int, default=8)
    ap.add_argument("--n_embd", type=int, default=512)
    ap.add_argument("--head_hidden", type=int, default=256)
    ap.add_argument("--n_groups", type=int, default=2)
    ap.add_argument("--head_mode", type=str, default="direct")
    
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    
    ap.add_argument("--lr_scheduler", action="store_true")
    ap.add_argument("--lr_scheduler_patience", type=int, default=3)
    ap.add_argument("--lr_scheduler_factor", type=float, default=0.5)
    ap.add_argument("--lr_scheduler_min_lr", type=float, default=1e-6)
    
    ap.add_argument("--early_stop", action="store_true")
    ap.add_argument("--early_stop_patience", type=int, default=20)
    
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--log_dir", type=str, default=None)
    ap.add_argument("--resume_checkpoint", type=str, default=None)

    args = ap.parse_args()

    if args.head_mode != "direct":
        print(f"\n{'='*60}")
        print(f"[WARNING] head_mode='{args.head_mode}' may cause NaN!")
        print(f"[WARNING] Recommended: --head_mode direct")
        print(f"{'='*60}\n")

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Using device: {device}")

    tb_log_dir = args.log_dir or os.path.join(args.save_dir, "tb")
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)

    if not os.path.exists(args.encoder_path):
        raise FileNotFoundError(f"Encoder not found: {args.encoder_path}")
    if not os.path.exists(args.decoder_path):
        raise FileNotFoundError(f"Decoder not found: {args.decoder_path}")

    print("[INFO] Loading datasets...")
    train_ds = Libri2MixDataset(
        args.train_csv, 
        sample_rate=args.sample_rate, 
        segment_seconds=args.seg_sec, 
        subset_frac=args.subset_frac
    )
    valid_ds = Libri2MixDataset(
        args.valid_csv, 
        sample_rate=args.sample_rate, 
        segment_seconds=args.seg_sec, 
        subset_frac=args.subset_frac
    )
    
    print(f"[INFO] Train samples: {len(train_ds)}")
    print(f"[INFO] Valid samples: {len(valid_ds)}")
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4, 
        collate_fn=collate_fn, 
        drop_last=True,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_ds, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=2, 
        collate_fn=collate_fn, 
        drop_last=True,
        pin_memory=True
    )

    print("[INFO] Initializing SepFormer processor...")
    sepformer_processor = SepFormerProcessor(
        encoder_path=args.encoder_path,
        decoder_path=args.decoder_path
    ).to(device)
    
    print("[INFO] Determining feature dimensions...")
    mix_example, _ = next(iter(train_loader))
    mix_example = mix_example.to(device)
    with torch.no_grad():
        features_example = sepformer_processor.encode(mix_example)
    _, codec_dim, T_enc = features_example.shape
    
    print(f"[INFO] Encoder output: [{codec_dim}, {T_enc}]")

    print("[INFO] Building RWKV separator...")
    model = build_rwkv7_separator(
        n_embd=args.n_embd,
        codec_dim=codec_dim,
        n_layer=args.n_layer,
        num_sources=2,
        head_mode=args.head_mode,
        enforce_bf16=False,
        head_hidden=args.head_hidden,
        n_groups=args.n_groups,
    ).to(device)

    rwkv_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n[INFO] ===== Configuration =====")
    print(f"[INFO] RWKV parameters:   {rwkv_params:,}")
    print(f"[INFO] head_mode:         {args.head_mode}")
    print(f"[INFO] Segment length:    {args.seg_sec}s")
    print(f"[INFO] Batch size:        {args.batch_size}")
    print(f"[INFO] AMP:               Disabled (FP32 only)")
    print(f"[INFO] Cache clearing:    Every 50 batches")
    print(f"[INFO] ============================\n")

    print("[INFO] Running pre-training sanity check...")
    with torch.no_grad():
        test_mix = mix_example[:2].to(device)
        test_feat = sepformer_processor.encode(test_mix).transpose(1, 2)
        test_sep = model(test_feat)
        test_sep = test_sep.permute(0, 2, 3, 1)
        test_wav = sepformer_processor.decode(test_sep)
        
        if torch.isnan(test_wav).any():
            print("[ERROR] ❌ Model outputs NaN even before training!")
            return
        else:
            print("[INFO] ✓ Pre-training check passed")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    scheduler = None
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="min", 
            factor=args.lr_scheduler_factor,
            patience=args.lr_scheduler_patience, 
            min_lr=args.lr_scheduler_min_lr
        )

    start_epoch = 1
    best_val = float("inf")

    if args.resume_checkpoint is not None:
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = float(ckpt.get("val_loss", float("inf")))
        print(f"[INFO] Resumed from epoch {start_epoch-1}")

    epochs_no_improve = 0

    print(f"\n[INFO] Starting training from epoch {start_epoch}...\n")
    
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            epoch, 
            model, 
            sepformer_processor, 
            train_loader, 
            optimizer, 
            device,
            grad_clip=args.grad_clip, 
            writer=writer
        )
        torch.cuda.empty_cache()

        val_loss = validate(
            epoch, 
            model, 
            sepformer_processor, 
            valid_loader, 
            device
        )
        torch.cuda.empty_cache()

        current_lr = optimizer.param_groups[0]["lr"]
        
        if np.isnan(train_loss) or np.isnan(val_loss):
            print(f"[EPOCH {epoch:03d}] train=nan val=nan lr={current_lr:.3e} ❌")
            break
        else:
            print(f"[EPOCH {epoch:03d}] train={train_loss:.4f} val={val_loss:.4f} lr={current_lr:.3e}")

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", current_lr, epoch)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            epochs_no_improve = 0
            ckpt_path = os.path.join(args.save_dir, f"best_epoch{epoch:03d}_loss{val_loss:.4f}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": vars(args),
            }, ckpt_path)
            print(f"  ✅ Saved: {ckpt_path}")
        else:
            epochs_no_improve += 1

        if scheduler is not None:
            scheduler.step(val_loss)

        if args.early_stop and epochs_no_improve >= args.early_stop_patience:
            print(f"[EARLY STOP] No improvement for {epochs_no_improve} epochs")
            break

    print(f"\n[DONE] Best validation loss: {best_val:.4f}")
    writer.close()


if __name__ == "__main__":
    main()