#!/usr/bin/env python3
"""
RWKV-v7 TF-Domain Speech Separation Training Script (STFT-based)

This script uses STFT as a FIXED transform (no learning needed for encoder/decoder).
RWKV processes magnitude spectrograms and predicts separated spectrograms.

Key differences from time-domain:
1. STFT/iSTFT are fixed transforms (not learned)
2. RWKV processes [B, T_tf, F] spectrograms
3. Uses magnitude + mixture phase for reconstruction
4. More stable than learned encoder/decoder

Usage:
python train_rwkv_TFtest.py --train_csv train.csv --valid_csv dev.csv \
    --sample_rate 16000 --epochs 100 --device cuda --lr_scheduler --n_fft 512 --hop_length 128 --n_layer 8 \
    --n_embd 512 --head_hidden 256 --n_groups 2 --batch_size 8 \
    --lr 1e-3 --seg_sec 3.0 --save_dir ./checkpoints_tf
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
from parameter_analysis import quick_analysis, estimate_epoch_time
from complexity_measurement import ComplexityMeasurement


# =========================
#   STFT PROCESSOR (FIXED TRANSFORM)
# =========================

class STFTProcessor(nn.Module):
    """
    Fixed STFT/iSTFT transform for TF-domain processing.
    No learning needed - pure mathematical transform.
    """
    def __init__(self, n_fft=512, hop_length=128, win_length=None, window='hann'):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.window_type = window
        
        # Register window as buffer (non-trainable)
        if window == 'hann':
            self.register_buffer('window', torch.hann_window(self.win_length))
        elif window == 'hamming':
            self.register_buffer('window', torch.hamming_window(self.win_length))
        else:
            self.register_buffer('window', torch.ones(self.win_length))
    
    def stft(self, x):
        """
        Compute STFT
        Args:
            x: [B, 1, T_wav] or [B, T_wav] waveform
        Returns:
            magnitude: [B, F, T_tf]
            phase: [B, F, T_tf]
        """
        if x.dim() == 3:
            x = x.squeeze(1)  # [B, T_wav]
        
        # Compute STFT
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
            normalized=False,
        )  # [B, F, T_tf]
        
        magnitude = spec.abs()
        phase = spec.angle()
        
        return magnitude, phase
    
    def istft(self, magnitude, phase, length=None):
        """
        Compute inverse STFT
        Args:
            magnitude: [B, F, T_tf] or [B, S, F, T_tf]
            phase: [B, F, T_tf]
            length: original waveform length
        Returns:
            wav: [B, T_wav] or [B, S, T_wav]
        """
        if magnitude.dim() == 4:  # [B, S, F, T_tf]
            B, S, F, T_tf = magnitude.shape
            # Expand phase for each source
            phase = phase.unsqueeze(1).expand(B, S, F, T_tf)
            
            # Reconstruct complex spectrogram
            real = magnitude * torch.cos(phase)
            imag = magnitude * torch.sin(phase)
            complex_spec = torch.complex(real, imag)
            
            # Reshape for batch processing
            complex_spec = complex_spec.reshape(B * S, F, T_tf)
            
            # iSTFT
            wav = torch.istft(
                complex_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                length=length,
                center=True,
                normalized=False,
            )  # [B*S, T_wav]
            
            wav = wav.view(B, S, -1)
        else:  # [B, F, T_tf]
            real = magnitude * torch.cos(phase)
            imag = magnitude * torch.sin(phase)
            complex_spec = torch.complex(real, imag)
            
            wav = torch.istft(
                complex_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                length=length,
                center=True,
                normalized=False,
            )  # [B, T_wav]
        
        return wav


# =========================
#   DATASET
# =========================

class Libri2MixDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        sample_rate: int = 16000,
        segment_seconds: float = 3.0,
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

        sources = torch.stack([s1, s2], dim=0)
        return {"mix": mix, "sources": sources}


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = [b["mix"].shape[-1] for b in batch]
    T_max = max(lengths)

    mix_list = []
    sources_list = []

    for b in batch:
        mix = b["mix"]
        sources = b["sources"]
        T = mix.shape[-1]
        pad_T = T_max - T

        if pad_T > 0:
            mix = F.pad(mix, (0, pad_T))
            sources = F.pad(sources, (0, pad_T))

        mix_list.append(mix)
        sources_list.append(sources)

    return torch.stack(mix_list, dim=0), torch.stack(sources_list, dim=0)


# =========================
#   SI-SDR + PIT
# =========================

def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)

    ref_energy = (ref ** 2).sum(dim=-1, keepdim=True) + eps
    optimal_scaling = (est * ref).sum(dim=-1, keepdim=True) / ref_energy

    ref_scaled = optimal_scaling * ref

    signal = ref_scaled
    noise = est - ref_scaled

    signal_power = (signal ** 2).sum(dim=-1)
    noise_power = (noise ** 2).sum(dim=-1)

    si_sdr_value = 10.0 * torch.log10(signal_power / (noise_power + eps) + eps)
    return si_sdr_value


def pit_si_sdr_loss(est_sources: torch.Tensor, ref_sources: torch.Tensor, 
                    eps: float = 1e-8) -> torch.Tensor:
    B, S, T = est_sources.shape
    assert S == 2, "PIT currently only supports 2 sources"

    perm_0_ref = ref_sources
    perm_1_ref = ref_sources.flip(dims=[1])

    est_flat = est_sources.reshape(B * S, T)
    perm_0_flat = perm_0_ref.reshape(B * S, T)
    perm_1_flat = perm_1_ref.reshape(B * S, T)

    sdr_perm_0 = si_sdr(est_flat, perm_0_flat, eps).view(B, S)
    sdr_perm_1 = si_sdr(est_flat, perm_1_flat, eps).view(B, S)

    loss_perm_0 = sdr_perm_0.sum(dim=1)
    loss_perm_1 = sdr_perm_1.sum(dim=1)

    best_sdr = torch.max(loss_perm_0, loss_perm_1)

    return -best_sdr.mean()


# =========================
#   TRAINING LOOP
# =========================

def train_one_epoch(
    epoch: int,
    model: nn.Module,
    stft_processor: STFTProcessor,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 5.0,
    writer: SummaryWriter = None,
    complexity_meter: "ComplexityMeasurement" = None,
) -> float:
    model.train()

    total_samples = 0
    if complexity_meter:
        complexity_meter.start_epoch_timing()

    total_loss = 0.0
    num_batches_done = 0

    pbar = tqdm(train_loader, desc=f"[Epoch {epoch:03d} TRAIN]", ncols=120)

    for batch_idx, (mix_wav, src_wav) in enumerate(pbar):
        B = mix_wav.shape[0]
        total_samples += B
        
        mix_wav = mix_wav.to(device)
        src_wav = src_wav.to(device)
        
        B, S, _, T_wav = src_wav.shape
        src_wav = src_wav.squeeze(2)  # [B, S, T_wav]
        
        optimizer.zero_grad()

        # Compute STFT
        with torch.no_grad():
            mix_mag, mix_phase = stft_processor.stft(mix_wav)  # [B, F, T_tf]
        
        # Transpose for RWKV: [B, F, T_tf] -> [B, T_tf, F]
        mix_mag_input = mix_mag.transpose(1, 2)
        
        # Forward through model
        sep_features = model(mix_mag_input)  # [B, T_tf, S, F]
        
        # Transpose back: [B, T_tf, S, F] -> [B, S, F, T_tf]
        sep_mag = sep_features.transpose(1, 2).transpose(2, 3)
        
        # Apply ReLU to ensure non-negative magnitudes
        sep_mag = F.relu(sep_mag)
        
        # Reconstruct waveforms using mixture phase
        sep_wav = stft_processor.istft(sep_mag, mix_phase, length=T_wav)  # [B, S, T_wav]
        
        # Align lengths
        T_min = min(sep_wav.size(-1), src_wav.size(-1))
        sep_wav_aligned = sep_wav[..., :T_min]
        src_wav_aligned = src_wav[..., :T_min]
        
        # Compute loss
        loss = pit_si_sdr_loss(sep_wav_aligned, src_wav_aligned)
        
        # Backward
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()

        batch_loss = float(loss.item())
        total_loss += batch_loss
        num_batches_done += 1

        pbar.set_postfix(train_loss=total_loss / num_batches_done)

        if writer is not None:
            global_step = (epoch - 1) * len(train_loader) + batch_idx
            writer.add_scalar("batch/train_loss", batch_loss, global_step)

    return total_loss / max(1, num_batches_done)


def validate(
    epoch: int,
    model: nn.Module,
    stft_processor: STFTProcessor,
    valid_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()

    total_samples = 0
    if complexity_meter:
        complexity_meter.start_epoch_timing()

    total_loss = 0.0
    num_batches_done = 0

    pbar = tqdm(valid_loader, desc=f"[Epoch {epoch:03d} VALID]", ncols=120)

    with torch.no_grad():
        for mix_wav, src_wav in pbar:
            mix_wav = mix_wav.to(device)
            src_wav = src_wav.to(device)
            
            B, S, _, T_wav = src_wav.shape
            src_wav = src_wav.squeeze(2)

            mix_mag, mix_phase = stft_processor.stft(mix_wav)
            mix_mag_input = mix_mag.transpose(1, 2)
            
            sep_features = model(mix_mag_input)
            sep_mag = sep_features.transpose(1, 2).transpose(2, 3)
            sep_mag = F.relu(sep_mag)
            
            sep_wav = stft_processor.istft(sep_mag, mix_phase, length=T_wav)
            
            T_min = min(sep_wav.size(-1), src_wav.size(-1))
            sep_wav_aligned = sep_wav[..., :T_min]
            src_wav_aligned = src_wav[..., :T_min]
            
            loss = pit_si_sdr_loss(sep_wav_aligned, src_wav_aligned)

            batch_loss = float(loss.item())
            total_loss += batch_loss
            num_batches_done += 1

            pbar.set_postfix(val_loss=total_loss / num_batches_done)

    return total_loss / max(1, num_batches_done)


# =========================
#   MAIN
# =========================

def main():
    ap = argparse.ArgumentParser(description="RWKV-v7 TF-Domain Speech Separation")
    
    # Data
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--valid_csv", type=str, required=True)
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--seg_sec", type=float, required=True)
    ap.add_argument("--subset_frac", type=float, default=None)
    
    # STFT parameters
    ap.add_argument("--n_fft", type=int, default=512, help="FFT size")
    ap.add_argument("--hop_length", type=int, default=128, help="Hop length for STFT")
    
    # Model architecture
    ap.add_argument("--n_layer", type=int, default=8)
    ap.add_argument("--n_embd", type=int, required=True)
    ap.add_argument("--head_hidden", type=int, required=True)
    ap.add_argument("--n_groups", type=int, required=True)
    ap.add_argument("--head_mode", type=str, default="residual")
    
    # Training
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    
    # LR scheduler
    ap.add_argument("--lr_scheduler", action="store_true")
    ap.add_argument("--lr_scheduler_patience", type=int, default=5)
    ap.add_argument("--lr_scheduler_factor", type=float, default=0.5)
    ap.add_argument("--lr_scheduler_min_lr", type=float, default=1e-6)
    
    # Early stopping
    ap.add_argument("--early_stop", action="store_true")
    ap.add_argument("--early_stop_patience", type=int, default=20)
    
    # System
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--log_dir", type=str, default=None)
    ap.add_argument("--resume_checkpoint", type=str, default=None)

    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Using device: {device}")

    tb_log_dir = args.log_dir or os.path.join(args.save_dir, "tb")
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)

    # Dataset
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
        drop_last=True
    )
    valid_loader = DataLoader(
        valid_ds, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=2, 
        collate_fn=collate_fn, 
        drop_last=True
    )

    # STFT Processor (FIXED - no learning!)
    print("[INFO] Initializing STFT processor...")
    stft_processor = STFTProcessor(n_fft=args.n_fft, hop_length=args.hop_length).to(device)
    
    # Get feature dimension
    mix_example, _ = next(iter(train_loader))
    mix_example = mix_example.to(device)
    with torch.no_grad():
        mag, phase = stft_processor.stft(mix_example)
    _, F, T_tf = mag.shape
    
    print(f"[INFO] STFT dimensions: F={F}, T_tf={T_tf}")
    print(f"[INFO] n_fft={args.n_fft}, hop_length={args.hop_length}")

    # Model
    print("[INFO] Building RWKV separator...")
    model = build_rwkv7_separator(
        n_embd=args.n_embd,
        codec_dim=F,  # Frequency bins as feature dimension
        n_layer=args.n_layer,
        num_sources=2,
        head_mode=args.head_mode,
        enforce_bf16=False,
        head_hidden=args.head_hidden,
        n_groups=args.n_groups,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n[INFO] ===== Model Configuration =====")
    print(f"[INFO] RWKV parameters:   {num_params:,}")
    print(f"[INFO] STFT (fixed):      No parameters (mathematical transform)")
    print(f"[INFO] Head mode:         {args.head_mode}")
    print(f"[INFO] Number of layers:  {args.n_layer}")
    print(f"[INFO] Number of groups:  {args.n_groups}")
    print(f"[INFO] RWKV embedding:    {args.n_embd}")
    print(f"[INFO] Head hidden:       {args.head_hidden}")
    print(f"[INFO] Feature dimension: {F}")
    print(f"[INFO] Segment length:    {args.seg_sec}s")
    print(f"[INFO] ================================\n")

    optimizer = torch.optim.AdamW(
        model.parameters(),  # Only RWKV parameters (STFT is fixed!)
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

    # Initialize complexity measurement
    complexity_log_path = os.path.join(args.save_dir, "complexity_tf_domain.txt")
    complexity_meter = ComplexityMeasurement(model, device, log_file=complexity_log_path)
    complexity_meter.measure_model_complexity(input_shape=(args.batch_size, T_tf, F))
    
    # Measure CodecFormer-style inference MACs (for fair comparison)
    complexity_meter.measure_inference_macs(
        sample_rate=args.sample_rate,  # Typically 16000 for TF domain
        test_duration=2.0,  # Standard 2-second test
        domain='tf'
    )
    sys.exit()

    print(f"\n[INFO] Starting training from epoch {start_epoch}...\n")
    
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            epoch, 
            model, 
            stft_processor, 
            train_loader, 
            optimizer, 
            device,
            grad_clip=args.grad_clip, 
            writer=writer,
            complexity_meter=complexity_meter,
        )

        val_loss = validate(
            epoch, 
            model, 
            stft_processor, 
            valid_loader, 
            device
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        if args.early_stop and epochs_no_improve >= args.early_stop_patience:
            print(f"[EARLY STOP] No improvement for {epochs_no_improve} epochs")
            break

        current_lr = optimizer.param_groups[0]["lr"]
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
                "n_fft": args.n_fft,
                "hop_length": args.hop_length,
            }, ckpt_path)
            print(f"  ✅ Saved: {ckpt_path}")
        else:
            epochs_no_improve += 1

    print(f"\n[DONE] Best validation loss: {best_val:.4f}")
    complexity_meter.print_final_summary()
    writer.close()


if __name__ == "__main__":
    main()