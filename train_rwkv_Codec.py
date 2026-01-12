#!/usr/bin/env python3
"""
FIXED RWKV-v7 + DAC training script (CodecFormer-style).

Key fixes from original:
1. Length alignment between est_sources and codec_sources
2. Added auxiliary latent-domain loss option
3. Fixed SI-SDR eps handling
4. Gradient flow verification
5. Uses fixed separator model

python train_rwkv_Codec.py --train_csv train_min.csv --valid_csv dev_min.csv \
    --sample_rate 16000 --epochs 100 --device "cuda" --head_mode "residual" \
    --lr_scheduler --early_stop --n_layer 8 --batch_size 16 --latent_loss_weight 0.5
"""

import sys
import argparse
import csv
import math
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
from codecformer3 import DACWrapper
from parameter_analysis import quick_analysis, print_thesis_summary, estimate_epoch_time
from complexity_measurement import ComplexityMeasurement

# =========================
#   DATASET
# =========================

#class Libri2MixDataset(Dataset):
#    def __init__(self, csv_path: str, sample_rate: int = 16000, segment_seconds: float = 3.0):
#        super().__init__()
#        self.sample_rate = sample_rate
#        self.segment_samples = int(segment_seconds * sample_rate)
#        self.rows: List[Dict[str, str]] = []
#
#        with open(csv_path, "r", newline="") as f:
#            reader = csv.DictReader(f)
#            for row in reader:
#                if not row.get("mix_path") or not row.get("s1_path") or not row.get("s2_path"):
#                    continue
#                self.rows.append(row)
#
#        if not self.rows:
#            raise RuntimeError(f"No valid rows found in CSV: {csv_path}")

# ------------------------------------------------------------------------------------------------ #
'''
def check_mixture_consistency(ds, idx=0):
    sample = ds[idx]
    mix = sample["mix"]               # [1, T]
    s1, s2 = sample["sources"]        # [2, 1, T] -> s1, s2 each [1, T]

    approx_mix = s1 + s2
    diff = (mix - approx_mix).abs()
    print(f"Sample {idx}:")
    print("  mix shape:", mix.shape)
    print("  s1 shape: ", s1.shape)
    print("  s2 shape: ", s2.shape)
    print("  max |mix - (s1+s2)|:", diff.max().item())
    print("  mean |mix - (s1+s2)|:", diff.mean().item())

def inspect_dataset(ds, name, n=3):
    print(f"\n=== Inspecting {name} (first {n} samples) ===")
    for i in range(min(n, len(ds))):
        row = ds.rows[i]  # the underlying CSV row

        print(f"\n--- Sample {i} ---")
        print("mix_path:", row["mix_path"])
        print("s1_path:",  row["s1_path"])
        print("s2_path:",  row["s2_path"])

        sample = ds[i]
        mix = sample["mix"]         # [1, T]
        sources = sample["sources"] # [2, 1, T]

        print("mix shape:     ", mix.shape)
        print("sources shape: ", sources.shape)
'''
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

        # --- load all rows from CSV ---
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("mix_path") or not row.get("s1_path") or not row.get("s2_path"):
                    continue
                self.rows.append(row)

        if not self.rows:
            raise RuntimeError(f"No valid rows found in CSV: {csv_path}")

        # --- optionally create a subset ---
        # priority: max_samples > subset_frac > full dataset
        import random

        N = len(self.rows)

        if max_samples is not None:
            n_keep = min(max_samples, N)
        elif subset_frac is not None:
            n_keep = max(1, int(round(N * subset_frac)))
        else:
            n_keep = N  # keep everything

        if n_keep < N:
            rng = random.Random(subset_seed)
            indices = list(range(N))
            rng.shuffle(indices)
            keep_idx = indices[:n_keep]
            self.rows = [self.rows[i] for i in keep_idx]
# ------------------------------------------------------------------------------------------------ #
    
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
#   SI-SDR + PIT (FIXED)
# =========================

def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-invariant SDR (SI-SDR) in dB.
    Fixed: eps only in denominator, not redundantly in log.
    """
    ref_zm = ref - ref.mean(dim=-1, keepdim=True)
    est_zm = est - est.mean(dim=-1, keepdim=True)

    dot = (est_zm * ref_zm).sum(dim=-1, keepdim=True)
    ref_energy = (ref_zm ** 2).sum(dim=-1, keepdim=True) + eps
    s_target = dot / ref_energy * ref_zm

    e_noise = est_zm - s_target

    s_target_energy = (s_target ** 2).sum(dim=-1) + eps
    e_noise_energy = (e_noise ** 2).sum(dim=-1) + eps

    ratio = s_target_energy / e_noise_energy
    return 10 * torch.log10(ratio)  # FIXED: no redundant eps


def pit_si_sdr_loss(est_sources: torch.Tensor,
                    true_sources: torch.Tensor) -> torch.Tensor:
    """2-speaker PIT SI-SDR loss."""
    assert est_sources.ndim == 3 and true_sources.ndim == 3
    B, S, T = est_sources.shape
    assert S == 2

    est1 = est_sources[:, 0, :]
    est2 = est_sources[:, 1, :]
    s1 = true_sources[:, 0, :]
    s2 = true_sources[:, 1, :]

    sdr11 = si_sdr(est1, s1)
    sdr22 = si_sdr(est2, s2)
    loss_perm1 = -(sdr11 + sdr22)

    sdr12 = si_sdr(est1, s2)
    sdr21 = si_sdr(est2, s1)
    loss_perm2 = -(sdr12 + sdr21)

    loss = torch.minimum(loss_perm1, loss_perm2)
    return loss.mean()


def pit_latent_loss(est_lat: torch.Tensor, 
                    true_lat: torch.Tensor,
                    loss_type: str = "l1") -> torch.Tensor:
    """
    PIT loss in latent domain.
    est_lat, true_lat: [B, S, C, T_lat]
    """
    B, S, C, T = est_lat.shape
    assert S == 2
    
    if loss_type == "l1":
        loss_fn = F.l1_loss
    else:
        loss_fn = F.mse_loss
    
    # Perm 1: est[0]→true[0], est[1]→true[1]
    loss_perm1 = loss_fn(est_lat[:, 0], true_lat[:, 0], reduction='none').mean(dim=(1, 2)) + \
                 loss_fn(est_lat[:, 1], true_lat[:, 1], reduction='none').mean(dim=(1, 2))
    
    # Perm 2: est[0]→true[1], est[1]→true[0]
    loss_perm2 = loss_fn(est_lat[:, 0], true_lat[:, 1], reduction='none').mean(dim=(1, 2)) + \
                 loss_fn(est_lat[:, 1], true_lat[:, 0], reduction='none').mean(dim=(1, 2))
    
    loss = torch.minimum(loss_perm1, loss_perm2)
    return loss.mean()


# =========================
#   TRAINING LOOP (FIXED)
# =========================

def train_one_epoch(
    epoch: int,
    model: nn.Module,
    dac: DACWrapper,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 5.0,
    writer: SummaryWriter = None,
    latent_loss_weight: float = 0.0,
    check_gradients: bool = False,
                   complexity_meter=None,
) -> float:
    model.train()
    dac.model.eval()

    total_loss = 0.0
    num_batches_done = 0

    num_batches = len(loader)
    log_interval = max(1, num_batches // 20)
    pbar = tqdm(loader, desc=f"Train epoch {epoch:03d}", leave=False)

    for batch_idx, (mix, sources) in enumerate(pbar, start=1):
        mix = mix.to(device)
        sources = sources.to(device)
        B, S, _, T_orig = sources.shape
        assert S == 2

        optimizer.zero_grad()

        # ---------- DAC encode mixture ----------
        with torch.no_grad():
            mix_enc, orig_len = dac.get_encoded_features(mix)
            mix_q, _, _, _, _ = dac.get_quantized_features(mix_enc)

        z_mix = mix_q.permute(0, 2, 1).contiguous()  # [B, T_lat, C_lat]

        # ---------- RWKV-v7 separation ----------
        sep_lat = model(z_mix)  # [B, T_lat, S, C_lat]
        sep_lat = sep_lat.permute(0, 2, 3, 1).contiguous()  # [B, S, C, T_lat]

        # ---------- DAC decode ----------
        est_wavs = []
        for s_idx in range(S):
            z_s = sep_lat[:, s_idx, :, :]
            wav_hat = dac.get_decoded_signal(z_s, orig_len)
            est_wavs.append(wav_hat)

        est_sources = torch.stack(est_wavs, dim=1).squeeze(2)  # [B, S, T']

        # ---------- FIX: Length alignment ----------
        T_est = est_sources.shape[-1]
        
        # Codec targets
        sources_squeezed = sources.squeeze(2)  # [B, S, T_orig]
        sources_flat = sources_squeezed.reshape(B * S, 1, T_orig)

        with torch.no_grad():
            src_enc, src_orig_len = dac.get_encoded_features(sources_flat)
            src_q, _, _, _, _ = dac.get_quantized_features(src_enc)
            codec_src_flat = dac.get_decoded_signal(src_q, src_orig_len)

        T_codec = codec_src_flat.shape[-1]
        codec_sources = codec_src_flat.view(B, S, T_codec)

        # Align all lengths
        min_len = min(T_est, T_codec, T_orig)
        est_sources_aligned = est_sources[..., :min_len]
        codec_sources_aligned = codec_sources[..., :min_len]

        # ---------- PIT Codec SI-SDR loss ----------
        loss_wave = pit_si_sdr_loss(est_sources_aligned, codec_sources_aligned)
        
        # ---------- Optional: Latent domain loss ----------
        loss_latent = torch.tensor(0.0, device=device)
        if latent_loss_weight > 0:
            # Get target latents
            with torch.no_grad():
                _, T_lat, C_lat = z_mix.shape
                # Reshape src_q to [B, S, C, T_lat]
                target_lat = src_q.view(B, S, C_lat, -1)
                # Align latent time dimension
                T_lat_target = target_lat.shape[-1]
                T_lat_est = sep_lat.shape[-1]
                min_lat_len = min(T_lat_target, T_lat_est)
                
            sep_lat_aligned = sep_lat[..., :min_lat_len]
            target_lat_aligned = target_lat[..., :min_lat_len]
            
            loss_latent = pit_latent_loss(sep_lat_aligned, target_lat_aligned)
        
        # Combined loss
        loss = loss_wave + latent_loss_weight * loss_latent

        loss.backward()
        
        # ---------- Gradient check (first batch only) ----------
        if check_gradients and batch_idx == 1:
            print("\n[GRADIENT CHECK]")
            has_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 0:
                        has_grad = True
                    if batch_idx == 1 and "output" in name:
                        print(f"  {name}: grad_norm={grad_norm:.6f}")
            if not has_grad:
                print("  WARNING: No gradients detected! Check DAC decoder gradient flow.")
            else:
                print("  Gradients OK.")
        
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_loss = float(loss.item())
        total_loss += batch_loss
        num_batches_done += 1
        avg_loss = total_loss / num_batches_done

        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=avg_loss, wave=float(loss_wave.item()), lr=current_lr)

        if writer is not None and (batch_idx % log_interval == 0 or batch_idx == num_batches):
            global_step = (epoch - 1) * num_batches + batch_idx
            writer.add_scalar("loss/train_step", batch_loss, global_step)
            writer.add_scalar("loss/train_wave", float(loss_wave.item()), global_step)
            if latent_loss_weight > 0:
                writer.add_scalar("loss/train_latent", float(loss_latent.item()), global_step)

    return total_loss / max(1, num_batches_done)


@torch.no_grad()
def validate(
    epoch: int,
    model: nn.Module,
    dac: DACWrapper,
    loader: DataLoader,
    device: torch.device,
    latent_loss_weight: float = 0.0,
) -> float:
    model.eval()
    dac.model.eval()
    total_loss = 0.0
    num_batches_done = 0

    pbar = tqdm(loader, desc=f"Val   epoch {epoch:03d}", leave=False)

    for mix, sources in pbar:
        mix = mix.to(device)
        sources = sources.to(device)
        B, S, _, T_orig = sources.shape
        assert S == 2

        mix_enc, orig_len = dac.get_encoded_features(mix)
        mix_q, _, _, _, _ = dac.get_quantized_features(mix_enc)
        z_mix = mix_q.permute(0, 2, 1).contiguous()

        sep_lat = model(z_mix)
        sep_lat = sep_lat.permute(0, 2, 3, 1).contiguous()

        est_wavs = []
        for s_idx in range(S):
            z_s = sep_lat[:, s_idx, :, :]
            wav_hat = dac.get_decoded_signal(z_s, orig_len)
            est_wavs.append(wav_hat)

        est_sources = torch.stack(est_wavs, dim=1).squeeze(2)
        T_est = est_sources.shape[-1]

        sources_squeezed = sources.squeeze(2)
        sources_flat = sources_squeezed.reshape(B * S, 1, T_orig)

        src_enc, src_orig_len = dac.get_encoded_features(sources_flat)
        src_q, _, _, _, _ = dac.get_quantized_features(src_enc)
        codec_src_flat = dac.get_decoded_signal(src_q, src_orig_len)
        
        T_codec = codec_src_flat.shape[-1]
        codec_sources = codec_src_flat.view(B, S, T_codec)

        # Align lengths
        min_len = min(T_est, T_codec, T_orig)
        est_sources_aligned = est_sources[..., :min_len]
        codec_sources_aligned = codec_sources[..., :min_len]

        loss_wave = pit_si_sdr_loss(est_sources_aligned, codec_sources_aligned)
        
        # Optional latent loss
        loss_latent = torch.tensor(0.0, device=device)
        if latent_loss_weight > 0:
            _, T_lat, C_lat = z_mix.shape
            target_lat = src_q.view(B, S, C_lat, -1)
            T_lat_target = target_lat.shape[-1]
            T_lat_est = sep_lat.shape[-1]
            min_lat_len = min(T_lat_target, T_lat_est)
            
            sep_lat_aligned = sep_lat[..., :min_lat_len]
            target_lat_aligned = target_lat[..., :min_lat_len]
            loss_latent = pit_latent_loss(sep_lat_aligned, target_lat_aligned)
        
        loss = loss_wave + latent_loss_weight * loss_latent

        batch_loss = float(loss.item())
        total_loss += batch_loss
        num_batches_done += 1
        pbar.set_postfix(val_loss=total_loss / num_batches_done)

    return total_loss / max(1, num_batches_done)


# =========================
#   MAIN
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--valid_csv", type=str, required=True)
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--n_layer", type=int, default=8)
    ap.add_argument("--n_embd", type=int, required=True,
                   help="Working dimensions of RWKV model")
    ap.add_argument("--head_mode", type=str, default="residual",
                    choices=["residual", "mask", "softmax_mask", "direct"])
    ap.add_argument("--n_groups", type=int, required=True,
                    help="Number of groups for grouped Bi-RWKV")
    ap.add_argument("--log_dir", type=str, default=None)
    ap.add_argument("--seg_sec", type=float, required=True)
    ap.add_argument("--head_hidden", type=int, required=True)
    ap.add_argument("--subset_frac", type=float, default=None)
    
    # New: latent loss
    ap.add_argument("--latent_loss_weight", type=float, required=True,
                    help="Weight for auxiliary latent-domain loss (0 to disable)")
    
    # LR scheduler
    ap.add_argument("--lr_scheduler", action="store_true")
    ap.add_argument("--lr_scheduler_start_epoch", type=int, default=1)
    ap.add_argument("--lr_scheduler_patience", type=int, default=5)
    ap.add_argument("--lr_scheduler_factor", type=float, default=0.5)
    ap.add_argument("--lr_scheduler_min_lr", type=float, default=1e-6)
    
    # Early stopping
    ap.add_argument("--early_stop", action="store_true")
    ap.add_argument("--early_stop_patience", type=int, default=20)
    
    ap.add_argument("--resume_checkpoint", type=str, default=None)
    ap.add_argument("--benchmark_epoch_time", action="store_true")
    ap.add_argument("--check_gradients", action="store_true",
                    help="Check gradient flow on first batch")

    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Using device: {device}")

    tb_log_dir = args.log_dir or os.path.join(args.save_dir, "tb")
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)

    train_ds = Libri2MixDataset(args.train_csv, sample_rate=args.sample_rate, segment_seconds=args.seg_sec, subset_frac=args.subset_frac)
    valid_ds = Libri2MixDataset(args.valid_csv, sample_rate=args.sample_rate, segment_seconds=args.seg_sec, subset_frac=args.subset_frac)

#    inspect_dataset(train_ds, "train_ds", n=2)
#    inspect_dataset(valid_ds, "val_ds",   n=2)

#    check_mixture_consistency(train_ds, idx=0)
#    check_mixture_consistency(train_ds, idx=1)

#    print(len(train_ds), len(valid_ds))
    # you can check that they’re ~10% of full size
#    sys.exit()
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate_fn, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, collate_fn=collate_fn, drop_last=True)

    dac = DACWrapper(input_sample_rate=args.sample_rate, DAC_model_path=None,
                     DAC_sample_rate=16000, Freeze=True)
    dac.model.to(device)
    dac.dac_sampler.to(device)
    dac.org_sampler.to(device)

    mix_example, _ = next(iter(train_loader))
    mix_example = mix_example.to(device)
    with torch.no_grad():
        z_enc, _ = dac.get_encoded_features(mix_example)
    _, C_lat, _ = z_enc.shape
    print(f"[INFO] DAC latent channels: {C_lat}")

    model = build_rwkv7_separator(
        n_embd=args.n_embd,
        codec_dim=C_lat,
        n_layer=args.n_layer,
        num_sources=2,
        head_mode=args.head_mode,
        enforce_bf16=False,
        head_hidden=args.head_hidden,
        n_groups=args.n_groups,
    ).to(device)

    results = quick_analysis(model,
                             input_type='latent', # DAC latent domain
                             sequence_length=150, # Latent sequence length
                             latent_dim=C_lat,
                            )

    if args.benchmark_epoch_time:
        estimate_epoch_time(
            model=model,
            sequence_length=500,
            latent_dim=C_lat,
            batch_size=args.batch_size,
            steps_per_epoch=len(train_loader),
            device=device,
            warmup_steps=5,
            verbose=True,
        )
    
#    sys.exit()
    
    num_params = sum(p.numel() for p in model.parameters())
    
    print(f"[INFO] Model parameters: {num_params:,}")
    print(f"[INFO] Head mode: {args.head_mode}")
    print(f"[INFO] Number of layers: {args.n_layer}")
    print(f"[INFO] Number of groups: {args.n_groups}")
    print(f"[INFO] Latent loss weight: {args.latent_loss_weight}")
    print(f"[INFO] Hidden head size: {args.head_hidden}")
    print(f"[INFO] RWKV model size: {args.n_embd}")
    print(f"[INFO] Segment length: {args.seg_sec}")
    print(f"[INFO] Latent loss weight: {args.latent_loss_weight}")

#    sys.exit()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    scheduler = None
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_scheduler_factor,
            patience=args.lr_scheduler_patience, min_lr=args.lr_scheduler_min_lr)

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

    # ============================================
    # INITIALIZE COMPLEXITY MEASUREMENT
    # ============================================
    complexity_log_path = os.path.join(args.save_dir, "complexity_codec_domain.txt")
    complexity_meter = ComplexityMeasurement(model, device, log_file=complexity_log_path)
    
    # Measure training complexity
    _, T_lat_example, _ = z_enc.shape
    complexity_meter.measure_model_complexity(input_shape=(args.batch_size, T_lat_example, C_lat))


    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            epoch, model, dac, train_loader, optimizer, device,
            grad_clip=args.grad_clip, writer=writer,
            latent_loss_weight=args.latent_loss_weight,
            complexity_meter=complexity_meter,
            check_gradients=(args.check_gradients and epoch == start_epoch),
        )

        val_loss = validate(epoch, model, dac, valid_loader, device,
                           latent_loss_weight=args.latent_loss_weight)

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
                "latent_channels": C_lat,
            }, ckpt_path)
            print(f"  ✅ Saved: {ckpt_path}")
        else:
            epochs_no_improve += 1

        if scheduler is not None and epoch >= args.lr_scheduler_start_epoch:
            scheduler.step(val_loss)

        if args.early_stop and epochs_no_improve >= args.early_stop_patience:
            print(f"[EARLY STOP] No improvement for {epochs_no_improve} epochs")
            break

    print(f"[DONE] Best val loss: {best_val:.4f}")
    writer.close()


if __name__ == "__main__":
    main()