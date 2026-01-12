#!/usr/bin/env python3
"""
RWKV-v7 Time-Domain Speech Separation Inference Script - COMPLETE FIXED VERSION

All fixes included:
- SI-SDRi and SDRi improvement metrics
- PESQ/STOI shape handling fixes
- Proper error reporting
- Auto-detection of PESQ mode (NB/WB)

This script evaluates the time-domain separator trained with train_rwkv_TDtest.py.

Metrics included:
1. SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
2. SI-SDRi (SI-SDR Improvement over mixture)
3. SDRi (SDR Improvement over mixture)
4. PESQ (Perceptual Evaluation of Speech Quality)
5. STOI (Short-Time Objective Intelligibility)
6. NISQA (Non-Intrusive Speech Quality Assessment)
7. DNSMOS (Deep Noise Suppression Mean Opinion Score)

Usage:
    python infer_rwkv_TDtest_FIXED.py \
        --csv /path/to/test.csv \
        --checkpoint /path/to/best.pt \
        --encoder_path /path/to/encoder.pt \
        --decoder_path /path/to/decoder.pt \
        --sample_rate 8000 \
        --batch_size 4 \
        --enable_pesq --enable_stoi --enable_nisqa --enable_dnsmos
"""

import argparse
import csv
import os
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf
import numpy as np

# Optional metrics
try:
    import pesq as pesq_fn
    _HAS_PESQ = True
except ImportError:
    pesq_fn = None
    _HAS_PESQ = False

try:
    from pystoi import stoi as stoi_fn
    _HAS_STOI = True
except ImportError:
    stoi_fn = None
    _HAS_STOI = False

try:
    from torchmetrics.functional.audio.dnsmos import (
        deep_noise_suppression_mean_opinion_score as dnsmos_fn,
    )
    _HAS_DNSMOS = True
except ImportError:
    dnsmos_fn = None
    _HAS_DNSMOS = False

try:
    from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment as nisqa_fn
    _HAS_NISQA = True
except ImportError:
    nisqa_fn = None
    _HAS_NISQA = False

# RWKV settings
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
        if x.dim() == 4:  # [B, S, C, T_enc]
            B, S, C, T_enc = x.shape
            x = x.reshape(B * S, C, T_enc)
            x = self.transconv1d(x)
            if x.size(1) == 1:
                x = x.squeeze(1)
            x = x.view(B, S, -1)
        else:  # [B, C, T_enc]
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
        # Decoder should be in eval mode but gradients can flow through
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

class Libri2MixEvalDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        sample_rate: int = 8000,
        max_samples: int = None,
        subset_frac: float = None,
        subset_seed: int = 42,
    ):
        super().__init__()
        self.sample_rate = sample_rate
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
            wav = torch.from_numpy(data.T).mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        mix = self._load_mono(row["mix_path"])
        s1 = self._load_mono(row["s1_path"])
        s2 = self._load_mono(row["s2_path"])

        T = min(mix.size(-1), s1.size(-1), s2.size(-1))
        mix = mix[..., :T]
        s1 = s1[..., :T]
        s2 = s2[..., :T]

        sources = torch.stack([s1, s2], dim=0)
        return {"mix": mix, "sources": sources, "row": row}


def collate_fn(batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """Collate batch with padding to max length"""
    lengths = [b["mix"].shape[-1] for b in batch]
    T_max = max(lengths)

    mix_list, sources_list, rows = [], [], []
    for b in batch:
        mix = b["mix"]
        sources = b["sources"]
        pad_T = T_max - mix.shape[-1]

        if pad_T > 0:
            mix = F.pad(mix, (0, pad_T))
            sources = F.pad(sources, (0, pad_T))

        mix_list.append(mix)
        sources_list.append(sources)
        rows.append(b["row"])

    return torch.stack(mix_list, 0), torch.stack(sources_list, 0), rows


# =========================
#   SI-SDR, SDR + PIT
# =========================

def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """SI-SDR in dB. est, ref: [B, T] → returns [B]"""
    ref_zm = ref - ref.mean(dim=-1, keepdim=True)
    est_zm = est - est.mean(dim=-1, keepdim=True)

    dot = (est_zm * ref_zm).sum(dim=-1, keepdim=True)
    ref_energy = (ref_zm ** 2).sum(dim=-1, keepdim=True) + eps
    s_target = dot / ref_energy * ref_zm

    e_noise = est_zm - s_target
    s_target_energy = (s_target ** 2).sum(dim=-1) + eps
    e_noise_energy = (e_noise ** 2).sum(dim=-1) + eps

    return 10 * torch.log10(s_target_energy / e_noise_energy)


def sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Classic SDR (non-scale-invariant) in dB.
    est, ref: [B, T] → returns [B]
    
    SDR = 10 * log10( ||ref||^2 / ||ref - est||^2 )
    """
    err = ref - est
    num = (ref ** 2).sum(dim=-1) + eps
    den = (err ** 2).sum(dim=-1) + eps
    return 10 * torch.log10(num / den)


def pit_si_sdr_with_perm(est_sources: torch.Tensor,
                         true_sources: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns both SI-SDR and the best permutation indices.
    
    Args:
        est_sources: [B, S, T]
        true_sources: [B, S, T]
    
    Returns:
        si_sdr_scores: [B] - average SI-SDR per utterance
        best_perm: [B] - 0 if (est0→s0, est1→s1), 1 if (est0→s1, est1→s0)
    """
    B, S, T = est_sources.shape
    assert S == 2

    est1, est2 = est_sources[:, 0], est_sources[:, 1]
    s1, s2 = true_sources[:, 0], true_sources[:, 1]

    # Perm 1: est0→s0, est1→s1
    sdr11 = si_sdr(est1, s1)
    sdr22 = si_sdr(est2, s2)
    sum_perm1 = sdr11 + sdr22

    # Perm 2: est0→s1, est1→s0
    sdr12 = si_sdr(est1, s2)
    sdr21 = si_sdr(est2, s1)
    sum_perm2 = sdr12 + sdr21

    # Best permutation (0 or 1)
    best_perm = (sum_perm2 > sum_perm1).long()
    
    # Average SI-SDR for best perm
    avg_sdr = torch.where(
        best_perm == 0,
        (sdr11 + sdr22) / 2.0,
        (sdr12 + sdr21) / 2.0
    )

    return avg_sdr, best_perm


def reorder_sources(est_sources: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """
    Reorder estimated sources according to permutation.
    
    Args:
        est_sources: [B, S, T]
        perm: [B] - 0 or 1
    
    Returns:
        reordered: [B, S, T] - sources in correct order
    """
    B, S, T = est_sources.shape
    assert S == 2
    
    reordered = torch.zeros_like(est_sources)
    
    for i in range(B):
        if perm[i] == 0:
            # No swap: est0→s0, est1→s1
            reordered[i, 0] = est_sources[i, 0]
            reordered[i, 1] = est_sources[i, 1]
        else:
            # Swap: est0→s1, est1→s0
            reordered[i, 0] = est_sources[i, 1]
            reordered[i, 1] = est_sources[i, 0]
    
    return reordered


# =========================
#   EVALUATION
# =========================

def evaluate(
    model: nn.Module,
    sepformer_processor: SepFormerProcessor,
    loader: DataLoader,
    device: torch.device,
    input_sample_rate: int = 8000,
    save_audio_dir: Optional[str] = None,
    enable_pesq: bool = False,
    enable_stoi: bool = False,
    enable_dnsmos: bool = False,
    enable_nisqa: bool = False,
):
    """Evaluate time-domain model with comprehensive metrics"""
    
    # Check metric availability
    if enable_pesq and not _HAS_PESQ:
        print("[WARN] PESQ requested but not available. Install: pip install pesq")
        enable_pesq = False
    if enable_stoi and not _HAS_STOI:
        print("[WARN] STOI requested but not available. Install: pip install pystoi")
        enable_stoi = False
    if enable_dnsmos and not _HAS_DNSMOS:
        print("[WARN] DNSMOS requested but not available. Install: pip install torchmetrics[audio]")
        enable_dnsmos = False
    if enable_nisqa and not _HAS_NISQA:
        print("[WARN] NISQA requested but not available. Install: pip install torchmetrics[audio]")
        enable_nisqa = False
    
    # Initialize NISQA model if requested
    nisqa_model = None
    if enable_nisqa:
        try:
            nisqa_model = nisqa_fn(input_sample_rate).to(device)
            nisqa_model.eval()
        except Exception as e:
            print(f"[WARN] Failed to initialize NISQA: {e}")
            enable_nisqa = False
    
    if save_audio_dir:
        os.makedirs(save_audio_dir, exist_ok=True)
    
    model.eval()
    sepformer_processor.eval()
    
    # Metric storage
    all_si_sdr_clean = []
    all_si_sdr_mix = []    # For mixture SI-SDR
    all_sdr_mix = []       # For mixture SDR
    all_sdr_clean = []     # For separated SDR
    all_pesq = []
    all_stoi = []
    all_dnsmos_sig = []
    all_dnsmos_bak = []
    all_dnsmos_ovrl = []
    all_nisqa = []
    
    pesq_failures = 0
    stoi_failures = 0
    total_speaker_samples = 0
    
    print("\n" + "=" * 60)
    print("STARTING EVALUATION")
    print("=" * 60)
    print(f"[DEBUG] Metric flags:")
    print(f"  PESQ enabled: {enable_pesq}")
    print(f"  STOI enabled: {enable_stoi}")
    print(f"  DNSMOS enabled: {enable_dnsmos}")
    print(f"  NISQA enabled: {enable_nisqa}")
    
    with torch.no_grad():
        for batch_idx, (mix_wav, src_wav, rows) in enumerate(loader, 1):
            B, _, T_wav = mix_wav.shape
            S = src_wav.shape[1]
            
            mix_wav = mix_wav.to(device)
            src_wav = src_wav.to(device)
            
            # ---------- Forward pass ----------
            # 1. Encode mixture
            mix_feat = sepformer_processor.encode(mix_wav)  # [B, C, T_enc]
            
            # 2. Transpose for RWKV: [B, C, T_enc] → [B, T_enc, C]
            mix_feat_t = mix_feat.transpose(1, 2)
            
            # 3. RWKV separation
            model_output = model(mix_feat_t)
            
            # Handle different return types (tensor or tuple)
            if isinstance(model_output, tuple):
                sep_feat_t = model_output[0]  # Take only output, ignore state
            else:
                sep_feat_t = model_output
            
            # Debug: Check output shape on first batch
            if batch_idx == 1:
                print(f"[DEBUG] Model input shape: {mix_feat_t.shape}")
                print(f"[DEBUG] Model output shape: {sep_feat_t.shape}")
            
            # 4. Handle model output shape
            # Expected: either [B, T_enc, S*C] or [B, T_enc, S, C]
            if sep_feat_t.dim() == 4:
                # Model already separated: [B, T_enc, S, C] → [B, S, C, T_enc]
                sep_feat = sep_feat_t.permute(0, 2, 3, 1)
            elif sep_feat_t.dim() == 3:
                # Model output: [B, T_enc, S*C] → reshape → [B, S, C, T_enc]
                B_out, T_enc, sep_dim = sep_feat_t.shape
                n_channels = mix_feat.shape[1]
                sep_feat = sep_feat_t.view(B_out, T_enc, S, n_channels).permute(0, 2, 3, 1)
            else:
                raise ValueError(f"Unexpected model output shape: {sep_feat_t.shape}")
            
            # 5. Decode separated features
            sep_wav = sepformer_processor.decode(sep_feat, original_length=T_wav)  # [B, S, T_wav]
            
            # ---------- Align lengths ----------
            T_min = min(sep_wav.size(-1), src_wav.size(-1))
            sep_wav = sep_wav[..., :T_min]
            src_wav = src_wav[..., :T_min]
            
            # ---------- Compute mixture baselines (for improvement metrics) ----------
            mix_wav_aligned = mix_wav[..., :T_min].squeeze(1)  # [B, T_min]
            mix_for_sources = mix_wav_aligned.unsqueeze(1).expand(-1, S, -1)  # [B, S, T_min]
            
            # SI-SDR of mixture vs true sources
            mix_si_sdr_scores, _ = pit_si_sdr_with_perm(mix_for_sources, src_wav)
            all_si_sdr_mix.append(mix_si_sdr_scores)
            
            # SDR of mixture (PIT-based)
            mix1, mix2 = mix_for_sources[:, 0], mix_for_sources[:, 1]
            s1, s2 = src_wav[:, 0], src_wav[:, 1]
            
            sdr11_mix = sdr(mix1, s1)
            sdr22_mix = sdr(mix2, s2)
            sdr12_mix = sdr(mix1, s2)
            sdr21_mix = sdr(mix2, s1)
            
            sum_perm1_mix = sdr11_mix + sdr22_mix
            sum_perm2_mix = sdr12_mix + sdr21_mix
            best_perm_sdr_mix = (sum_perm2_mix > sum_perm1_mix).long()
            
            avg_sdr_mix = torch.where(
                best_perm_sdr_mix == 0,
                (sdr11_mix + sdr22_mix) / 2.0,
                (sdr12_mix + sdr21_mix) / 2.0
            )
            all_sdr_mix.append(avg_sdr_mix)
            
            # ---------- SI-SDR with PIT ----------
            si_sdr_scores, best_perm = pit_si_sdr_with_perm(sep_wav, src_wav)
            all_si_sdr_clean.append(si_sdr_scores)
            
            # ---------- SDR for separated signals ----------
            # Compute SDR using same PIT permutation as SI-SDR
            est1, est2 = sep_wav[:, 0], sep_wav[:, 1]
            
            sdr11_sep = sdr(est1, s1)
            sdr22_sep = sdr(est2, s2)
            sdr12_sep = sdr(est1, s2)
            sdr21_sep = sdr(est2, s1)
            
            avg_sdr_clean = torch.where(
                best_perm == 0,
                (sdr11_sep + sdr22_sep) / 2.0,
                (sdr12_sep + sdr21_sep) / 2.0
            )
            all_sdr_clean.append(avg_sdr_clean)
            
            # Reorder sources according to best permutation
            est_reordered = reorder_sources(sep_wav, best_perm)
            
            # ---------- Additional metrics ----------
            batch_pesq = []
            batch_stoi = []
            batch_dnsmos_sig = []
            batch_dnsmos_bak = []
            batch_dnsmos_ovrl = []
            batch_nisqa = []
            
            total_speaker_samples += B * S
            
            # Process each utterance and speaker
            for i in range(B):
                for s_idx in range(S):
                    # Get reference and estimated signals
                    ref = src_wav[i, s_idx].cpu().numpy()
                    deg = est_reordered[i, s_idx].cpu().numpy()
                    
                    # Ensure 1D arrays (flatten any extra dimensions)
                    ref = np.squeeze(ref)
                    deg = np.squeeze(deg)
                    
                    # Double-check they're 1D
                    if ref.ndim > 1:
                        ref = ref.flatten()
                    if deg.ndim > 1:
                        deg = deg.flatten()
                    
                    # PESQ
                    if enable_pesq:
                        try:
                            # Auto-select mode based on sample rate
                            mode = 'nb' if input_sample_rate == 8000 else 'wb'
                            score = pesq_fn.pesq(input_sample_rate, ref, deg, mode)
                            batch_pesq.append(score)
                        except Exception as e:
                            pesq_failures += 1
                            if pesq_failures == 1:
                                print(f"[DEBUG] PESQ error: {e}")
                                print(f"[DEBUG] ref shape: {ref.shape}, deg shape: {deg.shape}")
                    
                    # STOI
                    if enable_stoi:
                        try:
                            score = float(stoi_fn(ref, deg, input_sample_rate, extended=False))
                            batch_stoi.append(score)
                        except Exception as e:
                            stoi_failures += 1
                            if stoi_failures == 1:
                                print(f"[DEBUG] STOI error: {e}")
                                print(f"[DEBUG] ref shape: {ref.shape}, deg shape: {deg.shape}")
                    
                    # DNSMOS
                    if enable_dnsmos:
                        try:
                            wav_t = torch.from_numpy(deg).to(device)
                            scores = dnsmos_fn(wav_t, fs=input_sample_rate, personalized=False)
                            batch_dnsmos_sig.append(float(scores[0].item()))
                            batch_dnsmos_bak.append(float(scores[1].item()))
                            batch_dnsmos_ovrl.append(float(scores[2].item()))
                        except Exception as e:
                            if batch_idx == 1 and i == 0 and s_idx == 0:
                                print(f"[DEBUG] DNSMOS error: {e}")
                    
                    # NISQA
                    if enable_nisqa and nisqa_model is not None:
                        try:
                            wav_t = torch.from_numpy(deg).unsqueeze(0).to(device)
                            nisqa_score = nisqa_model(wav_t, input_sample_rate)
                            batch_nisqa.append(float(nisqa_score.item()))
                        except Exception as e:
                            if batch_idx == 1 and i == 0 and s_idx == 0:
                                print(f"[DEBUG] NISQA error: {e}")
            
            if batch_pesq:
                all_pesq.extend(batch_pesq)
            if batch_stoi:
                all_stoi.extend(batch_stoi)
            if batch_dnsmos_sig:
                all_dnsmos_sig.extend(batch_dnsmos_sig)
                all_dnsmos_bak.extend(batch_dnsmos_bak)
                all_dnsmos_ovrl.extend(batch_dnsmos_ovrl)
            if batch_nisqa:
                all_nisqa.extend(batch_nisqa)
            
            # ---------- Save audio ----------
            if save_audio_dir:
                est_save = est_reordered.cpu().numpy()
                for i in range(B):
                    base = os.path.splitext(os.path.basename(rows[i]["mix_path"]))[0]
                    for s_idx in range(S):
                        out_path = os.path.join(save_audio_dir, f"{base}_spk{s_idx+1}.wav")
                        sf.write(out_path, est_save[i, s_idx], samplerate=input_sample_rate)
            
            # ---------- Batch summary ----------
            msg = f"[BATCH {batch_idx:04d}] SI-SDR={si_sdr_scores.mean().item():.2f} dB"
            if batch_pesq:
                msg += f", PESQ={np.mean(batch_pesq):.3f}"
            if batch_stoi:
                msg += f", STOI={np.mean(batch_stoi):.3f}"
            if batch_nisqa:
                msg += f", NISQA={np.mean(batch_nisqa):.3f}"
            print(msg)
    
    # ---------- Final summary ----------
    all_si_sdr_clean = torch.cat(all_si_sdr_clean, dim=0)
    all_si_sdr_mix = torch.cat(all_si_sdr_mix, dim=0)
    all_sdr_mix = torch.cat(all_sdr_mix, dim=0)
    all_sdr_clean = torch.cat(all_sdr_clean, dim=0)
    
    # Compute improvement metrics
    si_sdri = all_si_sdr_clean - all_si_sdr_mix
    sdri = all_sdr_clean - all_sdr_mix
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total utterances: {all_si_sdr_clean.numel()}")
    print(f"Mean SI-SDR: {all_si_sdr_clean.mean().item():.3f} dB")
    print(f"Median SI-SDR: {all_si_sdr_clean.median().item():.3f} dB")
    print(f"Std SI-SDR: {all_si_sdr_clean.std().item():.3f} dB")
    
    print(f"\nImprovement Metrics:")
    print(f"Mean SI-SDRi: {si_sdri.mean().item():.3f} dB (improvement over mixture)")
    print(f"Median SI-SDRi: {si_sdri.median().item():.3f} dB")
    print(f"Mean SDRi: {sdri.mean().item():.3f} dB (improvement over mixture)")
    print(f"Median SDRi: {sdri.median().item():.3f} dB")
    
    print(f"\nBaseline (Mixture) Performance:")
    print(f"Mean SI-SDR (mixture): {all_si_sdr_mix.mean().item():.3f} dB")
    print(f"Mean SDR (mixture): {all_sdr_mix.mean().item():.3f} dB")
    
    if all_pesq:
        print(f"\nMean PESQ: {np.mean(all_pesq):.3f}")
        print(f"Median PESQ: {np.median(all_pesq):.3f}")
        if pesq_failures > 0:
            print(f"  (PESQ failures: {pesq_failures}/{total_speaker_samples})")
    
    if all_stoi:
        print(f"Mean STOI: {np.mean(all_stoi):.3f}")
        print(f"Median STOI: {np.median(all_stoi):.3f}")
        if stoi_failures > 0:
            print(f"  (STOI failures: {stoi_failures}/{total_speaker_samples})")
    
    if all_dnsmos_ovrl:
        print(f"Mean DNSMOS (SIG): {np.mean(all_dnsmos_sig):.3f}")
        print(f"Mean DNSMOS (BAK): {np.mean(all_dnsmos_bak):.3f}")
        print(f"Mean DNSMOS (OVRL): {np.mean(all_dnsmos_ovrl):.3f}")
    
    if all_nisqa:
        print(f"Mean NISQA: {np.mean(all_nisqa):.3f}")
        print(f"Median NISQA: {np.median(all_nisqa):.3f}")
    
    print("=" * 60 + "\n")


# =========================
#   MAIN
# =========================

def main():
    ap = argparse.ArgumentParser(description="RWKV-v7 Time-Domain Inference - FIXED")
    ap.add_argument("--csv", type=str, required=True, help="Path to test CSV")
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    ap.add_argument("--encoder_path", type=str, required=True, help="Path to frozen encoder")
    ap.add_argument("--decoder_path", type=str, required=True, help="Path to frozen decoder")
    ap.add_argument("--sample_rate", type=int, default=8000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_audio_dir", type=str, default=None)
    ap.add_argument("--subset_frac", type=float, default=None)
    ap.add_argument("--enable_pesq", action="store_true")
    ap.add_argument("--enable_stoi", action="store_true")
    ap.add_argument("--enable_dnsmos", action="store_true")
    ap.add_argument("--enable_nisqa", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    # Get config from checkpoint
    ckpt_config = ckpt.get("config", {})
    
    print("\n[INFO] Checkpoint configuration:")
    for k, v in ckpt_config.items():
        if k not in ['train_csv', 'valid_csv', 'save_dir', 'log_dir', 
                     'encoder_path', 'decoder_path']:
            print(f"  {k}: {v}")
    
    # Initialize SepFormer processor
    print(f"\n[INFO] Initializing SepFormer processor...")
    if not os.path.exists(args.encoder_path):
        raise FileNotFoundError(f"Encoder not found: {args.encoder_path}")
    if not os.path.exists(args.decoder_path):
        raise FileNotFoundError(f"Decoder not found: {args.decoder_path}")
    
    sepformer_processor = SepFormerProcessor(
        encoder_path=args.encoder_path,
        decoder_path=args.decoder_path
    ).to(device)
    
    # Determine feature dimensions from encoder
    print(f"\n[INFO] Determining feature dimensions...")
    test_wav = torch.randn(1, 1, args.sample_rate).to(device)  # 1 second test
    with torch.no_grad():
        test_feat = sepformer_processor.encode(test_wav)
    _, codec_dim, T_enc = test_feat.shape
    print(f"[INFO] Encoder output: C={codec_dim}, example T_enc={T_enc}")
    
    # Build model
    n_layer = ckpt_config.get("n_layer", 8)
    n_embd = ckpt_config.get("n_embd")
    head_hidden = ckpt_config.get("head_hidden")
    head_mode = ckpt_config.get("head_mode", "direct")
    n_groups = ckpt_config.get("n_groups", 2)
    
    if n_embd is None or head_hidden is None:
        raise RuntimeError(
            "Checkpoint missing model parameters (n_embd, head_hidden). "
            "Check checkpoint config."
        )
    
    print(f"\n[INFO] Building model...")
    model = build_rwkv7_separator(
        n_embd=n_embd,
        codec_dim=codec_dim,
        n_layer=n_layer,
        num_sources=2,
        head_mode=head_mode,
        enforce_bf16=False,
        head_hidden=head_hidden,
        n_groups=n_groups,
    ).to(device)
    
    # Load model weights (clean profiler keys if present)
    raw_state = ckpt["model_state_dict"]
    clean_state = {}
    skipped = []
    
    for k, v in raw_state.items():
        if (
            k.endswith("total_ops")
            or k.endswith("total_params")
            or ".total_ops" in k
            or ".total_params" in k
        ):
            skipped.append(k)
        else:
            clean_state[k] = v
    
    if skipped:
        print(f"[INFO] Cleaned {len(skipped)} profiler keys from checkpoint")
    
    model.load_state_dict(clean_state, strict=True)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {num_params:,}")
    print(f"[INFO] Head mode: {head_mode}, n_groups: {n_groups}")
    
    # Load dataset
    print(f"\n[INFO] Loading dataset: {args.csv}")
    eval_ds = Libri2MixEvalDataset(
        args.csv,
        sample_rate=args.sample_rate,
        subset_frac=args.subset_frac
    )
    print(f"[INFO] Total samples: {len(eval_ds)}")
    
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        drop_last=False,
    )
    
    # Evaluate
    evaluate(
        model=model,
        sepformer_processor=sepformer_processor,
        loader=eval_loader,
        device=device,
        input_sample_rate=args.sample_rate,
        save_audio_dir=args.save_audio_dir,
        enable_pesq=args.enable_pesq,
        enable_stoi=args.enable_stoi,
        enable_dnsmos=args.enable_dnsmos,
        enable_nisqa=args.enable_nisqa,
    )


if __name__ == "__main__":
    main()