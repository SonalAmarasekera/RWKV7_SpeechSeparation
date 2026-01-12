#!/usr/bin/env python3
"""
RWKV-v7 TF-Domain Speech Separation Inference Script (STFT-based)

This script evaluates the TF-domain separator trained with train_rwkv_TFtest.py.

Metrics included:
1. SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
2. PESQ (Perceptual Evaluation of Speech Quality)
3. STOI (Short-Time Objective Intelligibility)
4. NISQA (Non-Intrusive Speech Quality Assessment)
5. DNSMOS (Deep Noise Suppression Mean Opinion Score)

Key features:
- Fixed STFT/iSTFT transform (no learning)
- Proper PIT permutation for all metrics
- Consistent permutation across all metrics
- Length alignment fixes
- Handles both 3D and 4D model outputs

Model Output Formats Supported:
- 4D: [B, T_tf, S, F] - Sources already separated (most common)
- 3D: [B, T_tf, S*F] - Sources concatenated (alternative)

Usage:
    python infer_rwkv_TFtest.py \
        --csv /path/to/test.csv \
        --checkpoint /path/to/best.pt \
        --sample_rate 16000 \
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
#   STFT PROCESSOR
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

class Libri2MixEvalDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        sample_rate: int = 16000,
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
#   SI-SDR + PIT
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
    stft_processor: STFTProcessor,
    loader: DataLoader,
    device: torch.device,
    input_sample_rate: int = 16000,
    save_audio_dir: Optional[str] = None,
    enable_pesq: bool = False,
    enable_stoi: bool = False,
    enable_dnsmos: bool = False,
    enable_nisqa: bool = False,
):
    """Evaluate TF-domain model with comprehensive metrics"""
    
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
    
    # Metric storage
    all_si_sdr_clean = []
    all_si_sdr_mix = []
    all_sdr_mix = []
    all_sdr_clean = []
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
    
    with torch.no_grad():
        for batch_idx, (mix_wav, src_wav, rows) in enumerate(loader, 1):
            B, _, T_wav = mix_wav.shape
            S = src_wav.shape[1]
            
            mix_wav = mix_wav.to(device)
            src_wav = src_wav.to(device)
            
            # ---------- Forward pass ----------
            # 1. STFT of mixture
            mix_mag, mix_phase = stft_processor.stft(mix_wav)  # [B, F, T_tf]
            
            # 2. Transpose for RWKV: [B, F, T_tf] → [B, T_tf, F]
            mix_mag_t = mix_mag.transpose(1, 2)
            
            # 3. RWKV separation
            model_output = model(mix_mag_t)
            
            # Handle different return types (tensor or tuple)
            if isinstance(model_output, tuple):
                sep_mag_t = model_output[0]  # Take only the output, ignore state
            else:
                sep_mag_t = model_output
            
            # Debug: Check output shape
            if batch_idx == 1:
                print(f"[DEBUG] Model input shape: {mix_mag_t.shape}")
                print(f"[DEBUG] Model output shape: {sep_mag_t.shape}")
            
            # 4. Handle model output shape
            # Expected: either [B, T_tf, S*F] or [B, T_tf, S, F]
            if sep_mag_t.dim() == 4:
                # Model already separated into [B, T_tf, S, F]
                # Just permute to [B, S, F, T_tf]
                sep_mag = sep_mag_t.permute(0, 2, 3, 1)  # [B, T_tf, S, F] → [B, S, F, T_tf]
            elif sep_mag_t.dim() == 3:
                # Model output is [B, T_tf, S*F], need to reshape
                B_out, T_tf, sep_dim = sep_mag_t.shape
                n_freq = mix_mag.shape[1]
                sep_mag = sep_mag_t.view(B_out, T_tf, S, n_freq).permute(0, 2, 3, 1)
            else:
                raise ValueError(f"Unexpected model output shape: {sep_mag_t.shape}")
            
            # 5. Apply ReLU to ensure non-negative magnitudes
            sep_mag = F.relu(sep_mag)
            
            # 6. iSTFT with mixture phase
            sep_wav = stft_processor.istft(sep_mag, mix_phase, length=T_wav)  # [B, S, T_wav]
            
            # ---------- Align lengths ----------
            T_min = min(sep_wav.size(-1), src_wav.size(-1))
            sep_wav = sep_wav[..., :T_min]
            src_wav = src_wav[..., :T_min]

            # Compute mixture baselines (for improvement metrics)
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
            
            # ============ SDR FOR SEPARATED (FIXED) ============
            # Compute SDR using same PIT permutation as SI-SDR
            est1, est2 = sep_wav[:, 0], sep_wav[:, 1]
            
            # Use the same best_perm from SI-SDR PIT
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
                    # ============ FIXED: Ensure 1D arrays ============
                    # Get reference and estimated signals
                    ref = src_wav[i, s_idx].cpu().numpy()
                    deg = est_reordered[i, s_idx].cpu().numpy()
                    
                    # Ensure 1D arrays (flatten any extra dimensions)
                    ref = np.squeeze(ref)  # Remove any singleton dimensions
                    deg = np.squeeze(deg)  # Remove any singleton dimensions
                    
                    # Double-check they're 1D
                    if ref.ndim > 1:
                        ref = ref.flatten()
                    if deg.ndim > 1:
                        deg = deg.flatten()
                    # =================================================
                    
                    # PESQ
                    if enable_pesq:
                        try:
                            # Auto-select mode based on sample rate
                            mode = 'nb' if input_sample_rate == 8000 else 'wb'
                            score = pesq_fn.pesq(input_sample_rate, ref, deg, mode)
                            batch_pesq.append(score)
                        except Exception as e:
                            pesq_failures += 1
                            if pesq_failures == 1:  # Print first error for debugging
                                print(f"[DEBUG] PESQ error: {e}")
                                print(f"[DEBUG] ref shape: {ref.shape}, deg shape: {deg.shape}")
                    
                    # STOI
                    if enable_stoi:
                        try:
                            score = float(stoi_fn(ref, deg, input_sample_rate, extended=False))
                            batch_stoi.append(score)
                        except Exception as e:
                            stoi_failures += 1
                            if stoi_failures == 1:  # Print first error for debugging
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
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"[DEBUG] Metric flags:")
    print(f"  PESQ enabled: {enable_pesq}")
    print(f"  STOI enabled: {enable_stoi}")
    print(f"  DNSMOS enabled: {enable_dnsmos}")
    print(f"  NISQA enabled: {enable_nisqa}")
    print(f"[DEBUG] Library availability:")
    print(f"  PESQ available: {_HAS_PESQ}")
    print(f"  STOI available: {_HAS_STOI}")
    print(f"  DNSMOS available: {_HAS_DNSMOS}")
    print(f"  NISQA available: {_HAS_NISQA}")
    
    print(f"Total utterances: {all_si_sdr_clean.numel()}")
    print(f"Mean SI-SDR: {all_si_sdr_clean.mean().item():.3f} dB")
    print(f"Median SI-SDR: {all_si_sdr_clean.median().item():.3f} dB")
    print(f"Std SI-SDR: {all_si_sdr_clean.std().item():.3f} dB")

    # Compute and display improvement metrics
    all_si_sdr_mix = torch.cat(all_si_sdr_mix, dim=0)
    all_sdr_mix = torch.cat(all_sdr_mix, dim=0)
    all_sdr_clean = torch.cat(all_sdr_clean, dim=0)
    
    # SI-SDRi = SI-SDR(separated) - SI-SDR(mixture)
    si_sdri = all_si_sdr_clean - all_si_sdr_mix
    # SDRi = SDR(separated) - SDR(mixture)
    sdri = all_sdr_clean - all_sdr_mix
    
    print(f"\nImprovement Metrics:")
    print(f"Mean SI-SDRi: {si_sdri.mean().item():.3f} dB (improvement over mixture)")
    print(f"Median SI-SDRi: {si_sdri.median().item():.3f} dB")
    print(f"Mean SDRi: {sdri.mean().item():.3f} dB (improvement over mixture)")
    print(f"Median SDRi: {sdri.median().item():.3f} dB")
    
    print(f"\nBaseline (Mixture) Performance:")
    print(f"Mean SI-SDR (mixture): {all_si_sdr_mix.mean().item():.3f} dB")
    print(f"Mean SDR (mixture): {all_sdr_mix.mean().item():.3f} dB")
    
    if all_pesq:
        print(f"Mean PESQ: {np.mean(all_pesq):.3f}")
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
    ap = argparse.ArgumentParser(description="RWKV-v7 TF-Domain Inference")
    ap.add_argument("--csv", type=str, required=True, help="Path to test CSV")
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    ap.add_argument("--sample_rate", type=int, default=16000)
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
    n_fft = ckpt.get("n_fft")
    hop_length = ckpt.get("hop_length")
    
    if n_fft is None or hop_length is None:
        raise RuntimeError(
            "Checkpoint missing STFT parameters (n_fft, hop_length). "
            "Was this trained with train_rwkv_TFtest.py?"
        )
    
    print("\n[INFO] Checkpoint configuration:")
    print(f"  n_fft: {n_fft}")
    print(f"  hop_length: {hop_length}")
    for k, v in ckpt_config.items():
        if k not in ['train_csv', 'valid_csv', 'save_dir', 'log_dir']:
            print(f"  {k}: {v}")
    
    # Initialize STFT processor
    print(f"\n[INFO] Initializing STFT processor (n_fft={n_fft}, hop={hop_length})")
    stft_processor = STFTProcessor(n_fft=n_fft, hop_length=hop_length).to(device)
    
    # Get frequency dimension
    n_freq = n_fft // 2 + 1
    print(f"[INFO] Frequency bins: {n_freq}")
    
    # Build model
    n_layer = ckpt_config.get("n_layer", 8)
    n_embd = ckpt_config.get("n_embd")
    head_hidden = ckpt_config.get("head_hidden")
    head_mode = ckpt_config.get("head_mode", "residual")
    n_groups = ckpt_config.get("n_groups", 2)
    
    if n_embd is None or head_hidden is None:
        raise RuntimeError(
            "Checkpoint missing model parameters (n_embd, head_hidden). "
            "Check checkpoint config."
        )
    
    print(f"\n[INFO] Building model...")
    model = build_rwkv7_separator(
        n_embd=n_embd,
        codec_dim=n_freq,  # Frequency bins as feature dimension
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
        stft_processor=stft_processor,
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