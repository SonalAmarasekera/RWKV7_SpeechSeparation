#!/usr/bin/env python3
"""
FIXED RWKV-v7 + DAC inference script with 5 evaluation metrics.

Metrics included:
1. SI-SDR (Signal-to-Interference Ratio) - vs clean and codec targets
2. PESQ (Perceptual Evaluation of Speech Quality)
3. STOI (Short-Time Objective Intelligibility)
4. NISQA (Non-Intrusive Speech Quality Assessment)
5. DNSMOS (Deep Noise Suppression Mean Opinion Score)

Key features:
1. Correct PIT permutation for all metrics
2. Consistent permutation across all metrics
3. Better error handling and failure tracking
4. Length alignment fixes

Usage:
    python infer_rwkv_cfstyle_all.py \
        --csv /path/to/test.csv \
        --checkpoint /path/to/best.pt \
        --sample_rate 16000 \
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
import pesq as pesq_fn
_HAS_PESQ = True

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
from codecformer3 import DACWrapper


# =========================
#   DATASET
# =========================

class Libri2MixEvalDataset(Dataset):
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
#   SI-SDR + PIT (FIXED)
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
    Classic SDR (non–scale-invariant) in dB.
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
    FIXED: Returns both SI-SDR and the best permutation indices.
    
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
    best_perm = (sum_perm2 > sum_perm1).long()  # [B]
    best_sum = torch.maximum(sum_perm1, sum_perm2)

    return best_sum / 2.0, best_perm


def reorder_by_perm(est_sources: torch.Tensor, best_perm: torch.Tensor) -> torch.Tensor:
    """
    Reorder estimates according to best permutation.
    
    Args:
        est_sources: [B, S, T]
        best_perm: [B] - 0 or 1 per batch element
    
    Returns:
        reordered: [B, S, T] - reordered to match reference order
    """
    B, S, T = est_sources.shape
    reordered = est_sources.clone()
    
    for i in range(B):
        if best_perm[i] == 1:
            # Swap speakers
            reordered[i] = est_sources[i, [1, 0], :]
    
    return reordered


# =========================
#   EVALUATION LOOP (FIXED)
# =========================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dac: DACWrapper,
    loader: DataLoader,
    device: torch.device,
    input_sample_rate: int,
    save_audio_dir: Optional[str] = None,
    enable_pesq: bool = False,
    enable_stoi: bool = False,
    enable_dnsmos: bool = False,
    enable_nisqa: bool = False,
) -> None:
    model.eval()
    dac.model.eval()

    all_si_sdr_clean = []
    all_si_sdr_codec = []

    # NEW: codec-domain mixture baseline + SDR tracking
    all_csi_sdr_mix = []      # cSI-SDR of mixture vs codec refs
    all_csd_r_est = []        # cSDR of estimates vs codec refs
    all_csd_r_mix = []        # cSDR of mixture vs codec refs

    all_pesq = []
    all_stoi = []
    all_dnsmos_sig = []   # Signal distortion (1-5, higher=better)
    all_dnsmos_bak = []   # Background noise (1-5, higher=cleaner)
    all_dnsmos_ovrl = []  # Overall quality (1-5, higher=better)
    all_nisqa = []
    
    # Initialize NISQA if enabled
    nisqa_model = None
    if enable_nisqa and _HAS_NISQA:
        try:
            nisqa_model = nisqa_fn(fs=input_sample_rate)
            nisqa_model.to(device)
        except Exception as e:
            print(f"[WARN] Failed to initialize NISQA: {e}")
            nisqa_model = None
            enable_nisqa = False
    
    # Track failures
    pesq_failures = 0
    stoi_failures = 0
    total_speaker_samples = 0

    if save_audio_dir:
        os.makedirs(save_audio_dir, exist_ok=True)

    # Validate metric availability
    if enable_pesq and not _HAS_PESQ:
        print("[WARN] pesq package not found. Disabling PESQ.")
        enable_pesq = False
    if enable_stoi and not _HAS_STOI:
        print("[WARN] pystoi package not found. Disabling STOI.")
        enable_stoi = False
    if enable_dnsmos and not _HAS_DNSMOS:
        print("[WARN] torchmetrics DNSMOS not found. Disabling DNSMOS.")
        enable_dnsmos = False
    if enable_nisqa and not _HAS_NISQA:
        print("[WARN] torchmetrics NISQA not found. Disabling NISQA.")
        enable_nisqa = False

    pesq_mode = "nb" if input_sample_rate <= 8000 else "wb"

    for batch_idx, (mix, sources, rows) in enumerate(loader):
        mix = mix.to(device)
        sources = sources.to(device)
        B, S, _, T_orig = sources.shape
        assert S == 2

        # ---------- DAC encode mixture ----------
        mix_enc, orig_len = dac.get_encoded_features(mix)
        mix_q, _, _, _, _ = dac.get_quantized_features(mix_enc)
        z_mix = mix_q.permute(0, 2, 1).contiguous()

        # NEW: decode mixture through DAC to get codec-domain mixture baseline
        mix_codec = dac.get_decoded_signal(mix_q, orig_len)  # [B, 1, T_mix_codec]

        # ---------- RWKV separation ----------
        sep_lat = model(z_mix)
        sep_lat = sep_lat.permute(0, 2, 3, 1).contiguous()

        # ---------- DAC decode estimates ----------
        est_wavs = []
        for s_idx in range(S):
            z_s = sep_lat[:, s_idx, :, :]
            wav_hat = dac.get_decoded_signal(z_s, orig_len)
            est_wavs.append(wav_hat)

        est_sources = torch.stack(est_wavs, dim=1).squeeze(2)  # [B, S, T_est]
        T_est = est_sources.shape[-1]

        # ---------- Length alignment (clean domain) ----------
        min_T = min(T_est, T_orig)
        est_sources = est_sources[..., :min_T]
        true_sources_clean = sources.squeeze(2)[..., :min_T]

        # ---------- Codec references for clean sources ----------
        sources_flat = true_sources_clean.reshape(B * S, 1, min_T)
        src_enc, src_orig_len = dac.get_encoded_features(sources_flat)
        src_q, _, _, _, _ = dac.get_quantized_features(src_enc)
        codec_src_flat = dac.get_decoded_signal(src_q, src_orig_len)

        T_codec = codec_src_flat.shape[-1]
        T_mix_codec = mix_codec.shape[-1]

        # Final alignment across est, clean, codec refs, and codec mixture
        final_T = min(min_T, T_codec, T_mix_codec)

        est_sources = est_sources[..., :final_T]
        true_sources_clean = true_sources_clean[..., :final_T]
        codec_sources = codec_src_flat.view(B, S, T_codec)[..., :final_T]
        mix_codec = mix_codec[..., :final_T].squeeze(1)  # [B, T]

        # ---------- SI-SDR with permutation (codec targets) ----------
        # Use codec targets to determine permutation (matches training objective)
        si_sdr_codec, best_perm = pit_si_sdr_with_perm(est_sources, codec_sources)

        # Compute clean SI-SDR with SAME permutation for consistency
        est_reordered = reorder_by_perm(est_sources, best_perm)

        # Now compute clean SI-SDR on reordered estimates
        sdr_clean_s0 = si_sdr(est_reordered[:, 0], true_sources_clean[:, 0])
        sdr_clean_s1 = si_sdr(est_reordered[:, 1], true_sources_clean[:, 1])
        si_sdr_clean = (sdr_clean_s0 + sdr_clean_s1) / 2.0

        all_si_sdr_clean.append(si_sdr_clean.cpu())
        all_si_sdr_codec.append(si_sdr_codec.cpu())

        # ---------- NEW: codec-domain mixture baseline & SDR ----------

        # Broadcast codec mixture to both speaker slots: [B, T] -> [B, S, T]
        mix_codec_rep = mix_codec.unsqueeze(1).expand(-1, S, -1)

        # cSI-SDR of mixture vs codec references
        mix_csi_s0 = si_sdr(mix_codec_rep[:, 0], codec_sources[:, 0])
        mix_csi_s1 = si_sdr(mix_codec_rep[:, 1], codec_sources[:, 1])
        csi_sdr_mix = (mix_csi_s0 + mix_csi_s1) / 2.0
        all_csi_sdr_mix.append(csi_sdr_mix.cpu())

        # cSDR of estimates vs codec references
        est_csd_s0 = sdr(est_reordered[:, 0], codec_sources[:, 0])
        est_csd_s1 = sdr(est_reordered[:, 1], codec_sources[:, 1])
        csdr_est = (est_csd_s0 + est_csd_s1) / 2.0
        all_csd_r_est.append(csdr_est.cpu())

        # cSDR of mixture vs codec references
        mix_csd_s0 = sdr(mix_codec_rep[:, 0], codec_sources[:, 0])
        mix_csd_s1 = sdr(mix_codec_rep[:, 1], codec_sources[:, 1])
        csdr_mix = (mix_csd_s0 + mix_csd_s1) / 2.0
        all_csd_r_mix.append(csdr_mix.cpu())

        # ---------- Perceptual metrics on REORDERED estimates ----------
        if enable_pesq or enable_stoi or enable_dnsmos or enable_nisqa:
            est_np = est_reordered.cpu().numpy()  # Use reordered!
            clean_np = true_sources_clean.cpu().numpy()

        batch_pesq = []
        batch_stoi = []
        batch_dnsmos_sig = []
        batch_dnsmos_bak = []
        batch_dnsmos_ovrl = []
        batch_nisqa = []

        for i in range(B):
            for s_idx in range(S):
                total_speaker_samples += 1
                ref = clean_np[i, s_idx, :]
                deg = est_np[i, s_idx, :]

                # PESQ
                if enable_pesq:
                    try:
                        score = float(pesq_fn(input_sample_rate, ref, deg, pesq_mode))
                        batch_pesq.append(score)
                    except Exception:
                        pesq_failures += 1

                # STOI
                if enable_stoi:
                    try:
                        score = float(stoi_fn(ref, deg, input_sample_rate, extended=False))
                        batch_stoi.append(score)
                    except Exception:
                        stoi_failures += 1

                # DNSMOS
                if enable_dnsmos:
                    try:
                        wav_t = torch.from_numpy(deg).to(device)
                        scores = dnsmos_fn(wav_t, fs=input_sample_rate, personalized=False)
                        # DNSMOS returns 3 scores:
                        # scores[0] = SIG  (Signal distortion)
                        # scores[1] = BAK  (Background noise)
                        # scores[2] = OVRL (Overall quality)
                        batch_dnsmos_sig.append(float(scores[0].item()))
                        batch_dnsmos_bak.append(float(scores[1].item()))
                        batch_dnsmos_ovrl.append(float(scores[2].item()))
                    except Exception:
                        pass

                # NISQA (no-reference, only needs estimated signal)
                if enable_nisqa and nisqa_model is not None:
                    try:
                        wav_t = torch.from_numpy(deg).unsqueeze(0).to(device)
                        nisqa_score = nisqa_model(wav_t, input_sample_rate)
                        batch_nisqa.append(float(nisqa_score.item()))
                    except Exception:
                        pass

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
            est_save = est_reordered.cpu().numpy()  # Save reordered
            for i in range(B):
                base = os.path.splitext(os.path.basename(rows[i]["mix_path"]))[0]
                for s_idx in range(S):
                    out_path = os.path.join(save_audio_dir, f"{base}_spk{s_idx+1}.wav")
                    sf.write(out_path, est_save[i, s_idx], samplerate=input_sample_rate)

        # ---------- Batch summary ----------
        msg = (f"[BATCH {batch_idx:04d}] "
               f"SI-SDR clean={si_sdr_clean.mean().item():.2f} dB, "
               f"Codec SI-SDR={si_sdr_codec.mean().item():.2f} dB")
        if batch_pesq:
            msg += f", PESQ={np.mean(batch_pesq):.3f}"
        if batch_stoi:
            msg += f", STOI={np.mean(batch_stoi):.3f}"
        if batch_nisqa:
            msg += f", NISQA={np.mean(batch_nisqa):.3f}"
        print(msg)

    # ---------- Final summary ----------
    all_si_sdr_clean = torch.cat(all_si_sdr_clean, dim=0)
    all_si_sdr_codec = torch.cat(all_si_sdr_codec, dim=0)
    all_csi_sdr_mix = torch.cat(all_csi_sdr_mix, dim=0)
    all_csd_r_est = torch.cat(all_csd_r_est, dim=0)
    all_csd_r_mix = torch.cat(all_csd_r_mix, dim=0)

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total utterances: {all_si_sdr_clean.numel()}")
    print(f"Mean SI-SDR (vs clean):  {all_si_sdr_clean.mean().item():.3f} dB")
    print(f"Mean SI-SDR (vs codec):  {all_si_sdr_codec.mean().item():.3f} dB")
    # NOTE: this is *not* a true SI-SDRi; kept for backward compatibility
    print(f"SI-SDR 'gap' (codec - clean): {(all_si_sdr_codec.mean() - all_si_sdr_clean.mean()).item():.3f} dB")

    # --- NEW: codec-domain improvements (cSI-SDRi & cSDRi) ---
    mean_csi_sdr_est = all_si_sdr_codec.mean()
    mean_csi_sdr_mix = all_csi_sdr_mix.mean()
    csi_sdri = (mean_csi_sdr_est - mean_csi_sdr_mix).item()

    mean_csdr_est = all_csd_r_est.mean()
    mean_csdr_mix = all_csd_r_mix.mean()
    csdri = (mean_csdr_est - mean_csdr_mix).item()

    print(f"Mean cSI-SDR (est vs codec refs):   {mean_csi_sdr_est.item():.3f} dB")
    print(f"Mean cSI-SDR (mix vs codec refs):   {mean_csi_sdr_mix.item():.3f} dB")
    print(f"cSI-SDRi (improvement over mix):    {csi_sdri:.3f} dB")

    print(f"Mean cSDR (est vs codec refs):      {mean_csdr_est.item():.3f} dB")
    print(f"Mean cSDR (mix vs codec refs):      {mean_csdr_mix.item():.3f} dB")
    print(f"cSDRi (improvement over mix):       {csdri:.3f} dB")

    if all_pesq:
        print(f"Mean PESQ:               {np.mean(all_pesq):.3f}")
        if pesq_failures > 0:
            print(f"  (PESQ failures: {pesq_failures}/{total_speaker_samples})")
    if all_stoi:
        print(f"Mean STOI:               {np.mean(all_stoi):.3f}")
        if stoi_failures > 0:
            print(f"  (STOI failures: {stoi_failures}/{total_speaker_samples})")
    if all_dnsmos_ovrl:
        print(f"Mean DNSMOS (SIG):       {np.mean(all_dnsmos_sig):.3f}")
        print(f"Mean DNSMOS (BAK):       {np.mean(all_dnsmos_bak):.3f}")
        print(f"Mean DNSMOS (OVRL):      {np.mean(all_dnsmos_ovrl):.3f}")
    if all_nisqa:
        print(f"Mean NISQA (overall):    {np.mean(all_nisqa):.3f}")
    print("=" * 50 + "\n")


# =========================
#   MAIN
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_audio_dir", type=str, default=None)
    ap.add_argument("--subset_frac", type=float, default=None)
    ap.add_argument("--enable_pesq", action="store_true")
    ap.add_argument("--enable_stoi", action="store_true")
    ap.add_argument("--enable_dnsmos", action="store_true")
    ap.add_argument("--enable_nisqa", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_config = ckpt.get("config", {})
    latent_channels = ckpt.get("latent_channels")

    if latent_channels is None:
        raise RuntimeError("Checkpoint missing 'latent_channels'")

    print("[INFO] Checkpoint config:")
    for k, v in ckpt_config.items():
        print(f"  {k}: {v}")

    # DAC
    dac = DACWrapper(
        input_sample_rate=args.sample_rate,
        DAC_model_path=None,
        DAC_sample_rate=16000,
        Freeze=True,
    )
    dac.model.to(device)
    dac.dac_sampler.to(device)
    dac.org_sampler.to(device)

    # Model
    n_layer = ckpt_config.get("n_layer", 8)
    head_mode = ckpt_config.get("head_mode", "residual")
    n_groups = ckpt_config.get("n_groups", 2)
    n_embd = ckpt_config.get("n_embd", latent_channels)
    head_hidden = ckpt_config.get("head_hidden", None)

    if head_hidden is None:
        raise RuntimeError(
            "Checkpoint config missing 'head_hidden'; "
            "did you train with train_rwkv_Final.py?"
        )

    model = build_rwkv7_separator(
        n_embd=n_embd,
        codec_dim=latent_channels,
        n_layer=n_layer,
        num_sources=2,
        head_mode=head_mode,
        enforce_bf16=False,
        head_hidden=head_hidden,
        n_groups=n_groups,
    ).to(device)

    # -------------------------------
    # Clean profiler keys from state_dict
    # -------------------------------
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
        print(f"[INFO] Stripped {len(skipped)} profiler keys from checkpoint:")

    model.load_state_dict(clean_state, strict=True)
    model.eval()


    print(f"[INFO] Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"[INFO] Head mode: {head_mode}, n_groups: {n_groups}")

    # Data
    eval_ds = Libri2MixEvalDataset(args.csv, sample_rate=args.sample_rate, subset_frac=args.subset_frac)
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
        dac=dac,
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
