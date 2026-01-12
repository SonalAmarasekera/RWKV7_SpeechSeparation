# -*- coding: utf-8 -*-
"""
FIXED Grouped Bi-RWKV Separator for DAC 16kHz (12 Codebooks).

Key fixes:
1. Changed default head_mode to "residual" (predicts offset from mixture, not mask)
2. Added softmax-based masking option for sum-to-one constraint
3. Reduced n_groups to 2 for better cross-dimension learning
4. Added latent-domain auxiliary loss support
"""
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Import Snake1d from your environment or define fallback ---
try:
    from dac.nn.layers import Snake1d
except ImportError:
    class Snake1d(nn.Module):
        def __init__(self, channels: int, alpha: float = 1.0):
            super().__init__()
            self.alpha = nn.Parameter(torch.ones(1, 1, channels) * alpha)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            a = self.alpha
            return x + (torch.sin(a * x) ** 2) / a.clamp_min(1e-4)

class SnakeChannelsLast(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.snake = Snake1d(channels)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.snake(x)
        x = x.transpose(1, 2)
        return x

# --- CUDA Kernel Import ---
RUN_CUDA = None
_run_import_err: Optional[str] = None
try:
    from RWKV.RWKV_v7.train_temp.src.model import RUN_CUDA_RWKV7g as RUN_CUDA
except Exception as e:
    pass

# --- Helpers ---
def time_shift_1(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x.new_zeros(x.size(0), 1, x.size(2)), x[:, :-1, :]], dim=1)

def pad_T_to_multiple(x: torch.Tensor, m: int) -> Tuple[torch.Tensor, int]:
    T = x.size(1)
    rem = T % m
    if rem == 0: return x, 0
    pad = m - rem
    return F.pad(x, (0, 0, 0, pad)), pad

# --- Config ---
@dataclass
class SeparatorV7Config:
    n_embd: int = 512
    codec_dim: int = 1024
    n_layer: int = 8
    head_size_a: int = 64
    enforce_bf16: bool = True
    num_sources: int = 2
    head_hidden: int = 256
    head_mode: str = "residual"  # CHANGED: default to residual instead of mask
    dropout: float = 0.1
    # CHANGED: Reduced from 4 to 2 groups for better cross-dimension learning
    n_groups: int = 2  

# --- Full RWKV Parameters ---
class Full_Tmix_Params(nn.Module):
    def __init__(self, n_embd: int, n_layer: int, layer_id: int, head_size: int):
        super().__init__()
        self.layer_id = layer_id
        self.n_embd = n_embd
        self.head_size = head_size
        
        C = n_embd
        H = n_embd // head_size
        ratio_0_to_1 = layer_id / max(1, (n_layer - 1))
        ratio_1_to_almost0 = 1.0 - (layer_id / max(1, n_layer))
        
        ddd = torch.ones(1, 1, C)
        for i in range(C): ddd[0, 0, i] = i / C
        self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
        self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
        self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
        self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
        
        def ortho_init(x, scale):
            with torch.no_grad():
                shape = x.shape
                if len(shape) == 2:
                    gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                    nn.init.orthogonal_(x, gain=gain * scale)
                elif len(shape) == 3:
                    gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                    for i in range(shape[0]):
                        nn.init.orthogonal_(x[i], gain=gain * scale)
                return x

        www = torch.zeros(C); zigzag = torch.zeros(C); linear = torch.zeros(C)
        for n in range(C):
            linear[n] = n / (C - 1) - 0.5
            zigzag[n] = ((n % head_size) - ((head_size - 1) / 2)) / ((head_size - 1) / 2)
            zigzag[n] = zigzag[n] * abs(zigzag[n])
            www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)

        D_DECAY_LORA = max(32, int(round((2.5 * (C ** 0.5)) / 32) * 32))
        self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
        self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
        self.w0 = nn.Parameter(www.reshape(1, 1, C) + 0.5 + zigzag * 2.5)

        D_AAA_LORA = max(32, int(round((2.5 * (C ** 0.5)) / 32) * 32))
        self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
        self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
        self.a0 = nn.Parameter(torch.zeros(1, 1, C) - 0.19 + zigzag * 0.3 + linear * 0.4)

        D_MV_LORA = max(32, int(round((1.7 * (C ** 0.5)) / 32) * 32))
        self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
        self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
        self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 0.73 - linear * 0.4)

        D_GATE_LORA = max(32, int(round((5 * (C ** 0.5)) / 32) * 32))
        self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
        self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

        self.k_k = nn.Parameter(torch.zeros(1, 1, C) + 0.71 - linear * 0.1)
        self.k_a = nn.Parameter(torch.zeros(1, 1, C) + 1.02)
        self.r_k = nn.Parameter(torch.zeros(n_embd // head_size, head_size) - 0.04)

        self.receptance = nn.Linear(C, C, bias=False)
        self.key        = nn.Linear(C, C, bias=False)
        self.value      = nn.Linear(C, C, bias=False)
        self.output     = nn.Linear(C, C, bias=False)
        self.ln_x       = nn.GroupNorm(n_embd // head_size, C, eps=64e-5)
        
        self.receptance.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
        self.key.weight.data.uniform_(-0.05 / (C ** 0.5), 0.05 / (C ** 0.5))
        self.value.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
        self.output.weight.data.zero_()

    def project(self, x, v_first):
        B, T, C = x.shape
        xx = time_shift_1(x) - x
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        
        if self.layer_id == 0:
            v_first = v
        else:
            if v_first is None: v_first = v
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
            
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        H = self.n_embd // self.head_size
        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), p=2.0, dim=-1).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)
        return r, w, k, v, a, g, kk, v_first

    def post_kernel(self, y, r, k, v, g):
        B, T, C = y.shape
        H = self.n_embd // self.head_size
        y = self.ln_x(y.view(B * T, C)).view(B, T, C)
        y = y + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k)
                 .sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        y = self.output(y * g)
        return y

# --- CGM (Fusion) ---
class CGM(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.conv = nn.Conv1d(2 * n_embd, 2 * n_embd, kernel_size=3, padding=1, groups=1)
    def forward(self, x_fwd, x_bwd):
        x_cat = torch.cat([x_fwd, x_bwd], dim=-1).transpose(1, 2)
        gate_out = F.glu(self.conv(x_cat), dim=1)
        return gate_out.transpose(1, 2)

# --- BiTimeMix (Single Group) ---
class BiTimeMixFull(nn.Module):
    def __init__(self, n_embd: int, n_layer: int, layer_id: int, head_size_a: int):
        super().__init__()
        self.fwd_params = Full_Tmix_Params(n_embd, n_layer, layer_id, head_size_a)
        self.bwd_params = Full_Tmix_Params(n_embd, n_layer, layer_id, head_size_a)
        self.cgm = CGM(n_embd)

    def _run_rwkv(self, params, x, v_in, bf16):
        r, w, k, v, a, g, kk, v_out = params.project(x.float(), v_in)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda and bf16):
            y = RUN_CUDA(r, w, k, v, -kk, kk*a)
        y = params.post_kernel(y.float(), r, k, v, g)
        return y, v_out

    def forward(self, x, v_state, bf16_enabled):
        if RUN_CUDA is None: raise RuntimeError("CUDA Missing")
        v_f_in, v_b_in = v_state
        
        y_fwd, v_f_out = self._run_rwkv(self.fwd_params, x, v_f_in, bf16_enabled)
        
        x_rev = torch.flip(x, dims=[1])
        y_bwd_rev, v_b_out = self._run_rwkv(self.bwd_params, x_rev, v_b_in, bf16_enabled)
        y_bwd = torch.flip(y_bwd_rev, dims=[1])
        
        y = self.cgm(y_fwd, y_bwd)
        return y, (v_f_out, v_b_out)

# --- Dual Context Aggregation (Cross-Group Mixing) ---
class DualContextAggregation(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.fc_global = nn.Linear(n_embd, n_embd)
        self.conv_local = nn.Conv1d(n_embd, n_embd, kernel_size=3, padding=1, groups=n_embd)
        self.lambda_param = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        u_avg = x.mean(dim=1)
        u_global = torch.sigmoid(self.fc_global(u_avg)).unsqueeze(1)
        u_local = self.conv_local(x.transpose(1, 2)).transpose(1, 2)
        u_local = torch.sigmoid(u_local)
        w = self.lambda_param * u_global + (1.0 - self.lambda_param) * u_local
        return x * w

# --- Grouped Bi-RWKV Block (FIXED: fewer groups) ---
class GroupedBiTimeMix(nn.Module):
    def __init__(self, n_embd: int, n_layer: int, layer_id: int, head_size_a: int, n_groups: int = 2):
        super().__init__()
        assert n_embd % n_groups == 0, "n_embd must be divisible by n_groups"
        self.n_groups = n_groups
        self.dim_per_group = n_embd // n_groups
        
        # Independent Bi-RWKV per group
        self.group_blocks = nn.ModuleList([
            BiTimeMixFull(self.dim_per_group, n_layer, layer_id, head_size_a)
            for _ in range(n_groups)
        ])
        # Mix them after processing
        self.dca = DualContextAggregation(n_embd)

    def forward(self, x, v_state_list, bf16_enabled):
        B, T, C = x.shape
        x_groups = x.view(B, T, self.n_groups, self.dim_per_group)
        
        y_outs = []
        v_states_out = []
        
        for i, blk in enumerate(self.group_blocks):
            x_slice = x_groups[:, :, i, :].contiguous()
            v_in = v_state_list[i] if v_state_list is not None else (None, None)
            y_slice, v_out = blk(x_slice, v_in, bf16_enabled)
            y_outs.append(y_slice)
            v_states_out.append(v_out)
            
        y = torch.cat(y_outs, dim=-1)
        y = self.dca(y)
        return y, v_states_out

# --- Full ChannelMix ---
class Full_CMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id):
        super().__init__()
        self.n_embd = n_embd
        C = n_embd
        ratio_1_to_almost0 = 1.0 - (layer_id / max(1, n_layer))
        ddd = torch.ones(1, 1, C)
        for i in range(C): ddd[0, 0, i] = i / C
        self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0 ** 4))
        self.key = nn.Linear(C, C * 4, bias=False)
        self.value = nn.Linear(C * 4, C, bias=False)
        self.key.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
        self.value.weight.data.zero_()
    def forward(self, x):
        xx = time_shift_1(x.float()) - x.float()
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)

# --- Block Wrapper ---
class Block(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, head_size_a, bf16, n_groups):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.n_groups = n_groups
        
        # USE GROUPED MIXER with configurable n_groups
        self.tmix = GroupedBiTimeMix(n_embd, n_layer, layer_id, head_size_a, n_groups=n_groups)
        
        self.cmix = Full_CMix(n_embd, n_layer, layer_id)
        self.bf16 = bf16
        self.dropout = nn.Dropout(SeparatorV7Config.dropout)
        if layer_id == 0: self.ln0 = nn.LayerNorm(n_embd)

    def forward(self, x, v_state_list):
        if self.layer_id == 0: x = self.ln0(x)
        
        x_attn, v_state_out = self.tmix(self.ln1(x), v_state_list, self.bf16)
        x = x + self.dropout(x_attn)
        x = x + self.dropout(self.cmix(self.ln2(x)))
        return x, v_state_out

# --- Core ---
class V7Core(nn.Module):
    def __init__(self, n_embd, n_layer, head_size_a, bf16, n_groups):
        super().__init__()
        self.n_groups = n_groups
        self.blocks = nn.ModuleList([
            Block(n_embd, n_layer, i, head_size_a, bf16, n_groups) for i in range(n_layer)
        ])
    def forward(self, x):
        v_state = [ (None, None) for _ in range(self.n_groups) ]
        for blk in self.blocks:
            x, v_state = blk(x, v_state)
        return x

# --- FIXED Separation Head with Multiple Modes ---
class SeparationHead(nn.Module):
    """
    Fixed separation head with multiple modes:
    - "residual": Predicts offset from mixture (recommended for latent domain)
    - "mask": Independent sigmoid masks (original, not recommended)
    - "softmax_mask": Softmax masks that sum to 1 across sources
    """
    def __init__(self, codec_dim: int, hidden: int, num_sources: int, 
                 mode: str = "residual"):
        super().__init__()
        self.mode = mode
        self.num_sources = num_sources
        self.codec_dim = codec_dim
        
#        self.pre = nn.Sequential(
#            nn.LayerNorm(codec_dim),
#            nn.Linear(codec_dim, hidden, bias=False),
#            nn.GELU(),
#        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden),
            SnakeChannelsLast(hidden),
            nn.Linear(hidden, num_sources * codec_dim, bias=False),
        )
        
        # Initialize output layer to small values for residual mode
        if mode == "residual":
            with torch.no_grad():
                self.output[-1].weight.data.mul_(0.1)
    
    def forward(self, x, x_ref):
        """
        x: [B, T, C] - processed features from RWKV core
        x_ref: [B, T, C] - original mixture latents
        Returns: [B, T, S, C] - separated latents for each source
        """
#        h = self.pre(x)
        y = self.output(x).view(x.size(0), x.size(1), self.num_sources, -1)
        
        if self.mode == "softmax_mask":
            # Improved mask mode: masks sum to 1 across sources
            # y: [B, T, S, C]
            m = F.softmax(y, dim=2)  # Softmax over source dimension
            return m * x_ref.unsqueeze(2)
        
        elif self.mode == "residual":
            # Recommended mode: predict offset from mixture
            # Each source = mixture + learned_offset
            return x_ref.unsqueeze(2) + y
        
        elif self.mode == "direct":
            # Direct prediction: just output y without reference
            return y
        
        else:
            raise ValueError(f"Unknown head mode: {self.mode}")

# --- Main Model Class ---
class RWKVv7Separator(nn.Module):
    def __init__(self, cfg: SeparatorV7Config):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Sequential(nn.LayerNorm(cfg.codec_dim), nn.Linear(cfg.codec_dim, cfg.n_embd, bias=False), nn.GELU())
        self.core = V7Core(cfg.n_embd, cfg.n_layer, cfg.head_size_a, cfg.enforce_bf16, cfg.n_groups)
        self.output_proj = nn.Sequential(nn.LayerNorm(cfg.n_embd), nn.Linear(cfg.n_embd, cfg.head_hidden, bias=False), nn.GELU())
        self.head = SeparationHead(cfg.codec_dim, cfg.head_hidden, cfg.num_sources, mode=cfg.head_mode)

    def forward(self, x):
        """
        x: [B, T, C] - Quantized mixture latents
        Returns: [B, T, S, C] - Separated latents for each source
        """
        x_compressed = self.input_proj(x)
        x_pad, pad = pad_T_to_multiple(x_compressed, 16)
        h = self.core(x_pad)
        if pad: h = h[:, :-pad, :]
        h_reconstructed = self.output_proj(h)
        y = self.head(h_reconstructed, x)
        return y

# --- Factory ---
def build_rwkv7_separator(
    n_embd: int,
    codec_dim: int,
    n_layer: int,
    num_sources: int = 2,
    *,
    head_size_a: int = 64,
    head_hidden: int,
    head_mode: str = "residual",  # CHANGED: default to residual
    enforce_bf16: bool = True,
    n_groups: int = 2,  # ADDED: configurable groups
) -> RWKVv7Separator:
    cfg = SeparatorV7Config(
        n_embd=n_embd,
        codec_dim=codec_dim,
        n_layer=n_layer,
        head_size_a=head_size_a,
        enforce_bf16=enforce_bf16,
        num_sources=num_sources,
        head_hidden=head_hidden,
        head_mode=head_mode,
        n_groups=n_groups,
    )
    return RWKVv7Separator(cfg)

if __name__ == "__main__":
    RWKVv7Separator(cfg)
