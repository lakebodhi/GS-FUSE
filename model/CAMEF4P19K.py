# -*- coding: utf-8 -*-
"""
CAMEF Enhanced Training Script v3 (Last-Layer Fine-tuning)
===========================================================
Enhanced to achieve MSE ~0.0005

Key Enhancements:
1. Raw TS residual added to TS instance embedding (from v2)
2. Learnable last-value residual - learns to predict CHANGES from last value
3. Multi-scale temporal features from raw TS
4. Direct temporal shortcut in decoder
5. LLM last layer fine-tuning with very low learning rate

The key insight: Financial time series are often near-random-walk. The best prediction
is often "last value + small learned adjustment". This version learns that adjustment
while still leveraging the full multimodal fusion.

LLM Fine-tuning: Only the last transformer block of the LLM is fine-tuned with a very
low learning rate (1e-6), enabling better text-TS alignment while preserving pretrained
knowledge.
"""

import os
import gc
import sys
import warnings
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.optimization import get_cosine_schedule_with_warmup

# ----- Kronos optional import -----
try:
    # adjust path if your Kronos package lives elsewhere
    from model.kronos import Kronos, KronosTokenizer, KronosAdapter
except Exception:
    Kronos = None
    KronosTokenizer = None
    KronosAdapter = None



# Use float32 globally
torch.set_default_dtype(torch.float32)

# ---------------- Env / Warnings ----------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning,
                        message="torch.utils._pytree._register_pytree_node is deprecated")


# ---------------- Logger Helper ----------------
class DualLogger(object):
    """Redirects sys.stdout to both the terminal and a log file."""

    def __init__(self, filename: str):
        self.terminal = sys.stdout
        # Line-buffered text file: writes appear promptly in the log file.
        self.log = open(filename, "w", encoding="utf-8", buffering=1)

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # Needed for python 3 compatibility
        self.terminal.flush()
        self.log.flush()

    def close(self):
        try:
            self.log.close()
        except Exception:
            pass

# ---------------- Config ----------------
LLAMA_HIDDEN = 3072  # hidden size
USE_4BIT = False

FUSE_DIM = 1024
NUM_HEADS = 8
PRED_DROPOUT = 0.1

# ============== KEY ENHANCEMENT FLAGS ==============
# Raw TS residual - adds projected raw TS to TS instance embedding
USE_RAW_TS_RESIDUAL = True
RAW_TS_RESIDUAL_WEIGHT = 1.0

# Learnable last-value residual - model predicts DELTA from last value
# This is the KEY to achieving low MSE on near-random-walk time series
USE_LEARNABLE_LAST_VALUE = False  # NEW: Enable learnable last-value prediction
LAST_VALUE_INIT_WEIGHT = 0.9  # NEW: Initial weight for last value (high = more conservative)

# Multi-scale temporal features
USE_MULTI_SCALE_TS = True  # NEW: Extract features at multiple temporal scales

# EMA for training stability
USE_EMA = False
EMA_DECAY = 0.99

# ============== LLM Fine-tuning Configuration ==============
# Unfreeze the last transformer block of LLaMA for fine-tuning
FINETUNE_LLM_LAST_LAYER = True  # Enable fine-tuning of LLaMA's last layer
LLM_FINETUNE_LR = 1e-6  # Very low LR for LLM (10x lower than base)


# ============== Training Configuration ==============
class TrainingConfig:
    """Centralized training configuration for easy tuning."""
    # Learning rates
    BASE_LR = 1e-5
    LLM_PROJ_LR = 1e-5
    DECODER_LR = 1e-5
    LLM_LAST_LAYER_LR = 1e-6  # Very low LR for LLM fine-tuning

    # Gradient clipping
    GRAD_CLIP_NORM = 0.5

    # Warmup
    WARMUP_RATIO = 0.10

    # Loss weights
    CONTRASTIVE_LAMBDA = 0.2
    GATE_CAUSAL_LAMBDA = 0.1
    INV_LAMBDA = 0.1
    TOKEN_ALIGN_LAMBDA = 0.01
    TOKEN_ALIGN_USE_TOPK = True
    TOKEN_ALIGN_TOPK_TEXT = 256
    TOKEN_ALIGN_TAU_ALIGN = 0.2
    TOKEN_ALIGN_TAU_NCE = 0.07
    TS_ONLY_LAMBDA = 0.1
    GATE_ENTROPY_LAMBDA = 0.01

    # ---- Granger sigmoid scaling (prevents gc_score ~ 0.5000 when diff is tiny) ----
    GC_USE_ADAPTIVE_SCALE = True   # robust default; auto-scales to current diff magnitude
    GC_SIGMOID_SCALE = 4e-5        # used only when GC_USE_ADAPTIVE_SCALE = False
    GC_MIN_SCALE = 1e-6            # avoid division by zero / huge logits
    GC_GAIN = 1.0                  # increase (e.g., 2.0~5.0) for stronger push
    GC_LOGIT_CLAMP = 6.0           # avoid sigmoid saturation


    # Simplified mode
    SIMPLIFIED_MODE = False

    # ============== Three-Stage Training Configuration ==============
    # Stage 1: Time Series Pre-training (sliding window on TS modality only)
    STAGE1_EPOCHS = 2
    STAGE1_LR = 1e-4  # Higher LR for faster pre-training

    # Stage 2: Text Pre-training (text modality only, omit TS)
    STAGE2_EPOCHS = 2
    STAGE2_LR = 1e-4

    # Stage 3: Full Multimodal Training (current design)
    # Uses existing epoch count and learning rates


def _load_llama_for_feature_extraction(name: str, use_4bit: bool = True):
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float32
        )
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name,
        device_map="auto",
        torch_dtype=torch.float32,
        quantization_config=bnb_config if use_4bit else None
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.eval()
    return tok, model


# ---------------- Multi-Scale Temporal Feature Extractor ----------------
class MultiScaleTemporalExtractor(nn.Module):
    """
    Extract features at multiple temporal scales from raw time series.
    This helps capture both short-term patterns and longer-term trends.
    """

    def __init__(self, d_input, d_output, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.d_input = d_input

        # Scale 1: Point-wise (captures instantaneous values)
        self.scale1 = nn.Sequential(
            nn.Linear(d_input * seq_len, d_output),
            nn.GELU(),
        )

        # Scale 2: Short-term (last 5 steps)
        short_len = min(5, seq_len)
        self.scale2 = nn.Sequential(
            nn.Linear(d_input * short_len, d_output),
            nn.GELU(),
        )
        self.short_len = short_len

        # Scale 3: Medium-term (last 10 steps)
        med_len = min(10, seq_len)
        self.scale3 = nn.Sequential(
            nn.Linear(d_input * med_len, d_output),
            nn.GELU(),
        )
        self.med_len = med_len

        # Scale 4: Trend (difference between first and last halves)
        half_len = seq_len // 2
        self.scale4 = nn.Sequential(
            nn.Linear(d_input * 2, d_output),  # First half mean, last half mean
            nn.GELU(),
        )
        self.half_len = half_len

        # Combine all scales
        self.combine = nn.Sequential(
            nn.Linear(d_output * 4, d_output * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_output * 2, d_output),
        )

    def forward(self, x):
        """
        x: [B, d, L] channels-first time series
        Returns: [B, d_output]
        """
        B, d, L = x.shape

        # Scale 1: Full sequence
        x_flat = x.reshape(B, -1)
        s1 = self.scale1(x_flat)

        # Scale 2: Last short_len steps
        x_short = x[:, :, -self.short_len:].reshape(B, -1)
        s2 = self.scale2(x_short)

        # Scale 3: Last med_len steps
        x_med = x[:, :, -self.med_len:].reshape(B, -1)
        s3 = self.scale3(x_med)

        # Scale 4: Trend (first half vs last half)
        first_half = x[:, :, :self.half_len].mean(dim=2)  # [B, d]
        last_half = x[:, :, -self.half_len:].mean(dim=2)  # [B, d]
        trend = torch.cat([first_half, last_half], dim=1)  # [B, 2*d]
        s4 = self.scale4(trend)

        # Combine
        combined = torch.cat([s1, s2, s3, s4], dim=1)
        return self.combine(combined)


# ---------------- Learnable Last-Value Residual Module ----------------
class LearnableLastValueResidual(nn.Module):
    """
    Instead of naively adding last value, this module learns:
    1. How much to weight the last value vs the model prediction
    2. A learned adjustment/delta from the last value

    output = alpha * last_value_expanded + (1-alpha) * model_prediction + delta

    where alpha and delta are learned based on the context.
    """

    def __init__(self, context_dim, d_channels, pred_len, init_alpha=0.9):
        super().__init__()
        self.d = d_channels
        self.pred_len = pred_len

        # Learn alpha (mixing weight) from context
        self.alpha_net = nn.Sequential(
            nn.Linear(context_dim, context_dim // 2),
            nn.GELU(),
            nn.Linear(context_dim // 2, 1),
            nn.Sigmoid()
        )

        # Initialize to favor last value initially
        with torch.no_grad():
            # Bias the output to be close to init_alpha
            self.alpha_net[-2].bias.fill_(math.log(init_alpha / (1 - init_alpha + 1e-8)))

        # Learn delta (adjustment) from context + last values
        self.delta_net = nn.Sequential(
            nn.Linear(context_dim + d_channels, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, d_channels * pred_len),
        )

        # Initialize delta to be small
        with torch.no_grad():
            self.delta_net[-1].weight.mul_(0.01)
            self.delta_net[-1].bias.zero_()

    def forward(self, model_pred, last_value, context):
        """
        model_pred: [B, d, pred_len] - raw model prediction
        last_value: [B, d, 1] - last observed value
        context: [B, context_dim] - fused embedding for conditioning

        Returns: [B, d, pred_len] - refined prediction
        """
        B = model_pred.size(0)

        # Expand last value to prediction length
        last_expanded = last_value.expand(-1, -1, self.pred_len)  # [B, d, pred_len]

        # Compute mixing weight alpha
        alpha = self.alpha_net(context)  # [B, 1]
        alpha = alpha.unsqueeze(-1)  # [B, 1, 1]

        # Compute delta adjustment
        last_flat = last_value.squeeze(-1)  # [B, d]
        delta_input = torch.cat([context, last_flat], dim=-1)  # [B, context_dim + d]
        delta = self.delta_net(delta_input).view(B, self.d, self.pred_len)  # [B, d, pred_len]

        # Combine: weighted average + learned delta
        output = alpha * last_expanded + (1 - alpha) * model_pred + delta

        return output, alpha.squeeze()


# ---------------- TinyDecoder with Temporal Shortcut ----------------
class TinyDecoderV3(nn.Module):
    """
    Enhanced decoder with:
    1. Standard transformer decoder
    2. Direct temporal shortcut from raw TS features
    """

    def __init__(self, context_dim=FUSE_DIM, d=4, pred_len=35, width=FUSE_DIM,
                 n_heads=8, n_layers=4, dropout=PRED_DROPOUT, seq_len=35):
        super().__init__()
        self.pred_len = pred_len
        self.d = d
        self.seq_len = seq_len

        self.ctx_proj = nn.Linear(context_dim, width)
        self.query_tokens = nn.Parameter(torch.randn(pred_len, width) * 0.02)
        self.horizon_pos = nn.Parameter(torch.zeros(pred_len, width))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=width, nhead=n_heads, dim_feedforward=4 * width,
            batch_first=True, dropout=dropout, activation='gelu',
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, d)
        )

        self.ctx_to_channel = nn.Linear(context_dim, d)
        nn.init.zeros_(self.ctx_to_channel.weight)
        nn.init.zeros_(self.ctx_to_channel.bias)

        # NEW: Direct temporal shortcut
        # This allows the decoder to directly use patterns from the input sequence
        self.temporal_shortcut = nn.Sequential(
            nn.Linear(d * seq_len, width),
            nn.GELU(),
            nn.Linear(width, d * pred_len),
        )
        # Initialize to small values
        with torch.no_grad():
            self.temporal_shortcut[-1].weight.mul_(0.1)
            self.temporal_shortcut[-1].bias.zero_()

    def forward(self, fused_emb, x_cf=None):
        """
        fused_emb: [B, FUSE_DIM]
        x_cf: [B, d, L] raw time series (optional, for temporal shortcut)
        """
        B = fused_emb.size(0)
        ctx = self.ctx_proj(fused_emb).unsqueeze(1)  # [B,1,W]
        q = (self.query_tokens + self.horizon_pos).unsqueeze(0).expand(B, -1, -1)  # [B,L,W]
        out = self.decoder(tgt=q, memory=ctx)  # [B,L,W]
        out = self.norm(out)
        dec = self.head(out).transpose(1, 2)  # [B,d,L]
        bias = self.ctx_to_channel(fused_emb).unsqueeze(-1)
        main_out = dec + bias.expand(B, self.d, self.pred_len)

        # Add temporal shortcut if available
        if x_cf is not None:
            x_flat = x_cf.reshape(B, -1)
            shortcut = self.temporal_shortcut(x_flat).view(B, self.d, self.pred_len)
            main_out = main_out + shortcut

        return main_out


# ---------------- Token-Wise Alignment Module ----------------
class TokenWiseAlignmentModule(nn.Module):
    """
    Token-to-Token Alignment with Top-K text anchors (forecast-safer).

    Core:
      - similarity S[b,i,j] between text token i and TS token j
      - alignment map P_t2s = softmax_j(S)
      - soft-positive TS vector for each text token: z_ts_pos[b,i] = sum_j P_t2s[b,i,j] * z_ts[b,j]

    Loss (token-level InfoNCE):
      - Anchors: only Top-K *salient* text tokens per sample (reduces noise vs dense-all-tokens)
      - Negatives: TS tokens from other samples only (same-sample masked out)
      - Optional: entropy + cycle regularizers (masked, salience-weighted)

    Mask convention (consistent with torch MultiheadAttention):
      text_mask / ts_mask: [B, L] with True = PAD, False = valid
    """

    def __init__(
        self,
        dim: int = FUSE_DIM,
        proj_dim: int = 256,
        tau_align: float = 0.2,
        tau_nce: float = 0.07,
        use_topk: bool = True,
        topk_text: int = 8,
        lambda_entropy: float = 0.0,
        lambda_cycle: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.tau_align = float(tau_align)
        self.tau_nce = float(tau_nce)
        self.use_topk = bool(use_topk)
        self.topk_text = int(topk_text)
        self.lambda_entropy = float(lambda_entropy)
        self.lambda_cycle = float(lambda_cycle)
        self.eps = float(eps)

        self.text_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.ts_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # learned salience over text tokens (helps pick forecast-relevant anchors)
        self.text_salience = nn.Linear(dim, 1)

    @staticmethod
    def _select_topk(score: torch.Tensor, valid: torch.Tensor, k: int):
        """
        score: [B, L] (larger = more important)
        valid: [B, L] bool (True = valid token)
        Return:
          idx: [B, k]
          ok : [B, k] bool
        """
        score2 = score.masked_fill(~valid, -1e9)
        k = min(k, score2.size(1))
        topv, idx = torch.topk(score2, k=k, dim=1)
        ok = topv > -1e8
        return idx, ok

    @staticmethod
    def _gather_tokens(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        x:  [B, L, D]
        idx:[B, K]
        out:[B, K, D]
        """
        B, L, D = x.shape
        K = idx.shape[1]
        return x.gather(dim=1, index=idx.unsqueeze(-1).expand(B, K, D))

    def forward(
        self,
        text_tokens: torch.Tensor,  # [B, Lt, dim]
        ts_tokens: torch.Tensor,    # [B, Ls, dim]
        text_mask: torch.Tensor = None,  # [B, Lt] True=PAD
        ts_mask: torch.Tensor = None,    # [B, Ls] True=PAD
        return_alignment: bool = False,
    ):
        B, Lt, _ = text_tokens.shape
        Ls = ts_tokens.shape[1]
        device = text_tokens.device

        tau_a = max(self.tau_align, 1e-6)
        tau_c = max(self.tau_nce, 1e-6)

        # valid masks (True = valid token)
        text_valid = (~text_mask) if text_mask is not None else torch.ones(B, Lt, dtype=torch.bool, device=device)
        ts_valid   = (~ts_mask)   if ts_mask is not None else torch.ones(B, Ls, dtype=torch.bool, device=device)

        # 1) Project into shared alignment space and normalize
        z_text = F.normalize(self.text_proj(text_tokens), dim=-1)  # [B, Lt, D']
        z_ts   = F.normalize(self.ts_proj(ts_tokens),   dim=-1)    # [B, Ls, D']

        # 2) Dense similarity
        S = torch.bmm(z_text, z_ts.transpose(1, 2)) / tau_a  # [B, Lt, Ls]

        # mask pads in similarity
        if ts_mask is not None:
            S = S.masked_fill(ts_mask.unsqueeze(1), -1e9)      # mask columns
        if text_mask is not None:
            S = S.masked_fill(text_mask.unsqueeze(2), -1e9)    # mask rows

        # 3) Soft alignments with safe renorm (avoid PAD rows/cols issues)
        P_t2s = F.softmax(S, dim=-1)  # [B, Lt, Ls]
        if ts_mask is not None:
            P_t2s = P_t2s.masked_fill(ts_mask.unsqueeze(1), 0.0)
        if text_mask is not None:
            P_t2s = P_t2s.masked_fill(text_mask.unsqueeze(2), 0.0)
        P_t2s = P_t2s / (P_t2s.sum(dim=-1, keepdim=True) + self.eps)

        P_s2t = F.softmax(S.transpose(1, 2), dim=-1)  # [B, Ls, Lt]
        if text_mask is not None:
            P_s2t = P_s2t.masked_fill(text_mask.unsqueeze(1), 0.0)
        if ts_mask is not None:
            P_s2t = P_s2t.masked_fill(ts_mask.unsqueeze(2), 0.0)
        P_s2t = P_s2t / (P_s2t.sum(dim=-1, keepdim=True) + self.eps)

        # 4) Soft positives (per text token)
        z_ts_pos = torch.bmm(P_t2s, z_ts)  # [B, Lt, D']

        # 5) Text token salience a[b,i] (distribution over valid tokens)
        sal = self.text_salience(text_tokens).squeeze(-1)  # [B, Lt]
        sal = sal.masked_fill(~text_valid, -1e9)
        a = F.softmax(sal, dim=1)  # [B, Lt]
        a = a * text_valid.to(a.dtype)
        a = a / (a.sum(dim=1, keepdim=True) + self.eps)

        # 6) Select anchors (Top-K) or use all valid tokens (dense)
        if self.use_topk:
            idx_t, ok_t = self._select_topk(a, text_valid, self.topk_text)  # [B, K]
            zt_anchor = self._gather_tokens(z_text, idx_t)       # [B, K, D']
            zs_pos    = self._gather_tokens(z_ts_pos, idx_t)     # [B, K, D']
            w_anchor  = self._gather_tokens(a.unsqueeze(-1), idx_t).squeeze(-1)  # [B, K]

            # flatten valid anchors
            flat_mask = ok_t
            zt_flat = zt_anchor[flat_mask]           # [N, D']
            zs_pos_flat = zs_pos[flat_mask]          # [N, D']
            w_flat = w_anchor.masked_fill(~ok_t, 0.0)[flat_mask]  # [N]
            anchor_sid = torch.arange(B, device=device).unsqueeze(1).expand_as(ok_t)[flat_mask]  # [N]
        else:
            zt_flat = z_text[text_valid]             # [N, D']
            zs_pos_flat = z_ts_pos[text_valid]       # [N, D']
            w_flat = a[text_valid]                   # [N]
            anchor_sid = torch.arange(B, device=device).unsqueeze(1).expand(B, Lt)[text_valid]  # [N]

        # Degenerate
        if zt_flat.numel() == 0:
            loss = z_text.sum() * 0.0
            if return_alignment:
                out = {"P_t2s": P_t2s, "P_s2t": P_s2t, "S": S, "a_text": a}
                if self.use_topk:
                    out.update({"idx_t": idx_t, "ok_t": ok_t})
                return loss, out
            return loss

        # 7) Candidate TS bank (valid TS tokens only)
        z_ts_flat = z_ts[ts_valid]  # [N_ts, D']
        if z_ts_flat.numel() == 0:
            loss = z_text.sum() * 0.0
            if return_alignment:
                out = {"P_t2s": P_t2s, "P_s2t": P_s2t, "S": S, "a_text": a}
                if self.use_topk:
                    out.update({"idx_t": idx_t, "ok_t": ok_t})
                return loss, out
            return loss

        ts_sid = torch.arange(B, device=device).unsqueeze(1).expand(B, Ls)[ts_valid]  # [N_ts]

        # 8) InfoNCE (cross-sample negatives only)
        pos_scores = (zt_flat * zs_pos_flat).sum(dim=-1) / tau_c  # [N]

        logits = torch.mm(zt_flat, z_ts_flat.t()) / tau_c         # [N, N_ts]
        same_sample = (anchor_sid.unsqueeze(1) == ts_sid.unsqueeze(0))
        logits = logits.masked_fill(same_sample, -1e9)

        logits_with_pos = torch.cat([pos_scores.unsqueeze(1), logits], dim=1)  # [N, 1+N_ts]
        per_tok = (-pos_scores + torch.logsumexp(logits_with_pos, dim=1))      # [N]

        loss_nce = (per_tok * w_flat).sum() / (w_flat.sum() + self.eps)
        loss = loss_nce

        # 9) Optional regularizers (salience-weighted, masked)
        if self.lambda_cycle > 0:
            C_t = torch.bmm(P_t2s, P_s2t)  # [B, Lt, Lt]
            diag = C_t.diagonal(dim1=1, dim2=2)  # [B, Lt]
            diag = diag.clamp_min(self.eps)
            cyc = -(diag.log())
            cyc = cyc.masked_fill(~text_valid, 0.0)
            cyc_loss = (cyc * a).sum() / (a.sum() + self.eps)
            loss = loss + self.lambda_cycle * cyc_loss

        if self.lambda_entropy > 0:
            ent = -(P_t2s.clamp_min(self.eps) * P_t2s.clamp_min(self.eps).log()).sum(dim=-1)  # [B, Lt]
            ent = ent.masked_fill(~text_valid, 0.0)
            ent_loss = (ent * a).sum() / (a.sum() + self.eps)
            loss = loss + self.lambda_entropy * ent_loss

        if return_alignment:
            out = {"P_t2s": P_t2s, "P_s2t": P_s2t, "S": S, "a_text": a, "loss_nce": loss_nce.detach()}
            if self.use_topk:
                out.update({"idx_t": idx_t, "ok_t": ok_t})
            return loss, out
        return loss


class CAMEF(nn.Module):
    def __init__(
            self,
            llama_name,
            moment=None,
            kronos_model=None,
            kronos_tokenizer=None,
            seq_len=35,
            pred_len=35,
            d=4,
            window=500,
            stride=400,
            batch_size=1,
            decoder_layers=3,
            decoder_heads=8,
            use_ts_memory=True,
            max_token_num=20000,
    ):
        super(CAMEF, self).__init__()

        self.seq_len, self.pred_len = seq_len, pred_len
        self.window, self.stride = window, stride
        self.d = d
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_text_tokens = max_token_num

        # -------- LLM text encoder --------
        self.tokenizer, self.llm = _load_llama_for_feature_extraction(llama_name, use_4bit=USE_4BIT)

        self.llm_proj = nn.Sequential(
            nn.LayerNorm(LLAMA_HIDDEN),
            nn.Linear(LLAMA_HIDDEN, 1024), nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(1024, FUSE_DIM), nn.ReLU(),
            nn.Dropout(0.05),
        ).to(self.device)

        # -------- Kronos TS encoder --------
        self.kronos_available = (Kronos is not None) and (KronosTokenizer is not None) and (KronosAdapter is not None) \
                                and (kronos_model is not None) and (kronos_tokenizer is not None)
        self.kronos_tokenizer = None
        self.kronos_model = None
        self.kronos_adapter = None
        kronos_dim = FUSE_DIM  # default fallback

        if self.kronos_available:
            self.kronos_tokenizer = KronosTokenizer.from_pretrained(kronos_tokenizer)
            self.kronos_model = Kronos.from_pretrained(kronos_model)
            self.kronos_model.to(self.device).eval()
            self.kronos_adapter = KronosAdapter(model=self.kronos_model, tokenizer=self.kronos_tokenizer)

            # Alias for backward compatibility: many training utilities expect `moment_model`
            self.moment_model = self.kronos_model

            # Probe output dimension safely
            try:
                with torch.no_grad():
                    dummy = torch.zeros(1, self.seq_len, self.d, device=self.device)  # [B,L,d]
                    tok_emb, inst_emb = self.kronos_adapter.embed(
                        dummy, x_timestamp=None, which="context", fuse="avg", pool="mean", normalize=False
                    )  # expect [B,L,Dk]
                    kronos_dim = tok_emb.shape[-1]
            except Exception:
                kronos_dim = FUSE_DIM  # safe fallback
        else:
            self.moment_model = None

        moment_dim = kronos_dim

        self.ts_feat_to_fuse = nn.Sequential(
            nn.LayerNorm(moment_dim),
            nn.Linear(moment_dim, FUSE_DIM),
            nn.GELU(),
        ).to(self.device)

        self.ts_inst_to_fuse = nn.Sequential(
            nn.LayerNorm(moment_dim),
            nn.Linear(moment_dim, FUSE_DIM),
            nn.GELU(),
        ).to(self.device)

        self.embed_project = nn.Sequential(
            nn.LayerNorm(moment_dim),
            nn.Linear(moment_dim, 1024, bias=True), nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(1024, moment_dim, bias=True), nn.GELU(),
        ).to(self.device)

        # Raw TS residual projection
        self.residual_project = nn.Sequential(
            nn.Linear(self.seq_len * self.d, self.seq_len * self.d, bias=True), nn.GELU(),
            nn.Linear(self.seq_len * self.d, self.seq_len * self.d, bias=True), nn.GELU(),
            nn.Linear(self.seq_len * self.d, FUSE_DIM, bias=True)
        ).to(self.device)

        # NEW: Multi-scale temporal feature extractor
        if USE_MULTI_SCALE_TS:
            self.multi_scale_extractor = MultiScaleTemporalExtractor(
                d_input=self.d, d_output=FUSE_DIM, seq_len=self.seq_len
            ).to(self.device)

        # Fallback TS tokenization
        self.ts_tokenizer = nn.Sequential(
            nn.Conv1d(self.d, FUSE_DIM, kernel_size=1), nn.GELU(),
            nn.Conv1d(FUSE_DIM, FUSE_DIM, kernel_size=5, padding=2), nn.GELU(),
            nn.LayerNorm([FUSE_DIM, self.seq_len])
        ).to(self.device)

        # Cross-attention
        self.cross_attn_text_to_ts = nn.MultiheadAttention(
            embed_dim=FUSE_DIM, num_heads=NUM_HEADS, batch_first=True
        ).to(self.device)
        self.cross_attn_ts_to_text = nn.MultiheadAttention(
            embed_dim=FUSE_DIM, num_heads=NUM_HEADS, batch_first=True
        ).to(self.device)

        # Alignment heads
        self.align_t2s = nn.Linear(FUSE_DIM, FUSE_DIM).to(self.device)
        self.align_s2t = nn.Linear(FUSE_DIM, FUSE_DIM).to(self.device)
        self.align_loss_w = 1.0

        # Token-wise alignment
        self.token_align_module = TokenWiseAlignmentModule(
            dim=FUSE_DIM,
            proj_dim=256,
            tau_align=TrainingConfig.TOKEN_ALIGN_TAU_ALIGN,
            tau_nce=TrainingConfig.TOKEN_ALIGN_TAU_NCE,
            use_topk=TrainingConfig.TOKEN_ALIGN_USE_TOPK,
            topk_text=TrainingConfig.TOKEN_ALIGN_TOPK_TEXT,
            lambda_entropy=0.01,
            lambda_cycle=0.05,
        ).to(self.device)
        self.token_align_loss_w = TrainingConfig.TOKEN_ALIGN_LAMBDA

        # Gating
        self.bi_gate_layer = nn.Sequential(
            nn.LayerNorm(2 * FUSE_DIM),
            nn.Linear(2 * FUSE_DIM, 2 * FUSE_DIM), nn.GELU(),
            nn.Linear(2 * FUSE_DIM, 2 * FUSE_DIM),
            nn.LayerNorm(2 * FUSE_DIM),
        ).to(self.device)
        self.gate_tau = 1.0

        # Enhanced decoder with temporal shortcut
        self.decoder = TinyDecoderV3(
            context_dim=FUSE_DIM, d=self.d, pred_len=self.pred_len, width=FUSE_DIM,
            n_heads=decoder_heads, n_layers=decoder_layers, dropout=PRED_DROPOUT,
            seq_len=self.seq_len
        ).to(self.device)

        # NEW: Learnable last-value residual module
        if USE_LEARNABLE_LAST_VALUE:
            self.last_value_module = LearnableLastValueResidual(
                context_dim=FUSE_DIM, d_channels=self.d, pred_len=self.pred_len,
                init_alpha=LAST_VALUE_INIT_WEIGHT
            ).to(self.device)

        # Memory conditioning
        self.mem_type_embed = nn.Embedding(2, FUSE_DIM).to(self.device)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=FUSE_DIM, nhead=NUM_HEADS, dim_feedforward=4 * FUSE_DIM,
            dropout=0.1, activation='gelu', batch_first=True,
            norm_first=True
        )
        self.mem_encoder = nn.TransformerEncoder(enc_layer, num_layers=1).to(self.device)
        self.mem_gate = nn.Sequential(
            nn.Linear(FUSE_DIM, FUSE_DIM), nn.GELU(),
            nn.Linear(FUSE_DIM, FUSE_DIM), nn.GELU()
        ).to(self.device)
        self.mem_gate_scalar = nn.Linear(FUSE_DIM, 1).to(self.device)

        # Auxiliary heads
        self.ts_only_head = nn.Sequential(
            nn.LayerNorm(FUSE_DIM),
            nn.Linear(FUSE_DIM, FUSE_DIM),
            nn.GELU(),
            nn.Linear(FUSE_DIM, self.d * self.pred_len)
        ).to(self.device)

        self.text_only_head = nn.Sequential(
            nn.LayerNorm(FUSE_DIM),
            nn.Linear(FUSE_DIM, FUSE_DIM),
            nn.GELU(),
            nn.Linear(FUSE_DIM, self.d * self.pred_len)
        ).to(self.device)

    # ---------- utilities ----------
    def _ensure_cf(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, f"Expected 3D tensor, got {x.shape}"
        if x.size(1) == self.d:
            return x
        if x.size(-1) == self.d:
            return x.transpose(1, 2)
        raise ValueError(f"TS shape incompatible with d={self.d}: {tuple(x.shape)}")

    @torch.no_grad()
    def text_embedding(self, text: str):
        ids = self.tokenizer(text, return_tensors='pt', truncation=False, add_special_tokens=True)
        input_ids = ids['input_ids'][0]
        attn_mask_full = torch.ones_like(input_ids)
        embeddings = []
        start = 0
        while start < input_ids.shape[0]:
            end = min(start + self.window, input_ids.shape[0])
            chunk_ids = input_ids[start:end].unsqueeze(0).to(self.device)
            chunk_mask = attn_mask_full[start:end].unsqueeze(0).to(self.device)
            out = self.llm.model(
                input_ids=chunk_ids, attention_mask=chunk_mask,
                output_hidden_states=True, use_cache=False
            )
            H = out.hidden_states[-1][0]
            emb = H.mean(dim=0, keepdim=True)
            embeddings.append(emb.float().cpu())
            start += self.stride
        emb_mean = torch.mean(torch.cat(embeddings, dim=0), dim=0, keepdim=False).to(self.device)
        return emb_mean

    @torch.no_grad()
    def get_text_tokens_all(self, text: str):
        ids = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_text_tokens,
            add_special_tokens=True,
            padding="max_length",
        )
        input_ids = ids['input_ids'][0].to(self.device)
        attn_mask_full = ids['attention_mask'][0].to(self.device)

        feats = []
        start = 0
        while start < input_ids.size(0):
            end = min(start + self.window, input_ids.size(0))
            chunk_ids = input_ids[start:end].unsqueeze(0)
            chunk_mask = attn_mask_full[start:end].unsqueeze(0)

            out = self.llm.model(
                input_ids=chunk_ids,
                attention_mask=chunk_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            H = out.hidden_states[-1][0].float()
            feats.append(H)
            start += self.stride

        if len(feats) == 0:
            tokens = torch.zeros(1, 1, FUSE_DIM, device=self.device)
            mask = torch.ones(1, 1, dtype=torch.bool, device=self.device)
            return tokens, mask

        tok_h = torch.cat(feats, dim=0)
        T = self.llm_proj(tok_h)
        T = T.unsqueeze(0)
        mask = torch.zeros(1, T.size(1), dtype=torch.bool, device=self.device)
        return T, mask

    def _pad_and_stack(self, seqs, masks, pad_value=0.0):
        B = len(seqs)
        Ls = [x.size(1) for x in seqs]
        Lmax = max(Ls) if B > 0 else 1
        Fdim = seqs[0].size(-1) if B > 0 else FUSE_DIM
        X = torch.full((B, Lmax, Fdim), pad_value, device=self.device)
        M = torch.ones(B, Lmax, dtype=torch.bool, device=self.device)
        for i, (x, m) in enumerate(zip(seqs, masks)):
            L = x.size(1)
            X[i, :L] = x.squeeze(0)
            M[i, :L] = m.squeeze(0)
        return X, M

    def _masked_mean(self, X, mask):
        if mask is None:
            return X.mean(dim=1)
        w = (~mask).float().unsqueeze(-1)
        s = (X * w).sum(dim=1)
        d = w.sum(dim=1).clamp_min(1e-6)
        return s / d

    def _compute_raw_ts_residual(self, x_cf: torch.Tensor) -> torch.Tensor:
        B = x_cf.size(0)
        x_flat = x_cf.reshape(B, -1)
        raw_ts_residual = self.residual_project(x_flat)
        return raw_ts_residual

    # ----- Kronos helper for per-timestep embeddings -----
    @torch.no_grad()
    def _kronos_encode_per_timestep(self, x_cf: torch.Tensor):
        """
        x_cf: [B, d, L] (channels-first)
        Returns per-timestep embeddings from Kronos as [B, L, Dk] and instance/global [B, Dk].
        """
        assert self.kronos_available and (self.kronos_adapter is not None)
        # Kronos expects [B, L, C]
        x_blc = x_cf.transpose(1, 2).contiguous()  # [B,L,d]
        token_emb, inst_emb = self.kronos_adapter.embed(
            x_blc, x_timestamp=None, which="context", fuse="avg", pool="mean", normalize=False
        )
        # inst_emb: [B,Dk] or [B,1,Dk] -> [B,Dk]
        if isinstance(inst_emb, torch.Tensor) and inst_emb.dim() == 3 and inst_emb.size(1) == 1:
            inst_emb = inst_emb[:, 0, :]
        return token_emb, inst_emb

    def get_ts_tokens_full(self, x_cf: torch.Tensor):
        """
        x_cf: [B, d, L]  (channels-first)
        Returns:
          ts_tok:   [B, L, FUSE_DIM]   (one token per timestep; Kronos-backed if available)
          ts_mask:  [B, L]             (all False; no pads for fixed length)
          ts_inst:  [B, FUSE_DIM]      (instance/global vector)
        """
        B = x_cf.size(0)

        if self.kronos_available:
            with torch.no_grad():
                ts_token_emb, ts_instance = self._kronos_encode_per_timestep(x_cf)  # [B,L,Dk], [B,Dk]
            ts_tok = self.ts_feat_to_fuse(ts_token_emb)   # -> [B,L,FUSE]
            ts_inst = self.ts_inst_to_fuse(ts_instance)   # -> [B,FUSE]
        else:
            # Fallback: conv tokenizer to tokens; pooled instance from tokens
            tok = self.ts_tokenizer(x_cf).transpose(1, 2).contiguous()  # [B,L,FUSE]
            ts_tok = tok
            ts_inst = tok.mean(dim=1)  # [B,FUSE]

        # Add raw TS residual to TS instance embedding
        if USE_RAW_TS_RESIDUAL:
            raw_ts_residual = self._compute_raw_ts_residual(x_cf)
            ts_inst = ts_inst + RAW_TS_RESIDUAL_WEIGHT * raw_ts_residual

        # Add multi-scale features if enabled
        if USE_MULTI_SCALE_TS:
            multi_scale_feat = self.multi_scale_extractor(x_cf)
            ts_inst = ts_inst + 0.5 * multi_scale_feat  # Weighted addition

        ts_mask = torch.zeros(B, ts_tok.size(1), dtype=torch.bool, device=self.device)
        return ts_tok, ts_mask, ts_inst

    def _fuse_from_sequences(self, text_tokens, ts_tokens, text_mask=None, ts_mask=None,
                             ts_inst=None, return_analysis: bool = False):
        attn_text2ts, w_text2ts = self.cross_attn_text_to_ts(
            query=text_tokens, key=ts_tokens, value=ts_tokens,
            key_padding_mask=ts_mask if ts_mask is not None else None,
            need_weights=return_analysis, average_attn_weights=False
        )
        attn_ts2text, w_ts2text = self.cross_attn_ts_to_text(
            query=ts_tokens, key=text_tokens, value=text_tokens,
            key_padding_mask=text_mask if text_mask is not None else None,
            need_weights=return_analysis, average_attn_weights=False
        )

        t_vec_raw = self._masked_mean(text_tokens, text_mask)

        if ts_inst is not None:
            s_vec_raw = ts_inst
        else:
            s_vec_raw = self._masked_mean(ts_tokens, ts_mask)

        t_vec = F.layer_norm(t_vec_raw, (FUSE_DIM,))
        s_vec = F.layer_norm(s_vec_raw, (FUSE_DIM,))

        H_text = text_tokens + attn_text2ts
        H_ts = ts_tokens + attn_ts2text
        t_vec_residual = F.layer_norm(self._masked_mean(H_text, text_mask), (FUSE_DIM,))

        s_vec_residual_base = F.layer_norm(self._masked_mean(H_ts, ts_mask), (FUSE_DIM,))
        if USE_RAW_TS_RESIDUAL and ts_inst is not None:
            s_vec_residual = s_vec_residual_base + (ts_inst - self._masked_mean(ts_tokens, ts_mask))
            s_vec_residual = F.layer_norm(s_vec_residual, (FUSE_DIM,))
        else:
            s_vec_residual = s_vec_residual_base

        if return_analysis:
            token_align_loss, token_align_info = self.token_align_module(
                text_tokens=H_text,
                ts_tokens=H_ts,
                text_mask=text_mask,
                ts_mask=ts_mask,
                return_alignment=True,
            )
        else:
            token_align_loss = self.token_align_module(
                text_tokens=H_text,
                ts_tokens=H_ts,
                text_mask=text_mask,
                ts_mask=ts_mask,
                return_alignment=False,
            )
            token_align_info = None

        t_vec_norm = F.normalize(t_vec_residual, p=2, dim=-1)
        s_vec_norm = F.normalize(s_vec_residual, p=2, dim=-1)

        concat_all = torch.cat([t_vec_norm, s_vec_norm], dim=-1)
        gate_logits = self.bi_gate_layer(concat_all).reshape(-1, 2, FUSE_DIM)
        gate = F.softmax(gate_logits / max(self.gate_tau, 1e-6), dim=1)
        fused_emb = gate[:, 0] * t_vec_norm + gate[:, 1] * s_vec_norm

        if return_analysis:
            return fused_emb, {
                "gate_softmax": gate,
                "attn_t2s": w_text2ts,
                "attn_s2t": w_ts2text,
                "token_align_loss": token_align_loss,
                "token_align_info": token_align_info,
                "t_vec": t_vec,
                "s_vec": s_vec,
                "s_vec_residual": s_vec_residual,
                "t_vec_residual": t_vec_residual,
                "text_tokens": text_tokens,
                "ts_tokens": ts_tokens,
                "text_mask": text_mask,
                "ts_mask": ts_mask,
                "ts_inst": ts_inst,
            }
        else:
            return fused_emb, {
                "t_vec": t_vec,
                "s_vec": s_vec,
                "s_vec_residual": s_vec_residual,
                "t_vec_residual": t_vec_residual,
                "text_tokens": text_tokens,
                "ts_tokens": ts_tokens,
                "text_mask": text_mask,
                "ts_mask": ts_mask,
                "token_align_loss": token_align_loss,
                "token_align_info": token_align_info,
                "ts_inst": ts_inst,
            }

    def predict_ts_only(self, batch_series_embeddings: torch.Tensor) -> torch.Tensor:
        B = batch_series_embeddings.size(0)
        out = self.ts_only_head(batch_series_embeddings)
        out = out.view(B, self.d, self.pred_len)
        return out

    def predict_text_only(self, batch_text_embeddings: torch.Tensor) -> torch.Tensor:
        B = batch_text_embeddings.size(0)
        out = self.text_only_head(batch_text_embeddings)
        out = out.view(B, self.d, self.pred_len)
        return out

    def predict_ts_only_full_decoder(self, ts_tok: torch.Tensor, ts_mask: torch.Tensor,
                                     s_vec_residual: torch.Tensor, x_cf: torch.Tensor) -> torch.Tensor:
        s_vec_norm = F.normalize(s_vec_residual, p=2, dim=-1)
        core = self.decoder(s_vec_norm, x_cf=x_cf)

        if USE_LEARNABLE_LAST_VALUE:
            last_val = x_cf[:, :, -1:].detach()
            output, _ = self.last_value_module(core, last_val, s_vec_norm)
        else:
            output = core
        return output

    def predict_text_only_full_decoder(self, text_tok: torch.Tensor, text_mask: torch.Tensor,
                                       t_vec_residual: torch.Tensor, x_cf: torch.Tensor) -> torch.Tensor:
        t_vec_norm = F.normalize(t_vec_residual, p=2, dim=-1)
        core = self.decoder(t_vec_norm, x_cf=x_cf)

        if USE_LEARNABLE_LAST_VALUE:
            last_val = x_cf[:, :, -1:].detach()
            output, _ = self.last_value_module(core, last_val, t_vec_norm)
        else:
            output = core
        return output

    def _apply_output_refinement(self, core_output, fused_emb, x_cf):
        """Apply learnable last-value refinement if enabled."""
        if USE_LEARNABLE_LAST_VALUE:
            last_val = x_cf[:, :, -1:].detach()
            output, alpha = self.last_value_module(core_output, last_val, fused_emb)
            return output, alpha
        else:
            return core_output, None

    def forward(self, text, seq_data):
        seq_data = seq_data.float().to(self.device)

        text_tok, text_mask = self.get_text_tokens_all(text)
        x_cf = self._ensure_cf(seq_data.unsqueeze(0))
        ts_tok, ts_mask, ts_inst = self.get_ts_tokens_full(x_cf)

        fused_emb, _ = self._fuse_from_sequences(text_tok, ts_tok, text_mask, ts_mask,
                                                 ts_inst=ts_inst, return_analysis=False)

        core = self.decoder(fused_emb, x_cf=x_cf)
        result, _ = self._apply_output_refinement(core, fused_emb, x_cf)
        return result

    def predict_batch(self, batch_text, batch_seq):
        tlist, mlist = [], []
        for t in batch_text:
            T, M = self.get_text_tokens_all(t)
            tlist.append(T)
            mlist.append(M)
        text_tok, text_mask = self._pad_and_stack(tlist, mlist)

        batch_seq = batch_seq.float().to(self.device)
        x_cf = self._ensure_cf(batch_seq)
        ts_tok, ts_mask, ts_inst = self.get_ts_tokens_full(x_cf)

        fused_emb, _ = self._fuse_from_sequences(text_tok, ts_tok, text_mask, ts_mask,
                                                 ts_inst=ts_inst, return_analysis=False)
        core = self.decoder(fused_emb, x_cf=x_cf)
        output, _ = self._apply_output_refinement(core, fused_emb, x_cf)

        batch_text_embeddings = self._masked_mean(text_tok, text_mask)
        return output, batch_text_embeddings

    def predict_batch_contrastive(
            self,
            batch_text,
            batch_sent_reports,
            batch_negative_type_reports,
            batch_seq,
            return_analysis: bool = False
    ):
        tlist, mlist = [], []
        for t in batch_text:
            T, M = self.get_text_tokens_all(t)
            tlist.append(T)
            mlist.append(M)
        text_tok, text_mask = self._pad_and_stack(tlist, mlist)

        batch_text_embeddings = []
        for text in batch_text:
            batch_text_embeddings.append(self.llm_proj(self.text_embedding(text)).unsqueeze(0))
        batch_text_embeddings = torch.cat(batch_text_embeddings, dim=0)

        batch_sent_report_embeddings = []
        for sent_reports in batch_sent_reports:
            report_embeddings = []
            for report in sent_reports:
                report_embeddings.append(self.llm_proj(self.text_embedding(report)).unsqueeze(0))
            batch_sent_report_embeddings.append(report_embeddings)
        batch_sent_report_embeddings = [torch.cat(e, dim=0) for e in batch_sent_report_embeddings]
        batch_sent_report_embeddings = torch.stack(batch_sent_report_embeddings, dim=0)
        batch_sent_report_embeddings = torch.cat(
            [batch_text_embeddings.unsqueeze(1), batch_sent_report_embeddings], dim=1
        )

        batch_negative_type_report_embeddings = []
        for negative_reports in batch_negative_type_reports:
            negative_report_embeddings = []
            for negative_report in negative_reports:
                negative_report_embeddings.append(self.llm_proj(self.text_embedding(negative_report)).unsqueeze(0))
            batch_negative_type_report_embeddings.append(negative_report_embeddings)
        batch_negative_type_report_embeddings = [
            torch.cat(e, dim=0) for e in batch_negative_type_report_embeddings
        ]
        batch_negative_type_report_embeddings = torch.stack(
            batch_negative_type_report_embeddings, dim=0
        )

        batch_seq = batch_seq.float().to(self.device)
        x_cf = self._ensure_cf(batch_seq)
        ts_tok, ts_mask, ts_inst = self.get_ts_tokens_full(x_cf)

        fused_emb, analysis = self._fuse_from_sequences(
            text_tok, ts_tok, text_mask, ts_mask, ts_inst=ts_inst, return_analysis=return_analysis
        )

        core = self.decoder(fused_emb, x_cf=x_cf)
        output, alpha = self._apply_output_refinement(core, fused_emb, x_cf)

        if analysis is not None:
            analysis["x_cf"] = x_cf
            analysis["last_value_alpha"] = alpha

        batch_series_embeddings_for_contrastive = analysis.get("s_vec_residual", ts_inst)

        return (
            output,
            fused_emb,
            batch_series_embeddings_for_contrastive,
            batch_text_embeddings,
            batch_sent_report_embeddings,
            batch_negative_type_report_embeddings,
            analysis,
        )

    def predict_single_case(self, batch_text, batch_seq):
        tlist, mlist = [], []
        for t in batch_text:
            T, M = self.get_text_tokens_all(t)
            tlist.append(T)
            mlist.append(M)
        text_tok, text_mask = self._pad_and_stack(tlist, mlist)

        batch_text_embeddings = []
        for text in batch_text:
            batch_text_embeddings.append(self.llm_proj(self.text_embedding(text)).unsqueeze(0))
        batch_text_embeddings = torch.cat(batch_text_embeddings, dim=0)

        batch_seq = batch_seq.float().to(self.device)
        x_cf = self._ensure_cf(batch_seq)
        ts_tok, ts_mask, ts_inst = self.get_ts_tokens_full(x_cf)

        fused_emb, _ = self._fuse_from_sequences(text_tok, ts_tok, text_mask, ts_mask,
                                                 ts_inst=ts_inst, return_analysis=False)
        core = self.decoder(fused_emb, x_cf=x_cf)
        output, _ = self._apply_output_refinement(core, fused_emb, x_cf)
        return output, fused_emb, ts_inst, batch_text_embeddings

    def predict_one(self, text, seq_data):
        return self.forward(text, seq_data)

    def moment_last_block_parameters(self):
        mm = getattr(self, "moment_model", None)
        if mm is None:
            return []

        for path in [
            ("transformer", "layers"),
            ("encoder", "layers"),
            ("layers",),
            ("backbone", "layers"),
        ]:
            cur = mm
            ok = True
            for seg in path:
                if hasattr(cur, seg):
                    cur = getattr(cur, seg)
                else:
                    ok = False
                    break
            if ok and hasattr(cur, "__len__") and len(cur) > 0:
                last_block = cur[-1]
                return list(last_block.parameters())

        candidates = [m for m in mm.modules() if isinstance(m, torch.nn.TransformerEncoderLayer)]
        if candidates:
            return list(candidates[-1].parameters())

        last_with_params = None
        for m in mm.modules():
            owns_params = any(True for _ in m.parameters(recurse=False))
            if owns_params:
                last_with_params = m
        return list(last_with_params.parameters()) if last_with_params is not None else []

    def llm_last_block_parameters(self):
        """
        Get parameters from the last transformer block of the LLM.
        This allows fine-tuning just the last layer with a very low learning rate.
        """
        llm = getattr(self, "llm", None)
        if llm is None:
            return []

        # Try different paths to find transformer layers
        for path in [
            ("model", "layers"),  # LLaMA: model.model.layers
            ("transformer", "h"),  # GPT-2 style
            ("model", "decoder", "layers"),  # T5 decoder style
        ]:
            cur = llm
            ok = True
            for seg in path:
                if hasattr(cur, seg):
                    cur = getattr(cur, seg)
                else:
                    ok = False
                    break
            if ok and hasattr(cur, "__len__") and len(cur) > 0:
                last_block = cur[-1]
                params = list(last_block.parameters())
                if params:
                    print(f"[LLM Fine-tune] Found last block with {len(params)} parameters via path {path}")
                    return params

        # Fallback: find LlamaDecoderLayer specifically
        try:
            from transformers.models.llama.modeling_llama import LlamaDecoderLayer
            candidates = [m for m in llm.modules() if isinstance(m, LlamaDecoderLayer)]
            if candidates:
                params = list(candidates[-1].parameters())
                print(f"[LLM Fine-tune] Found LlamaDecoderLayer with {len(params)} parameters")
                return params
        except ImportError:
            pass

        print("[LLM Fine-tune] Warning: Could not find LLM last block parameters")
        return []

    def _arch_metadata(self):
        llm_cfg = getattr(self.llm, "config", None)
        llm_hidden = int(getattr(llm_cfg, "hidden_size", 0) or 0)
        llm_name = getattr(llm_cfg, "name_or_path", None)
        dec_layers = getattr(self.decoder.decoder, "num_layers", None)
        try:
            dec_heads = self.decoder.decoder.layers[0].self_attn.num_heads
        except Exception:
            dec_heads = None
        return {
            "arch": "CAMEF",
            "version": "v3-learnable-last-value-llm-last-layer-finetune",
            "use_raw_ts_residual": USE_RAW_TS_RESIDUAL,
            "use_learnable_last_value": USE_LEARNABLE_LAST_VALUE,
            "use_multi_scale_ts": USE_MULTI_SCALE_TS,
            "finetune_llm_last_layer": FINETUNE_LLM_LAST_LAYER,
            "llm_finetune_lr": LLM_FINETUNE_LR,
            "seq_len": int(self.seq_len),
            "pred_len": int(self.pred_len),
            "d": int(self.d),
            "fuse_dim": int(FUSE_DIM),
            "llama_name": llm_name,
            "llm_hidden": llm_hidden,
            "use_4bit": bool(USE_4BIT),
            "decoder_layers": int(dec_layers) if dec_layers is not None else None,
            "decoder_heads": int(dec_heads) if dec_heads is not None else None,
            "gate_tau": float(getattr(self, "gate_tau", 1.0)),
            "num_heads_cross": int(NUM_HEADS),
            "device": str(self.device),
        }

    def save_model_combined(self, save_path="model_checkpoint.pth", **kwargs):
        state_dicts = {
            "moment": self.moment_model.state_dict(),
            "llm": self.llm.state_dict(),
            "llm_proj": self.llm_proj.state_dict(),
            "embed_project": self.embed_project.state_dict(),
            "ts_feat_to_fuse": self.ts_feat_to_fuse.state_dict(),
            "ts_inst_to_fuse": self.ts_inst_to_fuse.state_dict(),
            "residual_project": self.residual_project.state_dict(),
            "cross_attn_text_to_ts": self.cross_attn_text_to_ts.state_dict(),
            "cross_attn_ts_to_text": self.cross_attn_ts_to_text.state_dict(),
            "bi_gate_layer": self.bi_gate_layer.state_dict(),
            "decoder": self.decoder.state_dict(),
            "mem_type_embed": self.mem_type_embed.state_dict(),
            "mem_encoder": self.mem_encoder.state_dict(),
            "mem_gate": self.mem_gate.state_dict(),
            "mem_gate_scalar": self.mem_gate_scalar.state_dict(),
            "ts_tokenizer": self.ts_tokenizer.state_dict(),
            "align_t2s": self.align_t2s.state_dict(),
            "align_s2t": self.align_s2t.state_dict(),
            "ts_only_head": self.ts_only_head.state_dict(),
            "text_only_head": self.text_only_head.state_dict(),
            "token_align_module": self.token_align_module.state_dict(),
        }

        if USE_MULTI_SCALE_TS:
            state_dicts["multi_scale_extractor"] = self.multi_scale_extractor.state_dict()
        if USE_LEARNABLE_LAST_VALUE:
            state_dicts["last_value_module"] = self.last_value_module.state_dict()

        checkpoint = {
            "state_dicts": state_dicts,
            "metadata": {**self._arch_metadata(), **kwargs},
        }
        torch.save(checkpoint, save_path)
        print(f"[save] Model weights saved to: {save_path}")
        tok_dir = os.path.join(os.path.dirname(save_path), "tokenizer")
        os.makedirs(tok_dir, exist_ok=True)
        self.tokenizer.save_pretrained(tok_dir)
        print(f"[save] Tokenizer saved to: {tok_dir}")

    def load_model_combined(self, save_path="model_checkpoint.pth", strict=True):
        ckpt = torch.load(save_path, map_location=self.device)
        sd = ckpt["state_dicts"]
        meta = ckpt.get("metadata", {})

        if "moment" in sd and (sd["moment"] is not None):
            self.moment_model.load_state_dict(sd["moment"], strict=True)

        self.llm.load_state_dict(sd["llm"], strict=True)
        self.llm_proj.load_state_dict(sd["llm_proj"], strict=False)
        self.embed_project.load_state_dict(sd["embed_project"], strict=False)
        if "ts_feat_to_fuse" in sd and sd["ts_feat_to_fuse"] is not None:
            self.ts_feat_to_fuse.load_state_dict(sd["ts_feat_to_fuse"], strict=False)
        if "ts_inst_to_fuse" in sd and sd["ts_inst_to_fuse"] is not None:
            self.ts_inst_to_fuse.load_state_dict(sd["ts_inst_to_fuse"], strict=False)
        self.residual_project.load_state_dict(sd["residual_project"], strict=False)
        self.cross_attn_text_to_ts.load_state_dict(sd["cross_attn_text_to_ts"], strict=False)
        self.cross_attn_ts_to_text.load_state_dict(sd["cross_attn_ts_to_text"], strict=False)
        self.bi_gate_layer.load_state_dict(sd["bi_gate_layer"], strict=False)
        self.decoder.load_state_dict(sd["decoder"], strict=False)

        if USE_MULTI_SCALE_TS and "multi_scale_extractor" in sd:
            self.multi_scale_extractor.load_state_dict(sd["multi_scale_extractor"], strict=False)
        if USE_LEARNABLE_LAST_VALUE and "last_value_module" in sd:
            self.last_value_module.load_state_dict(sd["last_value_module"], strict=False)

        self.mem_type_embed.load_state_dict(sd["mem_type_embed"], strict=False)
        self.mem_encoder.load_state_dict(sd["mem_encoder"], strict=False)
        self.mem_gate.load_state_dict(sd["mem_gate"], strict=False)
        self.mem_gate_scalar.load_state_dict(sd["mem_gate_scalar"], strict=False)
        self.ts_tokenizer.load_state_dict(sd["ts_tokenizer"], strict=False)
        self.align_t2s.load_state_dict(sd["align_t2s"], strict=False)
        self.align_s2t.load_state_dict(sd["align_s2t"], strict=False)
        if "ts_only_head" in sd and sd["ts_only_head"] is not None:
            self.ts_only_head.load_state_dict(sd["ts_only_head"], strict=False)
        if "text_only_head" in sd and sd["text_only_head"] is not None:
            self.text_only_head.load_state_dict(sd["text_only_head"], strict=False)
        if "token_align_module" in sd and sd["token_align_module"] is not None:
            self.token_align_module.load_state_dict(sd["token_align_module"], strict=False)

        self.align_loss_w = float(meta.get("align_loss_w", getattr(self, "align_loss_w", 1.0)))
        self.gate_tau = float(meta.get("gate_tau", 1.0))

        tok_dir = os.path.join(os.path.dirname(save_path), "tokenizer")
        if os.path.isdir(tok_dir):
            self.tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[load] Loaded checkpoint: {save_path}")


# ===================== Loss & Utils =====================

def smape(y_hat: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num = (y_hat - y_true).abs()
    den = (y_hat.abs() + y_true.abs()).clamp_min(eps)
    return (2.0 * (num / den)).mean() * 100.0


def contrastive_loss_objective(
        anchor, batch_sent_report_embeddings, batch_negative_type_report_embeddings,
        margin=0.2, temperature=0.5, use_hard_negative=True,
        w_ce=0.5, w_triplet=0.5
):
    anchor = F.normalize(anchor, dim=-1)
    pos = F.normalize(batch_sent_report_embeddings[:, 0, :], dim=-1)

    neg_sent = batch_sent_report_embeddings[:, 1:, :]
    neg_type = batch_negative_type_report_embeddings
    neg_bank = torch.cat([neg_sent, neg_type], dim=1) if neg_type.numel() > 0 else neg_sent
    neg_bank = F.normalize(neg_bank, dim=-1)

    candidates = torch.cat([pos.unsqueeze(1), neg_bank], dim=1)
    logits = torch.einsum('bd,bkd->bk', anchor, candidates) / max(temperature, 1e-6)
    labels = torch.zeros(anchor.size(0), dtype=torch.long, device=logits.device)
    ce_loss = F.cross_entropy(logits, labels)

    if neg_bank.size(1) > 0:
        if use_hard_negative:
            sim_neg = torch.einsum('bd,bkd->bk', anchor, neg_bank)
            hard_idx = torch.argmax(sim_neg, dim=1)
            neg = neg_bank[torch.arange(anchor.size(0), device=neg_bank.device), hard_idx]
        else:
            neg = neg_bank.mean(dim=1)
        pos_dist = 1.0 - (anchor * pos).sum(dim=-1)
        neg_dist = 1.0 - (anchor * neg).sum(dim=-1)
        triplet_loss = F.relu(pos_dist - neg_dist + margin).mean()
    else:
        triplet_loss = torch.zeros([], device=anchor.device)

    return w_ce * ce_loss + w_triplet * triplet_loss


def make_horizon_weights(L, device, decay=0.06):
    h = torch.arange(L, device=device).float()
    w = torch.exp(-decay * h)
    return w / w.sum()


def weighted_mse_mae(y_hat, y_true, h_w):
    se = (y_hat - y_true) ** 2
    l1 = (y_hat - y_true).abs()
    se = (se * h_w).sum(dim=-1).mean()
    l1 = (l1 * h_w).sum(dim=-1).mean()
    return se, l1


# ============================= Train / Eval ==================================
def train(model, train_loader, test_loader, vali_loader, num_epochs, output_path=None,
          use_horizon_weight=True, horizon_decay=0.00,
          contrastive_lambda=None,
          gate_causal_lambda=None,
          inv_lambda=None,
          eval_every_steps: int = 100,
          simplified_mode: bool = None):
    if contrastive_lambda is None:
        contrastive_lambda = TrainingConfig.CONTRASTIVE_LAMBDA
    if gate_causal_lambda is None:
        gate_causal_lambda = TrainingConfig.GATE_CAUSAL_LAMBDA
    if inv_lambda is None:
        inv_lambda = TrainingConfig.INV_LAMBDA
    if simplified_mode is None:
        simplified_mode = TrainingConfig.SIMPLIFIED_MODE

    if simplified_mode:
        print("[SIMPLIFIED MODE] Training with only core regression loss")
        contrastive_lambda = 0.0
        gate_causal_lambda = 0.0
        inv_lambda = 0.0
        model.token_align_loss_w = 0.0

    print(f"[CONFIG] USE_RAW_TS_RESIDUAL: {USE_RAW_TS_RESIDUAL}")
    print(f"[CONFIG] USE_LEARNABLE_LAST_VALUE: {USE_LEARNABLE_LAST_VALUE}")
    print(f"[CONFIG] USE_MULTI_SCALE_TS: {USE_MULTI_SCALE_TS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    # Freeze / unfreeze
    for p in model.parameters():
        p.requires_grad = True
    for p in model.llm.parameters():
        p.requires_grad = False
    for p in model.llm_proj.parameters():
        p.requires_grad = True

    # Optionally unfreeze LLM's last transformer block for fine-tuning
    llm_last_block_params = []
    if FINETUNE_LLM_LAST_LAYER:
        llm_last_block_params = model.llm_last_block_parameters()
        for p in llm_last_block_params:
            p.requires_grad = True
        if llm_last_block_params:
            print(f"[CONFIG] FINETUNE_LLM_LAST_LAYER: True ({len(llm_last_block_params)} params, LR={LLM_FINETUNE_LR})")
        else:
            print("[CONFIG] FINETUNE_LLM_LAST_LAYER: True (but no params found)")
    else:
        print("[CONFIG] FINETUNE_LLM_LAST_LAYER: False")

    if model.moment_model is not None:
        for p in model.moment_model.parameters():
            p.requires_grad = False
        # for p in model.moment_last_block_parameters():
        #     p.requires_grad = True
    for p in model.decoder.parameters():
        p.requires_grad = True
    for p in model.bi_gate_layer.parameters():
        p.requires_grad = True
    for p in model.cross_attn_text_to_ts.parameters():
        p.requires_grad = True
    for p in model.cross_attn_ts_to_text.parameters():
        p.requires_grad = True
    for p in model.residual_project.parameters():
        p.requires_grad = True
    for p in model.embed_project.parameters():
        p.requires_grad = True
    for p in model.mem_type_embed.parameters():
        p.requires_grad = True
    for p in model.mem_encoder.parameters():
        p.requires_grad = True
    for p in model.mem_gate.parameters():
        p.requires_grad = True
    for p in model.mem_gate_scalar.parameters():
        p.requires_grad = True
    for p in model.ts_tokenizer.parameters():
        p.requires_grad = True
    for p in model.ts_feat_to_fuse.parameters():
        p.requires_grad = True
    for p in model.ts_inst_to_fuse.parameters():
        p.requires_grad = True
    for p in model.ts_only_head.parameters():
        p.requires_grad = True
    for p in model.text_only_head.parameters():
        p.requires_grad = True
    for p in model.token_align_module.parameters():
        p.requires_grad = True

    if USE_MULTI_SCALE_TS:
        for p in model.multi_scale_extractor.parameters():
            p.requires_grad = True
    if USE_LEARNABLE_LAST_VALUE:
        for p in model.last_value_module.parameters():
            p.requires_grad = True

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    criterion_huber = nn.SmoothL1Loss(beta=0.5)

    base_lr = TrainingConfig.BASE_LR
    trainable_groups = [
        {'params': model.llm_proj.parameters(), 'lr': base_lr, 'weight_decay': 0.01},
        {'params': model.embed_project.parameters(), 'lr': base_lr, 'weight_decay': 0.01},
        {'params': model.residual_project.parameters(), 'lr': base_lr, 'weight_decay': 0.01},
        {'params': model.cross_attn_text_to_ts.parameters(), 'lr': base_lr, 'weight_decay': 0.01},
        {'params': model.cross_attn_ts_to_text.parameters(), 'lr': base_lr, 'weight_decay': 0.01},
        {'params': model.bi_gate_layer.parameters(), 'lr': base_lr, 'weight_decay': 0.01},
        {'params': model.decoder.parameters(), 'lr': base_lr, 'weight_decay': 0.01},
        {'params': model.mem_type_embed.parameters(), 'lr': base_lr, 'weight_decay': 0.01},
        {'params': model.mem_encoder.parameters(), 'lr': base_lr, 'weight_decay': 0.01},
        {'params': model.mem_gate.parameters(), 'lr': base_lr, 'weight_decay': 0.01},
        {'params': model.mem_gate_scalar.parameters(), 'lr': base_lr, 'weight_decay': 0.01},
        {'params': model.ts_tokenizer.parameters(), 'lr': base_lr, 'weight_decay': 0.01},
        {'params': model.ts_feat_to_fuse.parameters(), 'lr': base_lr, 'weight_decay': 0.01},
        {'params': model.ts_inst_to_fuse.parameters(), 'lr': base_lr, 'weight_decay': 0.01},
        {'params': model.ts_only_head.parameters(), 'lr': base_lr, 'weight_decay': 0.01},
        {'params': model.text_only_head.parameters(), 'lr': base_lr, 'weight_decay': 0.01},
        {'params': model.token_align_module.parameters(), 'lr': base_lr, 'weight_decay': 0.01},
    ]

    if USE_MULTI_SCALE_TS:
        trainable_groups.append(
            {'params': model.multi_scale_extractor.parameters(), 'lr': base_lr, 'weight_decay': 0.01})
    if USE_LEARNABLE_LAST_VALUE:
        trainable_groups.append({'params': model.last_value_module.parameters(), 'lr': base_lr * 2,
                                 'weight_decay': 0.01})  # Higher LR for last value module

    # Add LLM last block parameters with very low learning rate
    if FINETUNE_LLM_LAST_LAYER and len(llm_last_block_params) > 0:
        trainable_groups.append({
            'params': llm_last_block_params,
            'lr': LLM_FINETUNE_LR,  # Very low LR (1e-6 by default)
            'weight_decay': 0.01
        })

    optimizer = optim.AdamW(trainable_groups, betas=(0.9, 0.999), eps=1e-8)
    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = int(TrainingConfig.WARMUP_RATIO * num_training_steps) if num_training_steps > 0 else 0
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    if USE_EMA:
        ema_params = [p.clone().detach() for p in model.parameters() if p.requires_grad]
        for p in ema_params:
            p.requires_grad_(False)

        def ema_update():
            i = 0
            for p in model.parameters():
                if p.requires_grad:
                    ema_params[i].mul_(EMA_DECAY).add_(p.detach(), alpha=1 - EMA_DECAY)
                    i += 1

        def swap_to_ema():
            i = 0
            backup = []
            for p in model.parameters():
                if p.requires_grad:
                    backup.append(p.data.clone())
                    p.data.copy_(ema_params[i])
                    i += 1
            return backup

        def swap_from_backup(backup):
            i = 0
            for p in model.parameters():
                if p.requires_grad:
                    p.data.copy_(backup[i])
                    i += 1
    else:
        def ema_update():
            pass

        def swap_to_ema():
            return []

        def swap_from_backup(_):
            pass

    # NOTE: Logging is handled globally via DualLogger in train_three_stage (sys.stdout redirection).
    # Keep a single source of truth: plain print() statements.
    model_output_path = output_path or "./model_output"
    Path(model_output_path).mkdir(parents=True, exist_ok=True)

    best_loss = float('inf')
    branch_names = ["Text", "TS"]
    h_w = make_horizon_weights(model.pred_len, device, decay=horizon_decay) if use_horizon_weight else None

    global_step = 0
    last_vali_metrics = None
    last_test_metrics = None

    # Subsampled training-modality evaluation (to avoid an expensive full pass over train_loader).
    # Set to 0/None to disable, or increase for less overhead.
    train_modality_every_steps = 10

    def _reset_window_accumulators():
        return {
            "gate_sum": torch.zeros(2, dtype=torch.float32),
            "gate_count": 0,
            "gc_scores": [],
            "gc_scores_raw": [],
            "gate_texts": [],
            "train_loss_sum": 0.0,
            "train_smape_sum": 0.0,
            # Forecasting metrics (aligned with test()/vali outputs)
            "train_mse_sum": 0.0,
            "train_mae_sum": 0.0,
            "train_steps": 0,
            "core_reg_sum": 0.0,
            "contrastive_sum": 0.0,
            "gate_causal_sum": 0.0,
            "token_align_sum": 0.0,
            "ts_only_sum": 0.0,
            "alpha_sum": 0.0,
            "alpha_count": 0,
        }

    window = _reset_window_accumulators()

    def _summarize_gate_and_causal(window_dict, tag: str = None, step: int = None):
        msgs = []
        if window_dict["gate_count"] > 0:
            gate_avg = (window_dict["gate_sum"] / window_dict["gate_count"]).tolist()
            gate_msg = "[Gate (softmax) openness | last window] " + " | ".join(
                f"{name}: {gate_avg[i]:.4f}" for i, name in enumerate(branch_names)
            )
            msgs.append(gate_msg)

        if len(window_dict["gc_scores"]) > 0:
            gc_all = torch.cat(window_dict["gc_scores"])
            gate_all = torch.cat(window_dict["gate_texts"])
            gc_mean = gc_all.mean().item()
            gate_text_mean = gate_all.mean().item()
            gc_improve_ratio = (gc_all > 0.5).float().mean().item()
            if gc_all.numel() > 1:
                mat = torch.corrcoef(torch.stack([gc_all, gate_all]))
                corr = mat[0, 1].item()
            else:
                corr = float('nan')

            gc_raw_mean = 0.0
            if len(window_dict["gc_scores_raw"]) > 0:
                gc_raw_all = torch.cat(window_dict["gc_scores_raw"])
                gc_raw_mean = gc_raw_all.mean().item()

            msg_causal = (
                f"[Causal | last window] mean Granger score: {gc_mean:.4f} (sigmoid), {gc_raw_mean:.6f} (raw), "
                f"mean text gate: {gate_text_mean:.4f}, "
                f"frac(text improves TS-only): {gc_improve_ratio:.4f}, "
                f"corr(gate,text_help): {corr:.4f}"
            )
            msgs.append(msg_causal)

        if window_dict["train_steps"] > 0:
            w_loss = window_dict["train_loss_sum"] / window_dict["train_steps"]
            w_smape = window_dict["train_smape_sum"] / window_dict["train_steps"]
            w_mse = window_dict.get("train_mse_sum", 0.0) / window_dict["train_steps"]
            w_mae = window_dict.get("train_mae_sum", 0.0) / window_dict["train_steps"]
            w_rmse = math.sqrt(w_mse)
            if tag is not None and step is not None:
                msgs.append(
                    f"[{tag}] Step {step} | Train: Comb {w_loss:.6f}, MSE {w_mse:.6f}, RMSE {w_rmse:.6f}, "
                    f"MAE {w_mae:.6f}, SMAPE(%) {w_smape:.4f}"
                )
            else:
                msgs.append(
                    f"[Train | last window] avg loss: {w_loss:.6f}, MSE {w_mse:.6f}, RMSE {w_rmse:.6f}, "
                    f"MAE {w_mae:.6f}, SMAPE(%) {w_smape:.4f}"
                )

            w_core = window_dict["core_reg_sum"] / window_dict["train_steps"]
            w_contr = window_dict["contrastive_sum"] / window_dict["train_steps"]
            w_gate = window_dict["gate_causal_sum"] / window_dict["train_steps"]
            w_align = window_dict["token_align_sum"] / window_dict["train_steps"]
            w_ts = window_dict["ts_only_sum"] / window_dict["train_steps"]
            msgs.append(
                f"[Loss Components] core_reg: {w_core:.4f}, contrastive: {w_contr:.4f}, "
                f"gate_causal: {w_gate:.4f}, token_align: {w_align:.4f}, ts_only: {w_ts:.4f}"
            )

            if window_dict["alpha_count"] > 0:
                avg_alpha = window_dict["alpha_sum"] / window_dict["alpha_count"]
                msgs.append(f"[Last-Value Alpha] avg: {avg_alpha:.4f} (higher = more weight on last value)")
        return msgs

    def _do_eval_and_log(tag: str):
        nonlocal last_vali_metrics, last_test_metrics

        for m in _summarize_gate_and_causal(window, tag=tag, step=global_step):
            print(m)

        _backup = swap_to_ema()
        vali_result = test(
            model, vali_loader, use_horizon_weight=use_horizon_weight, horizon_decay=horizon_decay,
            contrastive_lambda=contrastive_lambda, compute_modality_scores=True
        )
        test_result = test(
            model, test_loader, use_horizon_weight=use_horizon_weight, horizon_decay=horizon_decay,
            contrastive_lambda=contrastive_lambda, compute_modality_scores=True
        )
        swap_from_backup(_backup)

        (vali_combine_loss, vali_mse_loss, vali_mae_loss, vali_contras_loss, vali_rmse_loss,
         vali_smape_loss, vali_modality) = vali_result
        (test_combine_loss, test_mse_loss, test_mae_loss, test_contras_loss, test_rmse_loss,
         test_smape_loss, test_modality) = test_result

        last_vali_metrics = (vali_combine_loss, vali_mse_loss, vali_mae_loss, vali_contras_loss,
                             vali_rmse_loss, vali_smape_loss)
        last_test_metrics = (test_combine_loss, test_mse_loss, test_mae_loss, test_contras_loss,
                             test_rmse_loss, test_smape_loss)

        msg = (
            f"\n[{tag}] Step {global_step} | "
            f"Vali: Comb {vali_combine_loss:.6f}, MSE {vali_mse_loss:.6f}, RMSE {vali_rmse_loss:.6f}, "
            f"MAE {vali_mae_loss:.6f}, Contr {vali_contras_loss:.6f}, SMAPE(%) {vali_smape_loss:.4f}\n"
            f"[{tag}] Step {global_step} | "
            f"Test: Comb {test_combine_loss:.6f}, MSE {test_mse_loss:.6f}, RMSE {test_rmse_loss:.6f}, "
            f"MAE {test_mae_loss:.6f}, Contr {test_contras_loss:.6f}, SMAPE(%) {test_smape_loss:.4f}\n"
        )
        print(msg)

        if vali_modality is not None and test_modality is not None:
            vali_text_pct = vali_modality.get('granger_text_helps_pct', 0.0) * 100
            vali_ts_pct = vali_modality.get('granger_ts_helps_pct', 0.0) * 100
            test_text_pct = test_modality.get('granger_text_helps_pct', 0.0) * 100
            test_ts_pct = test_modality.get('granger_ts_helps_pct', 0.0) * 100

            vali_text_raw = vali_modality.get('granger_text_helps_raw', 0.0)
            vali_ts_raw = vali_modality.get('granger_ts_helps_raw', 0.0)
            test_text_raw = test_modality.get('granger_text_helps_raw', 0.0)
            test_ts_raw = test_modality.get('granger_ts_helps_raw', 0.0)

            vali_gate_text = vali_modality.get('gate_text', 0.0)
            vali_gate_ts = vali_modality.get('gate_ts', 0.0)
            test_gate_text = test_modality.get('gate_text', 0.0)
            test_gate_ts = test_modality.get('gate_ts', 0.0)

            # Ablated modality metrics (4 TS metrics)
            vali_ts_only_mse = float(vali_modality.get('ts_only_mse', 0.0))
            vali_ts_only_rmse = float(vali_modality.get('ts_only_rmse', math.sqrt(vali_ts_only_mse)))
            vali_ts_only_mae = float(vali_modality.get('ts_only_mae', 0.0))
            vali_ts_only_smape = float(vali_modality.get('ts_only_smape', 0.0))

            vali_text_only_mse = float(vali_modality.get('text_only_mse', 0.0))
            vali_text_only_rmse = float(vali_modality.get('text_only_rmse', math.sqrt(vali_text_only_mse)))
            vali_text_only_mae = float(vali_modality.get('text_only_mae', 0.0))
            vali_text_only_smape = float(vali_modality.get('text_only_smape', 0.0))

            test_ts_only_mse = float(test_modality.get('ts_only_mse', 0.0))
            test_ts_only_rmse = float(test_modality.get('ts_only_rmse', math.sqrt(test_ts_only_mse)))
            test_ts_only_mae = float(test_modality.get('ts_only_mae', 0.0))
            test_ts_only_smape = float(test_modality.get('ts_only_smape', 0.0))

            test_text_only_mse = float(test_modality.get('text_only_mse', 0.0))
            test_text_only_rmse = float(test_modality.get('text_only_rmse', math.sqrt(test_text_only_mse)))
            test_text_only_mae = float(test_modality.get('text_only_mae', 0.0))
            test_text_only_smape = float(test_modality.get('text_only_smape', 0.0))

            modality_msg = (
                f"[{tag}] Step {global_step} | Vali Gate (softmax) openness: Text: {vali_gate_text:.4f} | TS: {vali_gate_ts:.4f}\n"
                f"[{tag}] Step {global_step} | Vali TS Metrics: MSE {vali_mse_loss:.6f} | RMSE {vali_rmse_loss:.6f} | MAE {vali_mae_loss:.6f} | SMAPE(%) {vali_smape_loss:.4f}\n"
                f"[{tag}] Step {global_step} | Vali Ablation Metrics:\n"
                f"  - TS-only:   MSE {vali_ts_only_mse:.6f} | RMSE {vali_ts_only_rmse:.6f} | MAE {vali_ts_only_mae:.6f} | SMAPE(%) {vali_ts_only_smape:.4f}\n"
                f"  - Text-only: MSE {vali_text_only_mse:.6f} | RMSE {vali_text_only_rmse:.6f} | MAE {vali_text_only_mae:.6f} | SMAPE(%) {vali_text_only_smape:.4f}\n"
                f"[{tag}] Step {global_step} | Vali Modality Scores:\n"
                f"  - Text helps (vs TS-only): {vali_text_pct:+.2f}% error reduction, raw: {vali_text_raw:.6f}\n"
                f"  - TS helps (vs Text-only): {vali_ts_pct:+.2f}% error reduction, raw: {vali_ts_raw:.6f}\n"
                f"[{tag}] Step {global_step} | Test Gate (softmax) openness: Text: {test_gate_text:.4f} | TS: {test_gate_ts:.4f}\n"
                f"[{tag}] Step {global_step} | Test TS Metrics: MSE {test_mse_loss:.6f} | RMSE {test_rmse_loss:.6f} | MAE {test_mae_loss:.6f} | SMAPE(%) {test_smape_loss:.4f}\n"
                f"[{tag}] Step {global_step} | Test Ablation Metrics:\n"
                f"  - TS-only:   MSE {test_ts_only_mse:.6f} | RMSE {test_ts_only_rmse:.6f} | MAE {test_ts_only_mae:.6f} | SMAPE(%) {test_ts_only_smape:.4f}\n"
                f"  - Text-only: MSE {test_text_only_mse:.6f} | RMSE {test_text_only_rmse:.6f} | MAE {test_text_only_mae:.6f} | SMAPE(%) {test_text_only_smape:.4f}\n"
                f"[{tag}] Step {global_step} | Test Modality Scores:\n"
                f"  - Text helps (vs TS-only): {test_text_pct:+.2f}% error reduction, raw: {test_text_raw:.6f}\n"
                f"  - TS helps (vs Text-only): {test_ts_pct:+.2f}% error reduction, raw: {test_ts_raw:.6f}\n"
            )
            print(modality_msg)


            # ---- Extra horizon metrics (prefix horizons), computed inside test() ----
            vali_hres = vali_modality.get('horizon_results') if isinstance(vali_modality, dict) else None
            test_hres = test_modality.get('horizon_results') if isinstance(test_modality, dict) else None
            if isinstance(vali_hres, dict) and isinstance(test_hres, dict) and len(vali_hres) > 0 and len(test_hres) > 0:
                common_h = sorted(set(vali_hres.keys()) & set(test_hres.keys()))
                for hh in common_h:
                    v = vali_hres[hh]
                    t = test_hres[hh]

                    msg_h = (
                        f"[{tag}] Step {global_step} | Vali(H={hh}): Comb {v['combined_loss']:.6f}, MSE {v['mse']:.6f}, RMSE {v['rmse']:.6f}, "
                        f"MAE {v['mae']:.6f}, Contr {v['contrastive']:.6f}, SMAPE(%) {v['smape']:.4f}\n"
                        f"[{tag}] Step {global_step} | Test(H={hh}): Comb {t['combined_loss']:.6f}, MSE {t['mse']:.6f}, RMSE {t['rmse']:.6f}, "
                        f"MAE {t['mae']:.6f}, Contr {t['contrastive']:.6f}, SMAPE(%) {t['smape']:.4f}\n"
                        f"[{tag}] Step {global_step} | Vali Ablation Metrics (H={hh}):\n"
                        f"  - TS-only:   MSE {v['ts_only_mse']:.6f} | RMSE {v['ts_only_rmse']:.6f} | MAE {v['ts_only_mae']:.6f} | SMAPE(%) {v['ts_only_smape']:.4f}\n"
                        f"  - Text-only: MSE {v['text_only_mse']:.6f} | RMSE {v['text_only_rmse']:.6f} | MAE {v['text_only_mae']:.6f} | SMAPE(%) {v['text_only_smape']:.4f}\n"
                        f"[{tag}] Step {global_step} | Vali Modality Scores (H={hh}):\n"
                        f"  - Text helps (vs TS-only): {v['granger_text_helps_pct']*100:+.2f}% error reduction, raw: {v['granger_text_helps_raw']:.6f}\n"
                        f"  - TS helps (vs Text-only): {v['granger_ts_helps_pct']*100:+.2f}% error reduction, raw: {v['granger_ts_helps_raw']:.6f}\n"
                        f"[{tag}] Step {global_step} | Test Ablation Metrics (H={hh}):\n"
                        f"  - TS-only:   MSE {t['ts_only_mse']:.6f} | RMSE {t['ts_only_rmse']:.6f} | MAE {t['ts_only_mae']:.6f} | SMAPE(%) {t['ts_only_smape']:.4f}\n"
                        f"  - Text-only: MSE {t['text_only_mse']:.6f} | RMSE {t['text_only_rmse']:.6f} | MAE {t['text_only_mae']:.6f} | SMAPE(%) {t['text_only_smape']:.4f}\n"
                        f"[{tag}] Step {global_step} | Test Modality Scores (H={hh}):\n"
                        f"  - Text helps (vs TS-only): {t['granger_text_helps_pct']*100:+.2f}% error reduction, raw: {t['granger_text_helps_raw']:.6f}\n"
                        f"  - TS helps (vs Text-only): {t['granger_ts_helps_pct']*100:+.2f}% error reduction, raw: {t['granger_ts_helps_raw']:.6f}\n"
                    )
                    print(msg_h)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_smape_sum = 0.0
        # Forecasting metrics for the training set (aligned with test()/vali outputs)
        epoch_mse_sum = 0.0
        epoch_mae_sum = 0.0
        epoch_contrastive_sum = 0.0

        # Subsampled modality metrics on training batches (approximate, to keep runtime reasonable)
        epoch_modality_batches = 0
        epoch_ts_only_mse_sum = 0.0
        epoch_ts_only_mae_sum = 0.0
        epoch_ts_only_smape_sum = 0.0
        epoch_text_only_mse_sum = 0.0
        epoch_text_only_mae_sum = 0.0
        epoch_text_only_smape_sum = 0.0
        epoch_granger_text_helps_pct_sum = 0.0
        epoch_granger_ts_helps_pct_sum = 0.0
        epoch_granger_text_helps_raw_sum = 0.0
        epoch_granger_ts_helps_raw_sum = 0.0

        epoch_gate_sum = torch.zeros(2, dtype=torch.float32)
        epoch_gate_count = 0

        for step_in_epoch, (batch_text, batch_sent_reports, batch_negative_type_reports,
                            batch_seq, batch_pred, batch_seq_scale, batch_pred_scale) in enumerate(
            train_loader, start=1):

            global_step += 1
            optimizer.zero_grad()

            batch_sent_reports = list(map(list, zip(*batch_sent_reports)))
            batch_negative_type_reports = list(map(list, zip(*batch_negative_type_reports)))

            (output,
             batch_fusion_embeddings,
             batch_series_embeddings,
             batch_text_embeddings,
             batch_sent_report_embeddings,
             batch_negative_type_report_embeddings,
             analysis) = model.predict_batch_contrastive(
                batch_text, batch_sent_reports, batch_negative_type_reports, batch_seq_scale,
                return_analysis=True
            )

            target = batch_pred_scale.float().to(device)
            if h_w is not None:
                mse_w, mae_w = weighted_mse_mae(output.float(), target, h_w)
                core_reg = 0.7 * mse_w + 0.3 * mae_w
            else:
                mse_loss = criterion_mse(target, output.float())
                mae_loss = criterion_mae(target, output.float())
                huber = criterion_huber(target, output.float())
                core_reg = 0.5 * mse_loss + 0.3 * mae_loss + 0.2 * huber

            combined_loss = core_reg

            loss_core_reg = float(core_reg.detach().cpu())
            loss_contrastive = 0.0
            contrastive_loss_raw = 0.0  # unscaled (aligned with test()/vali 'Contr')
            loss_gate_causal = 0.0
            loss_token_align = 0.0
            loss_ts_only = 0.0

            if contrastive_lambda > 0:
                ts_anchor = analysis["s_vec_residual"] if (
                        analysis is not None and "s_vec_residual" in analysis) else batch_series_embeddings
                contrastive_loss = contrastive_loss_objective(
                    ts_anchor, batch_sent_report_embeddings, batch_negative_type_report_embeddings,
                    w_ce=0.0, w_triplet=1.0, temperature=0.5, margin=1.0
                )
                contrastive_loss_raw = float(contrastive_loss.detach().cpu())
                combined_loss = combined_loss + contrastive_lambda * contrastive_loss
                loss_contrastive = float(contrastive_loss.detach().cpu()) * contrastive_lambda

            if gate_causal_lambda > 0 and analysis is not None and ("gate_softmax" in analysis):
                gate = analysis["gate_softmax"]
                gate_text_scalar = gate[:, 0, :].mean(dim=-1)

                s_vec_residual = analysis.get("s_vec_residual")
                if s_vec_residual is not None:
                    with torch.no_grad():
                        ts_only_pred = model.predict_ts_only_full_decoder(
                            analysis["ts_tokens"], analysis["ts_mask"], s_vec_residual, analysis["x_cf"]
                        )
                    err_ts = ((ts_only_pred - target) ** 2).mean(dim=(1, 2))
                    err_full = ((output - target) ** 2).mean(dim=(1, 2))

                    diff = (err_ts - err_full)

                    if TrainingConfig.GC_USE_ADAPTIVE_SCALE:
                        s = torch.clamp(diff.detach().abs().mean(), min=TrainingConfig.GC_MIN_SCALE)
                    else:
                        s = torch.tensor(TrainingConfig.GC_SIGMOID_SCALE, device=diff.device, dtype=diff.dtype)

                    logit = TrainingConfig.GC_GAIN * diff / s
                    logit = torch.clamp(logit, -TrainingConfig.GC_LOGIT_CLAMP, TrainingConfig.GC_LOGIT_CLAMP)

                    gc_score = torch.sigmoid(logit).detach()

                    # gc_score = torch.sigmoid(err_ts - err_full).detach()

                    gate_gc_loss = F.mse_loss(gate_text_scalar, gc_score)
                    combined_loss = combined_loss + gate_causal_lambda * gate_gc_loss
                    loss_gate_causal = float(gate_gc_loss.detach().cpu()) * gate_causal_lambda

                    window["gc_scores"].append(gc_score.detach().cpu())
                    window["gc_scores_raw"].append(diff.detach().cpu())
                    window["gate_texts"].append(gate_text_scalar.detach().cpu())

            if inv_lambda > 0:
                if batch_sent_report_embeddings.size(1) > 1:
                    cf_style_emb = batch_sent_report_embeddings[:, 1, :]
                    inv_loss = F.mse_loss(batch_text_embeddings, cf_style_emb)
                    combined_loss = combined_loss + inv_lambda * inv_loss

            if not simplified_mode and analysis is not None and ("gate_softmax" in analysis):
                gate = analysis["gate_softmax"]
                gate_entropy = -(gate.clamp_min(1e-8) * gate.clamp_min(1e-8).log()).sum(dim=1).mean()
                combined_loss = combined_loss + TrainingConfig.GATE_ENTROPY_LAMBDA * gate_entropy  # penalize entropy => sharper gate

            if model.token_align_loss_w > 0 and analysis is not None and ("token_align_loss" in analysis) and (
                    analysis["token_align_loss"] is not None):
                token_align_loss = analysis["token_align_loss"]
                combined_loss = combined_loss + model.token_align_loss_w * token_align_loss
                loss_token_align = float(token_align_loss.detach().cpu()) * model.token_align_loss_w

            ts_only_lambda = TrainingConfig.TS_ONLY_LAMBDA if not simplified_mode else 0.0
            if ts_only_lambda > 0 and analysis is not None:
                s_vec_residual = analysis.get("s_vec_residual")
                ts_tokens = analysis.get("ts_tokens")
                ts_mask = analysis.get("ts_mask")
                x_cf = analysis.get("x_cf")
                if s_vec_residual is not None and ts_tokens is not None and x_cf is not None:
                    ts_only_pred = model.predict_ts_only_full_decoder(
                        ts_tokens, ts_mask, s_vec_residual, x_cf
                    )
                    ts_only_loss = F.mse_loss(ts_only_pred.float(), target)
                    combined_loss = combined_loss + ts_only_lambda * ts_only_loss
                    loss_ts_only = float(ts_only_loss.detach().cpu()) * ts_only_lambda

            batch_smape = smape(output.float(), target.float())

            combined_loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=TrainingConfig.GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            ema_update()

            step_loss = float(combined_loss.detach().cpu())
            step_smape = float(batch_smape.detach().cpu())

            # ---- Training-set forecasting metrics (aligned with test()/vali) ----
            with torch.no_grad():
                _mse = nn.MSELoss()(output.float(), target.float())
                _mae = nn.L1Loss()(output.float(), target.float())
                batch_mse_metric = float(_mse.detach().cpu())
                batch_mae_metric = float(_mae.detach().cpu())

            epoch_loss_sum += step_loss
            epoch_smape_sum += step_smape
            epoch_mse_sum += batch_mse_metric
            epoch_mae_sum += batch_mae_metric
            epoch_contrastive_sum += float(contrastive_loss_raw)

            window["train_loss_sum"] += step_loss
            window["train_smape_sum"] += step_smape
            window["train_mse_sum"] += batch_mse_metric
            window["train_mae_sum"] += batch_mae_metric
            window["train_steps"] += 1
            window["core_reg_sum"] += loss_core_reg
            window["contrastive_sum"] += loss_contrastive
            window["gate_causal_sum"] += loss_gate_causal
            window["token_align_sum"] += loss_token_align
            window["ts_only_sum"] += loss_ts_only

            # Track alpha if using learnable last value
            if USE_LEARNABLE_LAST_VALUE and analysis is not None and "last_value_alpha" in analysis:
                alpha = analysis["last_value_alpha"]
                if alpha is not None:
                    window["alpha_sum"] += float(alpha.mean().detach().cpu())
                    window["alpha_count"] += 1

            if analysis is not None and ("gate_softmax" in analysis) and (analysis["gate_softmax"] is not None):
                g = analysis["gate_softmax"].detach().cpu()
                g_mean_per_branch = g.mean(dim=(0, 2))
                window["gate_sum"] += g_mean_per_branch
                window["gate_count"] += 1
                epoch_gate_sum += g_mean_per_branch
                epoch_gate_count += 1

            # ---- Subsampled training modality diagnostics (ablations) ----
            if (train_modality_every_steps is not None and train_modality_every_steps > 0 and
                    (step_in_epoch % train_modality_every_steps == 0 or step_in_epoch == len(train_loader)) and
                    analysis is not None):
                t_vec_residual = analysis.get("t_vec_residual")
                s_vec_residual = analysis.get("s_vec_residual")
                text_tokens = analysis.get("text_tokens")
                ts_tokens = analysis.get("ts_tokens")
                text_mask = analysis.get("text_mask")
                ts_mask = analysis.get("ts_mask")
                x_cf = analysis.get("x_cf")

                if (t_vec_residual is not None and s_vec_residual is not None and
                        text_tokens is not None and ts_tokens is not None and
                        text_mask is not None and ts_mask is not None and x_cf is not None):
                    with torch.no_grad():
                        ts_only_pred_eval = model.predict_ts_only_full_decoder(
                            ts_tokens, ts_mask, s_vec_residual, x_cf
                        )
                        text_only_pred_eval = model.predict_text_only_full_decoder(
                            text_tokens, text_mask, t_vec_residual, x_cf
                        )

                        ts_only_mse = nn.MSELoss()(ts_only_pred_eval.float(), target.float())
                        text_only_mse = nn.MSELoss()(text_only_pred_eval.float(), target.float())
                        ts_only_mae = nn.L1Loss()(ts_only_pred_eval.float(), target.float())
                        text_only_mae = nn.L1Loss()(text_only_pred_eval.float(), target.float())
                        ts_only_sm = smape(ts_only_pred_eval.float(), target.float())
                        text_only_sm = smape(text_only_pred_eval.float(), target.float())

                        err_full = ((output.float() - target.float()) ** 2).mean(dim=(1, 2))
                        err_ts = ((ts_only_pred_eval.float() - target.float()) ** 2).mean(dim=(1, 2))
                        err_text = ((text_only_pred_eval.float() - target.float()) ** 2).mean(dim=(1, 2))

                        err_ts_safe = torch.clamp(err_ts, min=1e-8)
                        err_text_safe = torch.clamp(err_text, min=1e-8)

                        pct_text_helps = ((err_ts - err_full) / err_ts_safe).mean()
                        pct_ts_helps = ((err_text - err_full) / err_text_safe).mean()

                        raw_text_helps = (err_ts - err_full).mean()
                        raw_ts_helps = (err_text - err_full).mean()

                    epoch_modality_batches += 1
                    epoch_ts_only_mse_sum += float(ts_only_mse.detach().cpu())
                    epoch_ts_only_mae_sum += float(ts_only_mae.detach().cpu())
                    epoch_text_only_mse_sum += float(text_only_mse.detach().cpu())
                    epoch_text_only_mae_sum += float(text_only_mae.detach().cpu())
                    epoch_ts_only_smape_sum += float(ts_only_sm.detach().cpu())
                    epoch_text_only_smape_sum += float(text_only_sm.detach().cpu())
                    epoch_granger_text_helps_pct_sum += float(pct_text_helps.detach().cpu())
                    epoch_granger_ts_helps_pct_sum += float(pct_ts_helps.detach().cpu())
                    epoch_granger_text_helps_raw_sum += float(raw_text_helps.detach().cpu())
                    epoch_granger_ts_helps_raw_sum += float(raw_ts_helps.detach().cpu())

            print(
                f'Epoch {epoch + 1}/{num_epochs} | step {step_in_epoch}/{len(train_loader)} | global {global_step} | loss {step_loss:.6f}')

            del output, batch_fusion_embeddings, batch_series_embeddings, batch_text_embeddings
            del batch_sent_report_embeddings, batch_negative_type_report_embeddings, analysis
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if eval_every_steps is not None and eval_every_steps > 0 and (global_step % eval_every_steps == 0):
                _do_eval_and_log(tag=f"Eval@{eval_every_steps}steps")
                window = _reset_window_accumulators()

        epoch_avg_loss = epoch_loss_sum / max(1, len(train_loader))
        epoch_train_smape = epoch_smape_sum / max(1, len(train_loader))
        epoch_train_mse = epoch_mse_sum / max(1, len(train_loader))
        epoch_train_mae = epoch_mae_sum / max(1, len(train_loader))
        epoch_train_rmse = math.sqrt(epoch_train_mse)
        epoch_train_contr = epoch_contrastive_sum / max(1, len(train_loader))

        epoch_end_msg = (
            f'[Epoch End] {epoch + 1}/{num_epochs} | '
            f'train loss: {epoch_avg_loss:.6f} | '
            f'MSE {epoch_train_mse:.6f} | RMSE {epoch_train_rmse:.6f} | MAE {epoch_train_mae:.6f} | '
            f'Contr {epoch_train_contr:.6f} | train SMAPE(%): {epoch_train_smape:.4f}'
        )
        print(epoch_end_msg)

        if epoch_gate_count > 0:
            epoch_gate_avg = (epoch_gate_sum / epoch_gate_count).tolist()
            gate_epoch_msg = "[Epoch End Gate (softmax) openness] " + " | ".join(
                f"{name}: {epoch_gate_avg[i]:.4f}" for i, name in enumerate(branch_names)
            )
            print(gate_epoch_msg)

        # ---- Train Modality Scores (subsampled) ----
        if epoch_modality_batches > 0:
            avg_ts_only_mse = epoch_ts_only_mse_sum / epoch_modality_batches
            avg_text_only_mse = epoch_text_only_mse_sum / epoch_modality_batches
            avg_ts_only_mae = epoch_ts_only_mae_sum / epoch_modality_batches
            avg_text_only_mae = epoch_text_only_mae_sum / epoch_modality_batches
            avg_ts_only_rmse = math.sqrt(avg_ts_only_mse)
            avg_text_only_rmse = math.sqrt(avg_text_only_mse)
            avg_ts_only_sm = epoch_ts_only_smape_sum / epoch_modality_batches
            avg_text_only_sm = epoch_text_only_smape_sum / epoch_modality_batches

            train_text_pct = (epoch_granger_text_helps_pct_sum / epoch_modality_batches) * 100
            train_ts_pct = (epoch_granger_ts_helps_pct_sum / epoch_modality_batches) * 100
            train_text_raw = (epoch_granger_text_helps_raw_sum / epoch_modality_batches)
            train_ts_raw = (epoch_granger_ts_helps_raw_sum / epoch_modality_batches)

            train_modality_msg = (
                f'[Epoch End] Train Modality Scores (subsampled every {train_modality_every_steps} steps, n={epoch_modality_batches}):\n'
                f'  - Text helps (vs TS-only): {train_text_pct:+.2f}% error reduction, raw: {train_text_raw:.6f}\n'
                f'  - TS helps (vs Text-only): {train_ts_pct:+.2f}% error reduction, raw: {train_ts_raw:.6f}\n'
                f'  - Full (Text+TS): MSE {epoch_train_mse:.6f}, RMSE {epoch_train_rmse:.6f}, MAE {epoch_train_mae:.6f}, SMAPE(%) {epoch_train_smape:.4f}\n'
                f'  - TS-only:   MSE {avg_ts_only_mse:.6f}, RMSE {avg_ts_only_rmse:.6f}, MAE {avg_ts_only_mae:.6f}, SMAPE(%) {avg_ts_only_sm:.4f}\n'
                f'  - Text-only: MSE {avg_text_only_mse:.6f}, RMSE {avg_text_only_rmse:.6f}, MAE {avg_text_only_mae:.6f}, SMAPE(%) {avg_text_only_sm:.4f}\n'
            )
            print(train_modality_msg)

        if (eval_every_steps is None) or (eval_every_steps <= 0) or (global_step % eval_every_steps != 0):
            _do_eval_and_log(tag="Eval@epoch_end")
            window = _reset_window_accumulators()

        (vali_combine_loss, vali_mse_loss, vali_mae_loss, vali_contras_loss, vali_rmse_loss,
         vali_smape_loss) = last_vali_metrics
        (test_combine_loss, test_mse_loss, test_mae_loss, test_contras_loss, test_rmse_loss,
         test_smape_loss) = last_test_metrics

        model_output_path = output_path or "./model_output"
        Path(model_output_path).mkdir(parents=True, exist_ok=True)

        lastest_loss = float(vali_mse_loss)
        if lastest_loss < best_loss:
            best_loss = lastest_loss
            _backup = swap_to_ema()
            save_path = os.path.join(model_output_path, "best_model.pth")
            model.save_model_combined(save_path=save_path)
            swap_from_backup(_backup)
            print(
                f'[Save Best] epoch {epoch + 1} | '
                f'Vali MSE {vali_mse_loss:.6f} | Test MSE {test_mse_loss:.6f}'
            )

        save_path = os.path.join(model_output_path, "lastest_model.pth")
        model.save_model_combined(save_path=save_path)
@torch.no_grad()
def test(model, test_loader, return_analysis: bool = False,
         use_horizon_weight=True, horizon_decay=0.06, contrastive_lambda=0.2,
         compute_modality_scores: bool = True,
         eval_horizons=(35, 70)):
    """Evaluate the model.

    Notes:
      - The *main* metrics (Comb/MSE/MAE/Contr/RMSE/SMAPE + modality/ablation stats)
        are computed on the full horizon of length ``model.pred_len`` and are kept
        identical to the original implementation.
      - If ``eval_horizons`` is provided and contains values < ``model.pred_len``,
        the function additionally computes *prefix-horizon* metrics (e.g., 35/70)
        by slicing the prediction/target (and ablated predictions) on-the-fly.
        These extra results are returned inside modality_metrics['horizon_results']
        without changing the existing return tuple structure.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    criterion_huber = nn.SmoothL1Loss(beta=0.5)

    # -------- main-horizon accumulators (UNCHANGED semantics) --------
    combined_loss_output = 0.0
    mse_loss_output = 0.0
    mae_loss_output = 0.0
    contrastive_loss_output = 0.0
    smape_loss_output = 0.0

    ts_only_mse_output = 0.0
    ts_only_mae_output = 0.0
    ts_only_smape_output = 0.0

    text_only_mse_output = 0.0
    text_only_mae_output = 0.0
    text_only_smape_output = 0.0

    granger_scores_sum = 0.0
    granger_text_scores_sum = 0.0
    granger_count = 0

    granger_text_helps_pct_sum = 0.0
    granger_ts_helps_pct_sum = 0.0

    granger_text_helps_raw_sum = 0.0
    granger_ts_helps_raw_sum = 0.0

    gate_text_sum = 0.0
    gate_ts_sum = 0.0
    gate_count = 0

    analyses = []
    h_w = make_horizon_weights(model.pred_len, device, decay=horizon_decay) if use_horizon_weight else None

    # -------- extra horizons (prefix evaluation) --------
    extra_horizons = []
    if eval_horizons is not None:
        try:
            extra_horizons = sorted({int(h) for h in eval_horizons if int(h) > 0 and int(h) < int(model.pred_len)})
        except Exception:
            extra_horizons = []

    # Accumulators for extra horizons. We keep *the same metrics* as the main horizon.
    h_acc = {}
    h_w_map = {}
    for hh in extra_horizons:
        h_acc[hh] = {
            'combined': 0.0,
            'mse': 0.0,
            'mae': 0.0,
            'smape': 0.0,
            # ablations
            'ts_only_mse': 0.0,
            'ts_only_mae': 0.0,
            'ts_only_smape': 0.0,
            'text_only_mse': 0.0,
            'text_only_mae': 0.0,
            'text_only_smape': 0.0,
            # modality scores
            'granger_scores_sum': 0.0,
            'granger_text_scores_sum': 0.0,
            'granger_text_helps_pct_sum': 0.0,
            'granger_ts_helps_pct_sum': 0.0,
            'granger_text_helps_raw_sum': 0.0,
            'granger_ts_helps_raw_sum': 0.0,
            'granger_count': 0,
        }
        h_w_map[hh] = make_horizon_weights(hh, device, decay=horizon_decay) if use_horizon_weight else None

    for batch_idx, (batch_text, batch_sent_reports, batch_negative_type_reprots, batch_seq, batch_pred,
                    batch_seq_scale, batch_pred_scale) in enumerate(
        tqdm(test_loader, desc="Testing", ncols=100, dynamic_ncols=True)):
        batch_sent_reports = list(map(list, zip(*batch_sent_reports)))
        batch_negative_type_reprots = list(map(list, zip(*batch_negative_type_reprots)))

        need_analysis = return_analysis or compute_modality_scores
        (output,
         batch_fusion_embeddings,
         batch_series_embeddings,
         batch_text_embeddings,
         batch_sent_report_embeddings,
         batch_negative_type_report_embeddings,
         analysis) = model.predict_batch_contrastive(
            batch_text, batch_sent_reports, batch_negative_type_reprots, batch_seq_scale,
            return_analysis=need_analysis
        )
        if return_analysis:
            analyses.append(analysis)

        target = batch_pred_scale.to(device)

        # ---------------- main horizon loss (UNCHANGED) ----------------
        if h_w is not None:
            mse_w, mae_w = weighted_mse_mae(output, target, h_w)
            core_reg = 0.7 * mse_w + 0.3 * mae_w
        else:
            mse_loss = criterion_mse(output, target)
            mae_loss = criterion_mae(output, target)
            huber = criterion_huber(output, target)
            core_reg = 0.5 * mse_loss + 0.3 * mae_loss + 0.2 * huber

        ts_anchor = analysis["s_vec_residual"] if (
                analysis is not None and "s_vec_residual" in analysis) else batch_series_embeddings
        contrastive_loss = contrastive_loss_objective(
            ts_anchor, batch_sent_report_embeddings,
            batch_negative_type_report_embeddings, w_ce=0.0, w_triplet=1.0
        )

        combined_loss = core_reg + contrastive_lambda * contrastive_loss

        combined_loss_output += float(combined_loss.detach().cpu())
        mse_loss_output += float(nn.MSELoss()(output, target).detach().cpu())
        mae_loss_output += float(nn.L1Loss()(output, target).detach().cpu())
        contrastive_loss_output += float(contrastive_loss.detach().cpu())
        smape_loss_output += float(smape(output.float(), target.float()).detach().cpu())

        # ---------------- extra horizons: overall metrics ----------------
        for hh in extra_horizons:
            out_h = output[..., :hh]
            tgt_h = target[..., :hh]
            if h_w_map[hh] is not None:
                mse_w_h, mae_w_h = weighted_mse_mae(out_h, tgt_h, h_w_map[hh])
                core_reg_h = 0.7 * mse_w_h + 0.3 * mae_w_h
            else:
                mse_h = criterion_mse(out_h, tgt_h)
                mae_h = criterion_mae(out_h, tgt_h)
                huber_h = criterion_huber(out_h, tgt_h)
                core_reg_h = 0.5 * mse_h + 0.3 * mae_h + 0.2 * huber_h

            comb_h = core_reg_h + contrastive_lambda * contrastive_loss
            h_acc[hh]['combined'] += float(comb_h.detach().cpu())
            h_acc[hh]['mse'] += float(nn.MSELoss()(out_h, tgt_h).detach().cpu())
            h_acc[hh]['mae'] += float(nn.L1Loss()(out_h, tgt_h).detach().cpu())
            h_acc[hh]['smape'] += float(smape(out_h.float(), tgt_h.float()).detach().cpu())

        # ---------------- ablations / modality scores (main + extra horizons) ----------------
        if compute_modality_scores and analysis is not None:
            t_vec_residual = analysis.get("t_vec_residual")
            s_vec_residual = analysis.get("s_vec_residual")
            text_tokens = analysis.get("text_tokens")
            ts_tokens = analysis.get("ts_tokens")
            text_mask = analysis.get("text_mask")
            ts_mask = analysis.get("ts_mask")
            x_cf = analysis.get("x_cf")

            if (t_vec_residual is not None and s_vec_residual is not None and
                    text_tokens is not None and ts_tokens is not None and x_cf is not None):

                ts_only_pred = model.predict_ts_only_full_decoder(
                    ts_tokens, ts_mask, s_vec_residual, x_cf
                )
                ts_only_mse = nn.MSELoss()(ts_only_pred, target)
                ts_only_mae = nn.L1Loss()(ts_only_pred, target)
                ts_only_mse_output += float(ts_only_mse.detach().cpu())
                ts_only_mae_output += float(ts_only_mae.detach().cpu())
                ts_only_smape_output += float(smape(ts_only_pred.float(), target.float()).detach().cpu())

                text_only_pred = model.predict_text_only_full_decoder(
                    text_tokens, text_mask, t_vec_residual, x_cf
                )
                text_only_mse = nn.MSELoss()(text_only_pred, target)
                text_only_mae = nn.L1Loss()(text_only_pred, target)
                text_only_mse_output += float(text_only_mse.detach().cpu())
                text_only_mae_output += float(text_only_mae.detach().cpu())
                text_only_smape_output += float(smape(text_only_pred.float(), target.float()).detach().cpu())

                # ---- main horizon modality scores (UNCHANGED) ----
                err_full = ((output - target) ** 2).mean(dim=(1, 2))
                err_ts = ((ts_only_pred - target) ** 2).mean(dim=(1, 2))
                err_text = ((text_only_pred - target) ** 2).mean(dim=(1, 2))

                diff_ts = (err_ts - err_full)
                diff_text = (err_text - err_full)

                if TrainingConfig.GC_USE_ADAPTIVE_SCALE:
                    s_ts = torch.clamp(diff_ts.detach().abs().mean(), min=TrainingConfig.GC_MIN_SCALE)
                    s_text = torch.clamp(diff_text.detach().abs().mean(), min=TrainingConfig.GC_MIN_SCALE)
                else:
                    s_ts = torch.tensor(TrainingConfig.GC_SIGMOID_SCALE, device=diff_ts.device, dtype=diff_ts.dtype)
                    s_text = torch.tensor(TrainingConfig.GC_SIGMOID_SCALE, device=diff_text.device,
                                          dtype=diff_text.dtype)

                logit_ts = torch.clamp(TrainingConfig.GC_GAIN * diff_ts / s_ts, -TrainingConfig.GC_LOGIT_CLAMP,
                                       TrainingConfig.GC_LOGIT_CLAMP)
                logit_text = torch.clamp(TrainingConfig.GC_GAIN * diff_text / s_text, -TrainingConfig.GC_LOGIT_CLAMP,
                                         TrainingConfig.GC_LOGIT_CLAMP)

                gc_score = torch.sigmoid(logit_ts)
                gc_text_score = torch.sigmoid(logit_text)

                granger_scores_sum += float(gc_score.mean().detach().cpu())
                granger_text_scores_sum += float(gc_text_score.mean().detach().cpu())

                err_ts_safe = torch.clamp(err_ts, min=1e-8)
                err_text_safe = torch.clamp(err_text, min=1e-8)

                pct_text_helps = ((err_ts - err_full) / err_ts_safe).mean()
                pct_ts_helps = ((err_text - err_full) / err_text_safe).mean()

                granger_text_helps_pct_sum += float(pct_text_helps.detach().cpu())
                granger_ts_helps_pct_sum += float(pct_ts_helps.detach().cpu())

                raw_text_helps = (err_ts - err_full).mean()
                raw_ts_helps = (err_text - err_full).mean()
                granger_text_helps_raw_sum += float(raw_text_helps.detach().cpu())
                granger_ts_helps_raw_sum += float(raw_ts_helps.detach().cpu())

                granger_count += 1

                # ---- extra horizons: ablations + modality scores ----
                for hh in extra_horizons:
                    out_h = output[..., :hh]
                    tgt_h = target[..., :hh]
                    ts_h = ts_only_pred[..., :hh]
                    txt_h = text_only_pred[..., :hh]

                    h_acc[hh]['ts_only_mse'] += float(nn.MSELoss()(ts_h, tgt_h).detach().cpu())
                    h_acc[hh]['ts_only_mae'] += float(nn.L1Loss()(ts_h, tgt_h).detach().cpu())
                    h_acc[hh]['ts_only_smape'] += float(smape(ts_h.float(), tgt_h.float()).detach().cpu())

                    h_acc[hh]['text_only_mse'] += float(nn.MSELoss()(txt_h, tgt_h).detach().cpu())
                    h_acc[hh]['text_only_mae'] += float(nn.L1Loss()(txt_h, tgt_h).detach().cpu())
                    h_acc[hh]['text_only_smape'] += float(smape(txt_h.float(), tgt_h.float()).detach().cpu())

                    err_full_h = ((out_h - tgt_h) ** 2).mean(dim=(1, 2))
                    err_ts_h = ((ts_h - tgt_h) ** 2).mean(dim=(1, 2))
                    err_text_h = ((txt_h - tgt_h) ** 2).mean(dim=(1, 2))

                    diff_ts_h = (err_ts_h - err_full_h)
                    diff_text_h = (err_text_h - err_full_h)

                    if TrainingConfig.GC_USE_ADAPTIVE_SCALE:
                        s_ts_h = torch.clamp(diff_ts_h.detach().abs().mean(), min=TrainingConfig.GC_MIN_SCALE)
                        s_text_h = torch.clamp(diff_text_h.detach().abs().mean(), min=TrainingConfig.GC_MIN_SCALE)
                    else:
                        s_ts_h = torch.tensor(TrainingConfig.GC_SIGMOID_SCALE, device=diff_ts_h.device,
                                              dtype=diff_ts_h.dtype)
                        s_text_h = torch.tensor(TrainingConfig.GC_SIGMOID_SCALE, device=diff_text_h.device,
                                                dtype=diff_text_h.dtype)

                    logit_ts_h = torch.clamp(TrainingConfig.GC_GAIN * diff_ts_h / s_ts_h,
                                             -TrainingConfig.GC_LOGIT_CLAMP, TrainingConfig.GC_LOGIT_CLAMP)
                    logit_text_h = torch.clamp(TrainingConfig.GC_GAIN * diff_text_h / s_text_h,
                                               -TrainingConfig.GC_LOGIT_CLAMP, TrainingConfig.GC_LOGIT_CLAMP)

                    gc_score_h = torch.sigmoid(logit_ts_h)
                    gc_text_score_h = torch.sigmoid(logit_text_h)

                    h_acc[hh]['granger_scores_sum'] += float(gc_score_h.mean().detach().cpu())
                    h_acc[hh]['granger_text_scores_sum'] += float(gc_text_score_h.mean().detach().cpu())

                    err_ts_safe_h = torch.clamp(err_ts_h, min=1e-8)
                    err_text_safe_h = torch.clamp(err_text_h, min=1e-8)

                    pct_text_helps_h = ((err_ts_h - err_full_h) / err_ts_safe_h).mean()
                    pct_ts_helps_h = ((err_text_h - err_full_h) / err_text_safe_h).mean()

                    h_acc[hh]['granger_text_helps_pct_sum'] += float(pct_text_helps_h.detach().cpu())
                    h_acc[hh]['granger_ts_helps_pct_sum'] += float(pct_ts_helps_h.detach().cpu())

                    raw_text_helps_h = (err_ts_h - err_full_h).mean()
                    raw_ts_helps_h = (err_text_h - err_full_h).mean()
                    h_acc[hh]['granger_text_helps_raw_sum'] += float(raw_text_helps_h.detach().cpu())
                    h_acc[hh]['granger_ts_helps_raw_sum'] += float(raw_ts_helps_h.detach().cpu())

                    h_acc[hh]['granger_count'] += 1

        # Gate openness (same for all horizons)
        if analysis is not None and "gate_softmax" in analysis and analysis["gate_softmax"] is not None:
            g = analysis["gate_softmax"].detach()
            g_mean_per_branch = g.mean(dim=(0, 2))
            gate_text_sum += float(g_mean_per_branch[0].cpu())
            gate_ts_sum += float(g_mean_per_branch[1].cpu())
            gate_count += 1

    n_batches = max(1, len(test_loader))

    # -------- main horizon averages (UNCHANGED) --------
    avg_combined_loss = combined_loss_output / n_batches
    avg_mse_loss = mse_loss_output / n_batches
    avg_mae_loss = mae_loss_output / n_batches
    avg_contrastive_loss = contrastive_loss_output / n_batches
    avg_rmse_loss = math.sqrt(avg_mse_loss)
    avg_smape_loss = smape_loss_output / n_batches

    modality_metrics = None
    if compute_modality_scores:
        avg_ts_only_mse = ts_only_mse_output / n_batches
        avg_ts_only_mae = ts_only_mae_output / n_batches
        avg_ts_only_rmse = math.sqrt(avg_ts_only_mse)
        avg_ts_only_smape = ts_only_smape_output / n_batches

        avg_text_only_mse = text_only_mse_output / n_batches
        avg_text_only_mae = text_only_mae_output / n_batches
        avg_text_only_rmse = math.sqrt(avg_text_only_mse)
        avg_text_only_smape = text_only_smape_output / n_batches

        avg_granger_score = granger_scores_sum / max(1, granger_count)
        avg_granger_text_score = granger_text_scores_sum / max(1, granger_count)

        avg_granger_text_helps_pct = granger_text_helps_pct_sum / max(1, granger_count)
        avg_granger_ts_helps_pct = granger_ts_helps_pct_sum / max(1, granger_count)

        avg_granger_text_helps_raw = granger_text_helps_raw_sum / max(1, granger_count)
        avg_granger_ts_helps_raw = granger_ts_helps_raw_sum / max(1, granger_count)

        avg_gate_text = gate_text_sum / max(1, gate_count)
        avg_gate_ts = gate_ts_sum / max(1, gate_count)

        modality_metrics = {
            'ts_only_mse': avg_ts_only_mse,
            'ts_only_mae': avg_ts_only_mae,
            'ts_only_rmse': avg_ts_only_rmse,
            'ts_only_smape': avg_ts_only_smape,
            'text_only_mse': avg_text_only_mse,
            'text_only_mae': avg_text_only_mae,
            'text_only_rmse': avg_text_only_rmse,
            'text_only_smape': avg_text_only_smape,
            'granger_text_helps': avg_granger_score,
            'granger_ts_helps': avg_granger_text_score,
            'granger_text_helps_pct': avg_granger_text_helps_pct,
            'granger_ts_helps_pct': avg_granger_ts_helps_pct,
            'granger_text_helps_raw': avg_granger_text_helps_raw,
            'granger_ts_helps_raw': avg_granger_ts_helps_raw,
            'gate_text': avg_gate_text,
            'gate_ts': avg_gate_ts,
        }

        # -------- extra horizon summary (added; does NOT change existing keys/returns) --------
        if extra_horizons:
            horizon_results = {}
            for hh in extra_horizons:
                avg_comb_h = h_acc[hh]['combined'] / n_batches
                avg_mse_h = h_acc[hh]['mse'] / n_batches
                avg_mae_h = h_acc[hh]['mae'] / n_batches
                avg_rmse_h = math.sqrt(avg_mse_h)
                avg_smape_h = h_acc[hh]['smape'] / n_batches

                # contrastive loss is identical across horizons (embedding-level)
                avg_contr_h = avg_contrastive_loss

                avg_ts_only_mse_h = h_acc[hh]['ts_only_mse'] / n_batches
                avg_ts_only_mae_h = h_acc[hh]['ts_only_mae'] / n_batches
                avg_ts_only_rmse_h = math.sqrt(avg_ts_only_mse_h)
                avg_ts_only_smape_h = h_acc[hh]['ts_only_smape'] / n_batches

                avg_text_only_mse_h = h_acc[hh]['text_only_mse'] / n_batches
                avg_text_only_mae_h = h_acc[hh]['text_only_mae'] / n_batches
                avg_text_only_rmse_h = math.sqrt(avg_text_only_mse_h)
                avg_text_only_smape_h = h_acc[hh]['text_only_smape'] / n_batches

                gcount_h = max(1, int(h_acc[hh]['granger_count']))
                avg_granger_score_h = h_acc[hh]['granger_scores_sum'] / gcount_h
                avg_granger_text_score_h = h_acc[hh]['granger_text_scores_sum'] / gcount_h

                avg_granger_text_helps_pct_h = h_acc[hh]['granger_text_helps_pct_sum'] / gcount_h
                avg_granger_ts_helps_pct_h = h_acc[hh]['granger_ts_helps_pct_sum'] / gcount_h

                avg_granger_text_helps_raw_h = h_acc[hh]['granger_text_helps_raw_sum'] / gcount_h
                avg_granger_ts_helps_raw_h = h_acc[hh]['granger_ts_helps_raw_sum'] / gcount_h

                horizon_results[int(hh)] = {
                    'combined_loss': avg_comb_h,
                    'mse': avg_mse_h,
                    'rmse': avg_rmse_h,
                    'mae': avg_mae_h,
                    'contrastive': avg_contr_h,
                    'smape': avg_smape_h,
                    'ts_only_mse': avg_ts_only_mse_h,
                    'ts_only_rmse': avg_ts_only_rmse_h,
                    'ts_only_mae': avg_ts_only_mae_h,
                    'ts_only_smape': avg_ts_only_smape_h,
                    'text_only_mse': avg_text_only_mse_h,
                    'text_only_rmse': avg_text_only_rmse_h,
                    'text_only_mae': avg_text_only_mae_h,
                    'text_only_smape': avg_text_only_smape_h,
                    'granger_text_helps': avg_granger_score_h,
                    'granger_ts_helps': avg_granger_text_score_h,
                    'granger_text_helps_pct': avg_granger_text_helps_pct_h,
                    'granger_ts_helps_pct': avg_granger_ts_helps_pct_h,
                    'granger_text_helps_raw': avg_granger_text_helps_raw_h,
                    'granger_ts_helps_raw': avg_granger_ts_helps_raw_h,
                    'gate_text': avg_gate_text,
                    'gate_ts': avg_gate_ts,
                }

            modality_metrics['horizon_results'] = horizon_results

    if return_analysis:
        if compute_modality_scores:
            return (avg_combined_loss, avg_mse_loss, avg_mae_loss,
                    avg_contrastive_loss, avg_rmse_loss, avg_smape_loss, analyses, modality_metrics)
        else:
            return (avg_combined_loss, avg_mse_loss, avg_mae_loss,
                    avg_contrastive_loss, avg_rmse_loss, avg_smape_loss, analyses)
    else:
        if compute_modality_scores:
            return (avg_combined_loss, avg_mse_loss, avg_mae_loss,
                    avg_contrastive_loss, avg_rmse_loss, avg_smape_loss, modality_metrics)
        else:
            return (avg_combined_loss, avg_mse_loss, avg_mae_loss,
                    avg_contrastive_loss, avg_rmse_loss, avg_smape_loss)
@torch.no_grad()
def test_save(model, test_loader, output_folder=None,
              use_horizon_weight=True, horizon_decay=0.06, contrastive_lambda=0.2):
    if output_folder is None:
        output_folder = "multi-model-forecast/test_samples_analysis/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    combined_loss_output = 0.0
    mse_loss_output = 0.0
    mae_loss_output = 0.0
    contrastive_loss_output = 0.0
    smape_loss_output = 0.0

    os.makedirs(output_folder, exist_ok=True)
    h_w = make_horizon_weights(model.pred_len, device, decay=horizon_decay) if use_horizon_weight else None

    for batch_idx, (
            batch_text, batch_sent_reports, batch_negative_type_reprots, batch_seq, batch_pred, batch_seq_scale,
            batch_pred_scale) in enumerate(tqdm(test_loader, desc="Testing", ncols=100, dynamic_ncols=True)):
        batch_sent_reports = list(map(list, zip(*batch_sent_reports)))
        batch_negative_type_reprots = list(map(list, zip(*batch_negative_type_reprots)))

        (output,
         batch_fusion_embeddings,
         batch_series_embeddings,
         batch_text_embeddings,
         batch_sent_report_embeddings,
         batch_negative_type_report_embeddings,
         analysis) = model.predict_batch_contrastive(
            batch_text, batch_sent_reports, batch_negative_type_reprots, batch_seq_scale,
            return_analysis=True)

        target = batch_pred_scale.to(device)
        if h_w is not None:
            mse_w, mae_w = weighted_mse_mae(output, target, h_w)
            core_reg = 0.7 * mse_w + 0.3 * mae_w
        else:
            mse_loss = nn.MSELoss()(output, target)
            mae_loss = nn.L1Loss()(output, target)
            huber = nn.SmoothL1Loss(beta=0.5)(output, target)
            core_reg = 0.5 * mse_loss + 0.3 * mae_loss + 0.2 * huber

        ts_anchor = analysis["s_vec_residual"] if (
                analysis is not None and "s_vec_residual" in analysis) else batch_series_embeddings
        contrastive_loss = contrastive_loss_objective(
            ts_anchor, batch_sent_report_embeddings,
            batch_negative_type_report_embeddings, w_ce=0.0, w_triplet=1.0
        )

        combined_loss = core_reg + contrastive_lambda * contrastive_loss

        combined_loss_output += float(combined_loss.detach().cpu())
        mse_loss_output += float(nn.MSELoss()(output, target).detach().cpu())
        mae_loss_output += float(nn.L1Loss()(output, target).detach().cpu())
        contrastive_loss_output += float(contrastive_loss.detach().cpu())
        smape_loss_output += float(smape(output.float(), target.float()).detach().cpu())

        print(f"Testing completed. Results saved in {output_folder}")

    avg_combined_loss = combined_loss_output / max(1, len(test_loader))
    avg_mse_loss = mse_loss_output / max(1, len(test_loader))
    avg_mae_loss = mae_loss_output / max(1, len(test_loader))
    avg_contrastive_loss = contrastive_loss_output / max(1, len(test_loader))
    avg_rmse_loss = math.sqrt(avg_mse_loss)
    avg_smape_loss = smape_loss_output / max(1, len(test_loader))

    return avg_combined_loss, avg_mse_loss, avg_mae_loss, avg_contrastive_loss, avg_rmse_loss, avg_smape_loss


@torch.no_grad()
def real_test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    combined_loss_output = 0.0
    mse_loss_output = 0.0
    mae_loss_output = 0.0
    contrastive_loss_output = 0.0
    smape_loss_output = 0.0

    for batch_idx, (batch_text, batch_seq, batch_pred, batch_seq_scale, batch_pred_scale) in enumerate(
            tqdm(test_loader, desc="Testing", ncols=100, dynamic_ncols=True)):
        output, batch_fusion_embeddings, batch_series_embeddings, batch_text_embeddings = model.predict_single_case(
            batch_text, batch_seq_scale)

        mse_loss = nn.MSELoss()(batch_pred_scale.to(device), output)
        mae_loss = nn.L1Loss()(batch_pred_scale.to(device), output)
        combined_loss = mse_loss + mae_loss

        combined_loss_output += float(combined_loss.detach().cpu())
        mse_loss_output += float(mse_loss.detach().cpu())
        mae_loss_output += float(mae_loss.detach().cpu())
        smape_loss_output += float(smape(output.float(), batch_pred_scale.float().to(device)).detach().cpu())

    avg_combined_loss = combined_loss_output / max(1, len(test_loader))
    avg_mse_loss = mse_loss_output / max(1, len(test_loader))
    avg_mae_loss = mae_loss_output / max(1, len(test_loader))
    avg_contrastive_loss = 0.0
    avg_rmse_loss = math.sqrt(avg_mse_loss)
    avg_smape_loss = smape_loss_output / max(1, len(test_loader))

    print(f"Hit Rate for Upward Movements: {0.0:.2f}%")
    print(f"Hit Rate for Downward Movements: {0.0:.2f}%")
    print(f"Total Hit Rate: {0.0:.2f}%")
    print(f"Cumulative Profit: {0.0:.2f}")

    return avg_combined_loss, avg_mse_loss, avg_mae_loss, avg_contrastive_loss, avg_rmse_loss, avg_smape_loss


# ============================= Three-Stage Training ==================================


# ============================= Three-Stage Training ==================================
# Note: Sliding window dataset is imported from sliding_window_dataloader.py

def train_stage1_ts_only(model, sw_train_loader, sw_vali_loader, num_epochs=None, output_path=None):
    """
    Stage 1: Time Series Pre-training (Sliding Window Fashion)

    Fine-tune the decoder on the time series modality only, using sliding windows.
    This stage trains the decoder to understand time series patterns without textual information.

    IMPORTANT: This function expects sliding window data loaders from sliding_window_dataloader.py,
    NOT the event-based loaders. The sliding window loaders return 4 items per batch:
    (batch_seq, batch_pred, batch_seq_scale, batch_pred_scale)

    Components trained:
    - MOMENT last layer only (not full model)
    - Decoder (TinyDecoderV3)
    - TS feature projections (ts_feat_to_fuse, ts_inst_to_fuse)
    - Residual projection
    - Multi-scale extractor (if enabled)
    - Last value module (if enabled)

    Args:
        model: The CAMEF model
        sw_train_loader: Sliding window training data loader (from sliding_window_dataloader.py)
        sw_vali_loader: Sliding window validation data loader (from sliding_window_dataloader.py)
        num_epochs: Number of epochs for Stage 1
        output_path: Path to save model checkpoints
    """
    if num_epochs is None:
        num_epochs = TrainingConfig.STAGE1_EPOCHS

    print("=" * 80)
    print("STAGE 1: Time Series Pre-training (Sliding Window Fashion)")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning Rate: {TrainingConfig.STAGE1_LR}")
    print(f"  MOMENT: Last layer fine-tuning only")
    print(f"  Data: Sliding window loaders (continuous time series)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    # Freeze all parameters first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze only TS-related components and decoder
    for p in model.decoder.parameters():
        p.requires_grad = True
    for p in model.ts_feat_to_fuse.parameters():
        p.requires_grad = True
    for p in model.ts_inst_to_fuse.parameters():
        p.requires_grad = True
    for p in model.residual_project.parameters():
        p.requires_grad = True
    for p in model.ts_tokenizer.parameters():
        p.requires_grad = True

    if USE_MULTI_SCALE_TS:
        for p in model.multi_scale_extractor.parameters():
            p.requires_grad = True
    if USE_LEARNABLE_LAST_VALUE:
        for p in model.last_value_module.parameters():
            p.requires_grad = True

    # Unfreeze MOMENT last block only for fine-tuning
    # if model.moment_model is not None:
    #     for p in model.moment_last_block_parameters():
    #         p.requires_grad = False
    #     print(f"[Stage 1] MOMENT: Last layer frozen for fine-tuning")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Stage 1] Trainable parameters: {trainable_params:,}")

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    # Collect trainable parameters
    trainable_groups = [
        {'params': model.decoder.parameters(), 'lr': TrainingConfig.STAGE1_LR, 'weight_decay': 0.01},
        {'params': model.ts_feat_to_fuse.parameters(), 'lr': TrainingConfig.STAGE1_LR, 'weight_decay': 0.01},
        {'params': model.ts_inst_to_fuse.parameters(), 'lr': TrainingConfig.STAGE1_LR, 'weight_decay': 0.01},
        {'params': model.residual_project.parameters(), 'lr': TrainingConfig.STAGE1_LR, 'weight_decay': 0.01},
        {'params': model.ts_tokenizer.parameters(), 'lr': TrainingConfig.STAGE1_LR, 'weight_decay': 0.01},
    ]

    if USE_MULTI_SCALE_TS:
        trainable_groups.append(
            {'params': model.multi_scale_extractor.parameters(), 'lr': TrainingConfig.STAGE1_LR, 'weight_decay': 0.01}
        )
    if USE_LEARNABLE_LAST_VALUE:
        trainable_groups.append(
            {'params': model.last_value_module.parameters(), 'lr': TrainingConfig.STAGE1_LR * 2, 'weight_decay': 0.01}
        )

    # Add MOMENT last block parameters with slightly lower LR
    # if model.moment_model is not None:
    #     trainable_groups.append({
    #         'params': model.moment_last_block_parameters(),
    #         'lr': TrainingConfig.STAGE1_LR * 0.1,  # Lower LR for foundation model
    #         'weight_decay': 0.01
    #     })

    optimizer = optim.AdamW(trainable_groups, betas=(0.9, 0.999), eps=1e-8)
    num_training_steps = num_epochs * len(sw_train_loader)
    num_warmup_steps = int(TrainingConfig.WARMUP_RATIO * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    best_loss = float('inf')
    log_file = ""

    for epoch in range(num_epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_smape_sum = 0.0

        # Sliding window loader returns 4 items: (seq, pred, scaled_seq, scaled_pred)
        for step_in_epoch, (batch_seq, batch_pred, batch_seq_scale, batch_pred_scale) in enumerate(
                sw_train_loader, start=1):

            optimizer.zero_grad()

            batch_seq_scale = batch_seq_scale.float().to(device)
            x_cf = model._ensure_cf(batch_seq_scale)

            # Get TS tokens and instance embedding (TS modality only)
            ts_tok, ts_mask, ts_inst = model.get_ts_tokens_full(x_cf)

            # Use TS instance embedding as the fused embedding (no text)
            s_vec_residual = ts_inst
            s_vec_norm = F.normalize(s_vec_residual, p=2, dim=-1)

            # Decode using TS-only pathway
            core = model.decoder(s_vec_norm, x_cf=x_cf)

            if USE_LEARNABLE_LAST_VALUE:
                last_val = x_cf[:, :, -1:].detach()
                output, _ = model.last_value_module(core, last_val, s_vec_norm)
            else:
                output = core

            target = batch_pred_scale.float().to(device)

            # Simple regression loss
            mse_loss = criterion_mse(output, target)
            mae_loss = criterion_mae(output, target)
            combined_loss = 0.7 * mse_loss + 0.3 * mae_loss

            combined_loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=TrainingConfig.GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            step_loss = float(combined_loss.detach().cpu())
            step_smape = float(smape(output.float(), target.float()).detach().cpu())
            epoch_loss_sum += step_loss
            epoch_smape_sum += step_smape

            if step_in_epoch % 50 == 0:
                print(
                    f'[Stage 1] Epoch {epoch + 1}/{num_epochs} | Step {step_in_epoch}/{len(sw_train_loader)} | Loss: {step_loss:.6f}')

        epoch_avg_loss = epoch_loss_sum / max(1, len(sw_train_loader))
        epoch_avg_smape = epoch_smape_sum / max(1, len(sw_train_loader))

        # Validation
        model.eval()
        vali_loss = 0.0
        with torch.no_grad():
            for batch_seq, batch_pred, batch_seq_scale, batch_pred_scale in sw_vali_loader:

                batch_seq_scale = batch_seq_scale.float().to(device)
                x_cf = model._ensure_cf(batch_seq_scale)
                ts_tok, ts_mask, ts_inst = model.get_ts_tokens_full(x_cf)
                s_vec_norm = F.normalize(ts_inst, p=2, dim=-1)
                core = model.decoder(s_vec_norm, x_cf=x_cf)

                if USE_LEARNABLE_LAST_VALUE:
                    last_val = x_cf[:, :, -1:].detach()
                    output, _ = model.last_value_module(core, last_val, s_vec_norm)
                else:
                    output = core

                target = batch_pred_scale.float().to(device)
                vali_loss += float(nn.MSELoss()(output, target).detach().cpu())

        vali_loss /= max(1, len(sw_vali_loader))

        msg = f'[Stage 1] Epoch {epoch + 1}/{num_epochs} | Train Loss: {epoch_avg_loss:.6f} | Train SMAPE: {epoch_avg_smape:.4f} | Vali MSE: {vali_loss:.6f}'
        print(msg)
        log_file += msg + "\n"

        if vali_loss < best_loss:
            best_loss = vali_loss
            if output_path:
                save_path = os.path.join(output_path, "stage1_best_model.pth")
                Path(output_path).mkdir(parents=True, exist_ok=True)
                model.save_model_combined(save_path=save_path)
                print(f'[Stage 1] Saved best model with Vali MSE: {vali_loss:.6f}')

    print(f"[Stage 1] Completed. Best Vali MSE: {best_loss:.6f}")
    return log_file


def train_stage2_text_only(model, train_loader, vali_loader, num_epochs=None, output_path=None):
    """
    Stage 2: Text Pre-training

    Fine-tune the decoder on the textual modality only, omitting time series features.
    This stage trains the decoder to understand how textual information relates to predictions.

    Components trained:
    - Decoder (TinyDecoderV3)
    - LLM projection (llm_proj)
    - Text-only head
    - Last value module (if enabled)
    """
    if num_epochs is None:
        num_epochs = TrainingConfig.STAGE2_EPOCHS

    print("=" * 80)
    print("STAGE 2: Text Pre-training (Text modality only)")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning Rate: {TrainingConfig.STAGE2_LR}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    # Freeze all parameters first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze text-related components and decoder
    for p in model.decoder.parameters():
        p.requires_grad = True
    for p in model.llm_proj.parameters():
        p.requires_grad = True
    for p in model.text_only_head.parameters():
        p.requires_grad = True

    if USE_LEARNABLE_LAST_VALUE:
        for p in model.last_value_module.parameters():
            p.requires_grad = True

    # Optionally unfreeze LLM last block
    llm_last_block_params = []
    if FINETUNE_LLM_LAST_LAYER:
        llm_last_block_params = model.llm_last_block_parameters()
        for p in llm_last_block_params:
            p.requires_grad = False

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Stage 2] Trainable parameters: {trainable_params:,}")

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    # Collect trainable parameters
    trainable_groups = [
        {'params': model.decoder.parameters(), 'lr': TrainingConfig.STAGE2_LR, 'weight_decay': 0.01},
        {'params': model.llm_proj.parameters(), 'lr': TrainingConfig.STAGE2_LR, 'weight_decay': 0.01},
        {'params': model.text_only_head.parameters(), 'lr': TrainingConfig.STAGE2_LR, 'weight_decay': 0.01},
    ]

    if USE_LEARNABLE_LAST_VALUE:
        trainable_groups.append(
            {'params': model.last_value_module.parameters(), 'lr': TrainingConfig.STAGE2_LR * 2, 'weight_decay': 0.01}
        )

    if FINETUNE_LLM_LAST_LAYER and len(llm_last_block_params) > 0:
        trainable_groups.append({
            'params': llm_last_block_params,
            'lr': LLM_FINETUNE_LR,
            'weight_decay': 0.01
        })

    optimizer = optim.AdamW(trainable_groups, betas=(0.9, 0.999), eps=1e-8)
    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = int(TrainingConfig.WARMUP_RATIO * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    best_loss = float('inf')
    log_file = ""

    for epoch in range(num_epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_smape_sum = 0.0

        for step_in_epoch, (batch_text, batch_sent_reports, batch_negative_type_reports,
                            batch_seq, batch_pred, batch_seq_scale, batch_pred_scale) in enumerate(
            train_loader, start=1):

            optimizer.zero_grad()

            # Get text tokens
            tlist, mlist = [], []
            for t in batch_text:
                T, M = model.get_text_tokens_all(t)
                tlist.append(T)
                mlist.append(M)
            text_tok, text_mask = model._pad_and_stack(tlist, mlist)

            # Get text embedding (text modality only)
            t_vec = model._masked_mean(text_tok, text_mask)
            t_vec_norm = F.normalize(t_vec, p=2, dim=-1)

            # We still need x_cf for the decoder's temporal shortcut and last value module
            batch_seq_scale = batch_seq_scale.float().to(device)
            x_cf = model._ensure_cf(batch_seq_scale)

            # Decode using text-only pathway
            core = model.decoder(t_vec_norm, x_cf=x_cf)

            if USE_LEARNABLE_LAST_VALUE:
                last_val = x_cf[:, :, -1:].detach()
                output, _ = model.last_value_module(core, last_val, t_vec_norm)
            else:
                output = core

            target = batch_pred_scale.float().to(device)

            # Simple regression loss
            mse_loss = criterion_mse(output, target)
            mae_loss = criterion_mae(output, target)
            combined_loss = 0.7 * mse_loss + 0.3 * mae_loss

            combined_loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=TrainingConfig.GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            step_loss = float(combined_loss.detach().cpu())
            step_smape = float(smape(output.float(), target.float()).detach().cpu())
            epoch_loss_sum += step_loss
            epoch_smape_sum += step_smape

            if step_in_epoch % 50 == 0:
                print(
                    f'[Stage 2] Epoch {epoch + 1}/{num_epochs} | Step {step_in_epoch}/{len(train_loader)} | Loss: {step_loss:.6f}')

        epoch_avg_loss = epoch_loss_sum / max(1, len(train_loader))
        epoch_avg_smape = epoch_smape_sum / max(1, len(train_loader))

        # Validation
        model.eval()
        vali_loss = 0.0
        with torch.no_grad():
            for batch_text, batch_sent_reports, batch_negative_type_reports, \
                    batch_seq, batch_pred, batch_seq_scale, batch_pred_scale in vali_loader:

                tlist, mlist = [], []
                for t in batch_text:
                    T, M = model.get_text_tokens_all(t)
                    tlist.append(T)
                    mlist.append(M)
                text_tok, text_mask = model._pad_and_stack(tlist, mlist)

                t_vec = model._masked_mean(text_tok, text_mask)
                t_vec_norm = F.normalize(t_vec, p=2, dim=-1)

                batch_seq_scale = batch_seq_scale.float().to(device)
                x_cf = model._ensure_cf(batch_seq_scale)
                core = model.decoder(t_vec_norm, x_cf=x_cf)

                if USE_LEARNABLE_LAST_VALUE:
                    last_val = x_cf[:, :, -1:].detach()
                    output, _ = model.last_value_module(core, last_val, t_vec_norm)
                else:
                    output = core

                target = batch_pred_scale.float().to(device)
                vali_loss += float(nn.MSELoss()(output, target).detach().cpu())

        vali_loss /= max(1, len(vali_loader))

        msg = f'[Stage 2] Epoch {epoch + 1}/{num_epochs} | Train Loss: {epoch_avg_loss:.6f} | Train SMAPE: {epoch_avg_smape:.4f} | Vali MSE: {vali_loss:.6f}'
        print(msg)
        log_file += msg + "\n"

        if vali_loss < best_loss:
            best_loss = vali_loss
            if output_path:
                save_path = os.path.join(output_path, "stage2_best_model.pth")
                Path(output_path).mkdir(parents=True, exist_ok=True)
                model.save_model_combined(save_path=save_path)
                print(f'[Stage 2] Saved best model with Vali MSE: {vali_loss:.6f}')

    print(f"[Stage 2] Completed. Best Vali MSE: {best_loss:.6f}")
    return log_file


def train_three_stage(model, train_loader, test_loader, vali_loader, num_epochs,
                      sw_train_loader, sw_vali_loader,
                      output_path=None,
                      use_horizon_weight=True, horizon_decay=0.00,
                      contrastive_lambda=None,
                      gate_causal_lambda=None,
                      inv_lambda=None,
                      eval_every_steps: int = 100,
                      simplified_mode: bool = None,
                      stage1_epochs=None,
                      stage2_epochs=None):
    """
    Three-Stage Training Pipeline

    This function orchestrates the three-stage training process:
    1. Stage 1: Time Series Pre-training (TS modality only, sliding window)
    2. Stage 2: Text Pre-training (Text modality only)
    3. Stage 3: Full Multimodal Training (current design)

    Args:
        model: The CAMEF model
        train_loader: Event-based training data loader (for Stage 2 and 3)
        test_loader: Event-based test data loader
        vali_loader: Event-based validation data loader (for Stage 2 and 3)
        num_epochs: Number of epochs for Stage 3 (full multimodal training)
        sw_train_loader: Sliding window training loader (from sliding_window_dataloader.py) for Stage 1
        sw_vali_loader: Sliding window validation loader (from sliding_window_dataloader.py) for Stage 1
        output_path: Path to save model checkpoints
        use_horizon_weight: Whether to use horizon weighting
        horizon_decay: Decay factor for horizon weights
        contrastive_lambda: Weight for contrastive loss
        gate_causal_lambda: Weight for gate causal loss
        inv_lambda: Weight for invariance loss
        eval_every_steps: Evaluation frequency
        simplified_mode: Whether to use simplified training mode
        stage1_epochs: Override epochs for Stage 1 (default: TrainingConfig.STAGE1_EPOCHS)
        stage2_epochs: Override epochs for Stage 2 (default: TrainingConfig.STAGE2_EPOCHS)
    """
    # ------------------------------------------------------------------
    # Global logger: capture *everything* printed to stdout across Stage1/2/3
    # ------------------------------------------------------------------
    model_output_path = output_path or "./model_output"
    Path(model_output_path).mkdir(parents=True, exist_ok=True)
    log_file_path = os.path.join(model_output_path, "full_training_log.txt")

    logger = DualLogger(log_file_path)
    original_stdout = sys.stdout
    sys.stdout = logger

    try:
        print("=" * 80)
        print("THREE-STAGE TRAINING PIPELINE")
        print(f"Logging to: {log_file_path}")
        print("=" * 80)
        print(f"Stage 1 (TS Pre-training): {stage1_epochs or TrainingConfig.STAGE1_EPOCHS} epochs")
        print(f"  -> Using sliding window loaders (continuous time series)")
        print(f"Stage 2 (Text Pre-training): {stage2_epochs or TrainingConfig.STAGE2_EPOCHS} epochs")
        print(f"Stage 3 (Full Multimodal): {num_epochs} epochs")
        print("=" * 80)

        # ==================== Stage 1: TS Pre-training (Sliding Window) ====================
        train_stage1_ts_only(
            model, sw_train_loader, sw_vali_loader,
            num_epochs=stage1_epochs,
            output_path=output_path
        )
        print("\n=== STAGE 1 COMPLETED ===\n")

        # ==================== Stage 2: Text Pre-training ====================
        train_stage2_text_only(
            model, train_loader, vali_loader,
            num_epochs=stage2_epochs,
            output_path=output_path
        )
        print("\n=== STAGE 2 COMPLETED ===\n")

        # ==================== Stage 3: Full Multimodal Training ====================
        print("=" * 80)
        print("STAGE 3: Full Multimodal Training")
        print(f"  Epochs: {num_epochs}")
        print("=" * 80)

        train(
            model, train_loader, test_loader, vali_loader, num_epochs,
            output_path=output_path,
            use_horizon_weight=use_horizon_weight,
            horizon_decay=horizon_decay,
            contrastive_lambda=contrastive_lambda,
            gate_causal_lambda=gate_causal_lambda,
            inv_lambda=inv_lambda,
            eval_every_steps=eval_every_steps,
            simplified_mode=simplified_mode
        )

        print("=" * 80)
        print("THREE-STAGE TRAINING COMPLETED")
        print("=" * 80)
    finally:
        # Restore standard output and close file handle
        sys.stdout = original_stdout
        try:
            logger.close()
        except Exception:
            pass
        print(f"Full log saved to: {log_file_path}")
