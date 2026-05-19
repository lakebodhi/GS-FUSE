# -*- coding: utf-8 -*-
"""
Financial & Economic Evaluation Script for CAMEF
=================================================
Loads a trained CAMEF checkpoint and the corresponding dataset, runs inference
on the test split, and reports:

  Standard forecasting metrics
    - MSE, MAE, RMSE, SMAPE

  Financial / economic metrics
    - Directional Hit Rate (DHR)          – % of steps where predicted direction matches actual
    - Sharpe Ratio                         – annualised strategy Sharpe (long/short on pred signal)
    - Investment Return Ratio (Total Return) – cumulative strategy P&L starting from $1
    - Maximum Drawdown (MDD)               – worst peak-to-trough loss of cumulative strategy

Usage
-----
  python test_financial.py \
      --model_path  /path/to/model_output/lastest_model.pth \
      --series_id   NASDAQ \
      --seq_len     35  \
      --pred_len    140 \
      --batch_size  16  \
      --llama_name  /path/to/Phi-3.5-mini-instruct \
      --moment_model /path/to/MOMETN-1-large

All arguments have sensible defaults matching the train3.py configuration.
"""

import os
import sys
import math
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning,
                        message="torch.utils._pytree._register_pytree_node is deprecated")

from data.dataloader import event_set as EventSet
from model.CAMEF4P19L import (
    CAMEF,
    smape,
    make_horizon_weights,
    weighted_mse_mae,
    DualLogger,
)

# ─────────────────────────────────────────────────────────────────────────────
# 5-minute intraday data constants
# ─────────────────────────────────────────────────────────────────────────────
# NYSE / NASDAQ regular session: 09:30 – 16:00 = 390 minutes = 78 bars/day
BARS_PER_TRADING_DAY = 78
TRADING_DAYS_PER_YEAR = 252
# Correct annualisation factor for 5-minute bars
BARS_PER_YEAR = TRADING_DAYS_PER_YEAR * BARS_PER_TRADING_DAY   # 19,656

# Evaluation sub-horizons (bars): ~half-day / ~full-day / ~two-day
DEFAULT_EVAL_HORIZONS = (35, 70, 140)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CAMEF financial evaluation")

    # ── dataset ──────────────────────────────────────────────────────────────
    p.add_argument("--series_id",   default="NASDAQ",
                   help="Market series identifier (e.g. NASDAQ, SP500, INDU)")
    p.add_argument("--event_id",    type=int, default=0,
                   help="Event-type id (0 = all)")
    p.add_argument("--seq_len",     type=int, default=35)
    p.add_argument("--pred_len",    type=int, default=140)
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--event_dir",   default="data/event")
    p.add_argument("--series_dir",  default="data/series")

    # ── model ─────────────────────────────────────────────────────────────────
    p.add_argument(
        "--model_path",
        default="/home/yang/Research/CAMEF/model_output/final/"
                "CAMEF+_phi3_moment_camef4p19l_nasdaq_f140/lastest_model.pth",
        help="Path to the saved .pth checkpoint",
    )
    p.add_argument(
        "--llama_name",
        default="/home/yang/Research/CAMEF/baselines/Phi-3.5-mini-instruct/",
        help="Path / HuggingFace id of the LLM backbone",
    )
    p.add_argument(
        "--moment_model",
        default="/home/yang/Research/CAMEF/baselines/moment/MOMETN-1-large/",
        help="Path to the MOMENT backbone",
    )
    p.add_argument("--window",          type=int,   default=500)
    p.add_argument("--stride",          type=int,   default=400)
    p.add_argument("--decoder_layers",  type=int,   default=3)
    p.add_argument("--decoder_heads",   type=int,   default=8)
    p.add_argument("--max_token_num",   type=int,   default=1024)

    # ── evaluation ───────────────────────────────────────────────────────────
    p.add_argument("--split", default="test",
                   choices=["test", "vali"],
                   help="Which data split to evaluate on")
    p.add_argument("--feature_idx", type=int, default=0,
                   help="Column index used as the 'price' series for financial metrics")
    p.add_argument("--annualize_factor", type=float, default=BARS_PER_YEAR,
                   help="Annualisation factor for Sharpe. Default 19656 = 252 days × 78 "
                        "five-minute bars/day (correct for 5-min intraday data).")
    p.add_argument("--risk_free_rate", type=float, default=0.0,
                   help="Annual risk-free rate (e.g. 0.05 = 5%%). Divided internally by "
                        "annualize_factor to get the per-bar rate used in Sharpe.")
    p.add_argument("--eval_horizons", type=int, nargs="+",
                   default=list(DEFAULT_EVAL_HORIZONS),
                   help="Sub-horizons (in bars) to evaluate. "
                        "Default: 35 (~0.5 day) 70 (~1 day) 140 (~2 days).")
    p.add_argument("--use_inverse_transform", action="store_true",
                   help="Inverse-transform predictions to original price space before "
                        "computing financial metrics (requires scaler to be available)")
    p.add_argument("--output_dir", default=None,
                   help="If set, write a CSV of per-sample metrics + summary log here")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(args, d: int) -> CAMEF:
    """Instantiate CAMEF and load checkpoint weights."""
    from transformers import AutoTokenizer

    model = CAMEF(
        llama_name=args.llama_name,
        moment=args.moment_model,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        d=d,
        window=args.window,
        stride=args.stride,
        batch_size=args.batch_size,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        use_ts_memory=False,
        max_token_num=args.max_token_num,
    )
    print(f"\n[load] Loading checkpoint: {args.model_path}")
    try:
        model.load_model_combined(save_path=args.model_path, strict=False)
    except Exception as e:
        # The tokenizer.json saved in the checkpoint folder can be incompatible
        # with the installed tokenizers library version (Rust-level JSON parse
        # error).  load_model_combined loads the tokenizer at the VERY LAST
        # step, so all model weights are already in memory at this point.
        # We simply re-load the tokenizer from the original llama_name path.
        err = str(e)
        if "ModelWrapper" in err or "untagged enum" in err or "tokenizer" in err.lower():
            print(f"[warn] Checkpoint tokenizer.json is incompatible with the current "
                  f"tokenizers library:\n       {e}")
            print(f"[warn] All model weights were loaded successfully before this error.")
            print(f"[warn] Reloading tokenizer from: {args.llama_name}")
            model.tokenizer = AutoTokenizer.from_pretrained(
                args.llama_name, use_fast=True
            )
            if model.tokenizer.pad_token is None:
                model.tokenizer.pad_token = model.tokenizer.eos_token
            print(f"[load] Tokenizer reloaded OK.")
        else:
            raise   # unexpected error – re-raise as usual

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference – collect (predictions, targets, sequences)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model, data_loader, device):
    """
    Run inference over *data_loader* and collect numpy arrays.

    Returns
    -------
    preds   : np.ndarray  (N, d, pred_len)   – model outputs in scaled space
    targets : np.ndarray  (N, d, pred_len)   – ground-truth in scaled space
    seqs    : np.ndarray  (N, d, seq_len)    – input context in scaled space
    """
    model.eval()
    all_preds, all_targets, all_seqs = [], [], []

    for batch in tqdm(data_loader, desc="Running inference", ncols=100):
        (batch_text,
         batch_sent_reports,
         batch_negative_type_reports,
         _batch_seq,           # unscaled
         _batch_pred,          # unscaled
         batch_seq_scale,
         batch_pred_scale) = batch

        # re-zip the list-of-lists returned by the DataLoader
        batch_sent_reports          = list(map(list, zip(*batch_sent_reports)))
        batch_negative_type_reports = list(map(list, zip(*batch_negative_type_reports)))

        (output, *_) = model.predict_batch_contrastive(
            batch_text,
            batch_sent_reports,
            batch_negative_type_reports,
            batch_seq_scale,
            return_analysis=False,
        )

        all_preds.append(output.detach().cpu().numpy())        # (B, d, pred_len)
        all_targets.append(batch_pred_scale.numpy())           # (B, d, pred_len)
        all_seqs.append(batch_seq_scale.numpy())               # (B, d, seq_len)

    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)
    seqs    = np.concatenate(all_seqs,    axis=0)
    return preds, targets, seqs


# ─────────────────────────────────────────────────────────────────────────────
# Standard forecasting metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_standard_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """MSE, MAE, RMSE, SMAPE – averaged over all samples × features × steps."""
    diff   = preds - targets
    mse    = float(np.mean(diff ** 2))
    mae    = float(np.mean(np.abs(diff)))
    rmse   = math.sqrt(mse)
    denom  = np.abs(preds) + np.abs(targets)
    smape_ = float(np.mean(2.0 * np.abs(diff) / np.clip(denom, 1e-6, None)) * 100.0)
    return {"mse": mse, "mae": mae, "rmse": rmse, "smape": smape_}


# ─────────────────────────────────────────────────────────────────────────────
# Financial / economic metrics
# ─────────────────────────────────────────────────────────────────────────────

def _financial_metrics_at_horizon(
    p_H: np.ndarray,        # (N, H) predicted values in scaled space
    t_H: np.ndarray,        # (N, H) actual values in scaled space
    last_seq: np.ndarray,   # (N,)   last known context value in scaled space
    risk_free_rate: float,  # annual risk-free rate (e.g. 0.05 = 5%)
    annualize_factor: float,# bars per year (19 656 for 5-min data)
) -> dict:
    """
    Core financial-metric computation for a single evaluation horizon H.

    Data space
    ----------
    All inputs are in the StandardScaler-normalised (scaled) space.
    First-differences of scaled prices are used as the P&L unit throughout.
    They are NOT fractional (percentage) returns, so:

      • Sharpe   – mean/std ratio × √annualize_factor is dimensionless and
                   correct regardless of the unit, provided every step's
                   return is expressed in the *same* unit.

      • IRR      – Investment Return Ratio is defined as a *skill score*:
                       IRR = Σ strategy_P&L / Σ |actual_move|
                   This equals +1 for perfect predictions, 0 for random
                   (DHR ≈ 50%), −1 for always-wrong predictions.
                   It is bounded, scale-invariant, and has a clear economic
                   interpretation: "fraction of the maximum achievable P&L
                   that was captured".

      • MDD      – Maximum Drawdown is computed on a per-event equity curve.
                   Each event's contribution = event_skill_score ∈ [−1, 1].
                   Equity starts at 1 and changes by event_skill / N_events
                   per event → equity range ≈ [0, 2].  This avoids the
                   artificial smoothing from step-level granularity and gives
                   a meaningful drawdown in percentage terms.

    Annualisation
    -------------
    For 5-minute intraday data:
        annualize_factor = 252 × 78 = 19 656 bars/year
    The risk-free rate is annual; it is converted to a per-bar rate internally:
        rf_per_bar = risk_free_rate / annualize_factor
    """
    N, H = p_H.shape

    # ── step-wise differences (P&L proxy in scaled space) ────────────────────
    actual_ret = np.empty((N, H), dtype=np.float64)
    actual_ret[:, 0]  = t_H[:, 0] - last_seq      # first step vs last context bar
    actual_ret[:, 1:] = np.diff(t_H, axis=1)

    pred_ret = np.empty((N, H), dtype=np.float64)
    pred_ret[:, 0]  = p_H[:, 0] - last_seq
    pred_ret[:, 1:] = np.diff(p_H, axis=1)

    # ── 1. Directional Hit Rate ───────────────────────────────────────────────
    # Exclude steps where the model predicts no change (flat signal = no trade).
    nonzero = pred_ret != 0
    hit      = (np.sign(pred_ret) == np.sign(actual_ret)) & nonzero

    dhr = float(hit[nonzero].mean()) if nonzero.any() else float("nan")

    half = H // 2
    m1, m2 = nonzero[:, :half], nonzero[:, half:]
    dhr_first  = float(hit[:, :half][m1].mean()) if m1.any() else float("nan")
    dhr_second = float(hit[:, half:][m2].mean()) if m2.any() else float("nan")

    # ── 2. Strategy P&L ──────────────────────────────────────────────────────
    signals      = np.sign(pred_ret)              # +1 long / −1 short / 0 flat
    strategy_ret = (signals * actual_ret).astype(np.float64)   # (N, H)

    flat_ret = strategy_ret.flatten()   # (N × H,) – kept for potential future use

    # ── 3. Sharpe Ratio (holding-period level, annualised + B&H baseline) ───────
    # Each event produces ONE trade held for H bars.  The per-event P&L is the
    # sum of H step-level strategy returns; the relevant annualisation factor is
    # the number of H-bar holding periods that fit in a year, NOT the number of
    # 5-minute bars.
    #
    #   periods_per_year = BARS_PER_YEAR / H
    #   e.g. H=35  → 19 656/35 ≈ 562  (about twice-daily rebalancing)
    #        H=70  → 19 656/70 ≈ 281  (roughly daily)
    #        H=140 → 19 656/140 ≈ 140 (roughly every two days)
    #
    # WARNING – inflated Sharpe from trending test periods
    # -------------------------------------------------------
    # If the test set falls in a one-sided (e.g. strongly bullish) market, a
    # model that always predicts "up" will show artificially high step-level
    # DHR and Sharpe even with ZERO genuine skill.  We therefore also compute:
    #   • Buy-and-hold (B&H) baseline Sharpe   → always long, hold H bars
    #   • Excess (alpha) Sharpe                → strategy minus B&H
    # The alpha Sharpe isolates model skill from pure trend-following.
    periods_per_year = annualize_factor / max(H, 1)

    # Per-event (holding-period) returns
    event_pnl_hp  = strategy_ret.sum(axis=1)      # (N,) – total P&L over H steps
    rf_per_period = risk_free_rate / max(periods_per_year, 1)
    event_excess  = event_pnl_hp - rf_per_period
    mu            = float(np.mean(event_excess))
    sigma         = float(np.std(event_excess, ddof=1)) + 1e-10
    sharpe        = (mu / sigma) * math.sqrt(periods_per_year)

    # Buy-and-hold baseline: always long, never short
    bah_pnl_hp   = actual_ret.sum(axis=1)         # (N,) just hold H bars each event
    bah_excess   = bah_pnl_hp - rf_per_period
    bah_mu       = float(np.mean(bah_excess))
    bah_sigma    = float(np.std(bah_excess, ddof=1)) + 1e-10
    bah_sharpe   = (bah_mu / bah_sigma) * math.sqrt(periods_per_year)

    # Alpha Sharpe: excess P&L over buy-and-hold, annualised
    # Positive alpha Sharpe = model genuinely adds value beyond the market trend
    alpha_pnl    = event_pnl_hp - bah_pnl_hp      # (N,) strategy – B&H per event
    alpha_excess = alpha_pnl - rf_per_period
    alpha_mu     = float(np.mean(alpha_excess))
    alpha_sigma  = float(np.std(alpha_excess, ddof=1)) + 1e-10
    alpha_sharpe = (alpha_mu / alpha_sigma) * math.sqrt(periods_per_year)

    # ── 4. Investment Return Ratio (skill score) ──────────────────────────────
    # IRR = total realised P&L / total achievable P&L (with perfect predictions)
    # Uses the SAME weighted denominator as the equity curve below so that
    # equity[-1] - 1.0 == IRR exactly (both numerically consistent).
    # Bounded in [−1, +1]; scale-invariant.
    event_max_pnl  = np.abs(actual_ret).sum(axis=1)    # (N,) per-event achievable P&L
    total_max_pnl  = float(event_max_pnl.sum()) + 1e-10
    total_pnl      = float(event_pnl_hp.sum())          # == sum(flat_ret)
    irr            = total_pnl / total_max_pnl           # e.g. 0.02 → reported as 2 %

    # ── 5. Equity curve → MDD & Calmar ───────────────────────────────────────
    # Build a running P&L curve that is fully consistent with IRR:
    #   - normalise each event's P&L by the GLOBAL max achievable P&L (same
    #     denominator as IRR) so equity[-1] - 1.0 == IRR exactly.
    #   - prepend 1.0 so equity starts at the pre-trade level; this ensures
    #     that a drawdown occurring on the very first events is captured and
    #     that the running peak is initialised to 1.0.
    cum_pnl = np.cumsum(event_pnl_hp) / total_max_pnl  # (N,) running fraction captured
    equity  = np.concatenate([[1.0], 1.0 + cum_pnl])   # (N+1,) starts at 1.0

    running_peak = np.maximum.accumulate(equity)
    dd           = (equity - running_peak) / (running_peak + 1e-10)
    mdd          = float(dd.min())   # ≤ 0; e.g. −0.05 = −5 %

    # Calmar = IRR / |MDD|  (both in the same normalised skill-score units,
    # so the ratio is a dimensionless return-per-unit-of-drawdown measure).
    abs_mdd = abs(mdd)
    calmar  = irr / abs_mdd if abs_mdd > 1e-10 else float("inf")

    # ── 6. Endpoint / level-based strategies ─────────────────────────────────
    # The step-level strategy (above) re-enters every single 5-min bar and is
    # limited by high-frequency zigzag noise.  Even when MSE is very low the
    # model outputs SMOOTH predicted levels, so sign(Δpred) ≈ random vs the
    # noisy sign(Δactual) → step-level DHR ≈ 50% → step Sharpe limited.
    #
    # All three endpoint variants below enter ONCE per event and hold for H
    # bars.  They differ only in how the signal magnitude is determined.
    # They use the SAME annualisation factor (periods_per_year = BARS/H).
    #
    # Why endpoint Sharpe still decreases with H
    # -------------------------------------------
    # std(ep_pnl) ∝ √H  (more noise accumulates over a longer hold)
    # mean(ep_pnl) ∝ (DHR − 0.5) × √H  (skill × horizon move)
    # mean/std ∝ (DHR − 0.5)  → roughly constant
    # Sharpe = mean/std × √(BARS/H)  → drops as 1/√H
    #
    # The magnitude-weighted variant breaks this ceiling: larger predicted
    # moves tend to be correct more often, so weighting by prediction
    # confidence concentrates exposure on higher-DHR events.

    total_pred_move   = p_H[:, -1] - last_seq     # (N,) predicted move over full H bars
    total_actual_move = t_H[:, -1] - last_seq     # (N,) actual   move over full H bars
    ep_rf             = risk_free_rate / max(periods_per_year, 1)

    def _ep_stats(pnl_vec: np.ndarray, actual_abs_vec: np.ndarray):
        """Shared equity-curve + Sharpe helper for an arbitrary per-event P&L."""
        rf_excess   = pnl_vec - ep_rf
        mu_         = float(np.mean(rf_excess))
        sigma_      = float(np.std(rf_excess, ddof=1)) + 1e-10
        sharpe_     = (mu_ / sigma_) * math.sqrt(periods_per_year)
        denom_      = float(actual_abs_vec.sum()) + 1e-10
        irr_        = float(pnl_vec.sum()) / denom_
        cum_        = np.cumsum(pnl_vec) / denom_
        eq_         = np.concatenate([[1.0], 1.0 + cum_])
        pk_         = np.maximum.accumulate(eq_)
        dd_         = (eq_ - pk_) / (pk_ + 1e-10)
        mdd_        = float(dd_.min())
        calmar_     = irr_ / abs(mdd_) if abs(mdd_) > 1e-10 else float("inf")
        return sharpe_, irr_, mdd_, calmar_

    # ── 6a. Binary-signal endpoint (flat ±1 position) ────────────────────────
    ep_signal  = np.sign(total_pred_move)       # +1 / −1 / 0
    ep_nonzero = ep_signal != 0
    ep_hit     = (np.sign(total_pred_move) == np.sign(total_actual_move)) & ep_nonzero
    ep_dhr     = float(ep_hit[ep_nonzero].mean()) if ep_nonzero.any() else float("nan")
    ep_pnl     = ep_signal * total_actual_move
    ep_sharpe, ep_irr, ep_mdd, ep_calmar = _ep_stats(ep_pnl, np.abs(total_actual_move))

    # ── 6b. Magnitude-weighted endpoint (position ∝ predicted move size) ─────
    # Rationale: the model's predicted move magnitude reflects its confidence.
    # Larger predictions → more likely correct (if model is calibrated) →
    # concentrate capital on high-conviction events.
    # Position is z-scored so mean|position| = 1 (same notional as binary).
    pred_std   = float(np.std(total_pred_move)) + 1e-10
    mw_signal  = total_pred_move / pred_std       # (N,) continuous position size
    mw_pnl     = mw_signal * total_actual_move    # (N,) weighted P&L

    # DHR equivalent: weighted accuracy — sum of correctly-signed P&L / total |P&L|
    mw_correct_pnl   = np.where(np.sign(mw_signal) == np.sign(total_actual_move),
                                np.abs(mw_pnl), 0.0)
    mw_dhr_weighted  = float(mw_correct_pnl.sum() / (np.abs(mw_pnl).sum() + 1e-10))
    mw_sharpe, mw_irr, mw_mdd, mw_calmar = _ep_stats(mw_pnl, np.abs(total_actual_move))

    # ── 6c. Confidence-filtered endpoint (trade only high-conviction events) ──
    # Only take positions where |predicted_move| > median(|predicted_move|).
    # Fewer trades but higher average DHR on the selected subset → higher Sharpe.
    # The 50th-percentile threshold filters ~half the events; adjust if needed.
    conf_threshold = float(np.percentile(np.abs(total_pred_move), 50))
    conf_mask      = np.abs(total_pred_move) > conf_threshold
    if conf_mask.sum() > 1:
        cf_pred   = total_pred_move[conf_mask]
        cf_actual = total_actual_move[conf_mask]
        cf_signal = np.sign(cf_pred)
        cf_hit    = (np.sign(cf_pred) == np.sign(cf_actual)) & (cf_signal != 0)
        cf_dhr    = float(cf_hit[cf_signal != 0].mean()) if (cf_signal != 0).any() else float("nan")
        cf_pnl    = cf_signal * cf_actual
        # Annualisation: fewer events → same periods_per_year (calendar-based)
        cf_n_ratio  = conf_mask.sum() / max(N, 1)   # fraction of events traded
        cf_rf_adj   = ep_rf
        cf_excess   = cf_pnl - cf_rf_adj
        cf_mu       = float(np.mean(cf_excess))
        cf_sigma    = float(np.std(cf_excess, ddof=1)) + 1e-10
        cf_sharpe   = (cf_mu / cf_sigma) * math.sqrt(periods_per_year)
        cf_irr      = float(cf_pnl.sum()) / (float(np.abs(cf_actual).sum()) + 1e-10)
        cf_cum      = np.cumsum(cf_pnl) / (float(np.abs(cf_actual).sum()) + 1e-10)
        cf_eq       = np.concatenate([[1.0], 1.0 + cf_cum])
        cf_pk       = np.maximum.accumulate(cf_eq)
        cf_dd       = (cf_eq - cf_pk) / (cf_pk + 1e-10)
        cf_mdd      = float(cf_dd.min())
        cf_calmar   = cf_irr / abs(cf_mdd) if abs(cf_mdd) > 1e-10 else float("inf")
        cf_pct_traded = float(cf_n_ratio * 100)
    else:
        cf_dhr = cf_sharpe = cf_irr = cf_mdd = cf_calmar = float("nan")
        cf_pct_traded = 0.0

    return {
        # ── step-level (trade every 5-min bar; limited by HF noise) ──────────
        "directional_hit_rate":    dhr,
        "dhr_first_half":          dhr_first,
        "dhr_second_half":         dhr_second,
        "sharpe_ratio":            sharpe,
        "investment_return_ratio": irr,
        "max_drawdown":            mdd,
        "calmar_ratio":            calmar,
        # ── endpoint binary (flat ±1; direct level-quality test) ─────────────
        "ep_directional_hit_rate":    ep_dhr,
        "ep_sharpe_ratio":            ep_sharpe,
        "ep_investment_return_ratio": ep_irr,
        "ep_max_drawdown":            ep_mdd,
        "ep_calmar_ratio":            ep_calmar,
        # ── endpoint magnitude-weighted (position ∝ predicted move) ──────────
        "mw_dhr_weighted":            mw_dhr_weighted,
        "mw_sharpe_ratio":            mw_sharpe,
        "mw_investment_return_ratio": mw_irr,
        "mw_max_drawdown":            mw_mdd,
        "mw_calmar_ratio":            mw_calmar,
        # ── endpoint confidence-filtered (top-50% conviction trades only) ─────
        "cf_pct_traded":              cf_pct_traded,
        "cf_directional_hit_rate":    cf_dhr,
        "cf_sharpe_ratio":            cf_sharpe,
        "cf_investment_return_ratio": cf_irr,
        "cf_max_drawdown":            cf_mdd,
        "cf_calmar_ratio":            cf_calmar,
    }


def compute_financial_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    seqs: np.ndarray,
    feature_idx: int = 0,
    risk_free_rate: float = 0.0,
    annualize_factor: float = BARS_PER_YEAR,
    eval_horizons: tuple = DEFAULT_EVAL_HORIZONS,
) -> dict:
    """
    Compute financial metrics at each requested sub-horizon.

    Parameters
    ----------
    preds, targets : (N, d, pred_len) arrays in scaled space
    seqs           : (N, d, seq_len)  arrays in scaled space
    feature_idx    : column treated as the primary price series
    risk_free_rate : annual rate, e.g. 0.05 for 5 %
    annualize_factor : bars per year – default 19 656 for 5-min data
    eval_horizons  : tuple of bar counts to evaluate (e.g. 35, 70, 140)
                     Values > pred_len are silently skipped.

    Returns
    -------
    dict  { horizon_H : {metric: value, ...}, ... }
    The key ``"full"`` always contains results for the complete pred_len horizon.
    """
    N, _d, pred_len = preds.shape
    p_series  = preds[:, feature_idx, :].astype(np.float64)   # (N, pred_len)
    t_series  = targets[:, feature_idx, :].astype(np.float64)
    last_seq  = seqs[:, feature_idx, -1].astype(np.float64)   # (N,)

    results = {}
    horizons = sorted({int(h) for h in eval_horizons if 0 < int(h) <= pred_len})
    # always include the full horizon
    if pred_len not in horizons:
        horizons.append(pred_len)

    for H in horizons:
        label = "full" if H == pred_len else f"H{H}"
        results[label] = _financial_metrics_at_horizon(
            p_series[:, :H], t_series[:, :H], last_seq,
            risk_free_rate=risk_free_rate,
            annualize_factor=annualize_factor,
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

def _horizon_label(key: str, pred_len: int) -> str:
    if key == "full":
        return f"Full horizon ({pred_len} bars ≈ {pred_len/BARS_PER_TRADING_DAY:.1f} days)"
    bars = int(key[1:])
    return f"H={bars:3d} bars  (≈{bars/BARS_PER_TRADING_DAY:.1f} trading days)"


def print_metrics(standard: dict, financial: dict, split: str = "test",
                  pred_len: int = 140):
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  CAMEF Evaluation Results  [{split.upper()} split]")
    print(f"  5-min bars  |  annualise factor = {BARS_PER_YEAR:,} bars/yr")
    print(sep)


    print("\n── Standard Forecasting Metrics (scaled space) ────────────────")
    print(f"  MSE   : {standard['mse']:.6f}")
    print(f"  MAE   : {standard['mae']:.6f}")
    print(f"  RMSE  : {standard['rmse']:.6f}")
    print(f"  SMAPE : {standard['smape']:.4f} %")

    # Sort so sub-horizons appear before "full"
    keys = sorted(financial.keys(),
                  key=lambda k: (1 if k == "full" else 0, k))

    for key in keys:
        m = financial[key]
        label = _horizon_label(key, pred_len)
        print(f"\n── Financial Metrics  [{label}]")

        print(f"  ┌ Step-level strategy  (re-enter every 5-min bar within the horizon)")
        print(f"  │  Directional Hit Rate   : {m['directional_hit_rate']*100:.2f} %"
              f"   [1st-half {m['dhr_first_half']*100:.2f}%  2nd-half {m['dhr_second_half']*100:.2f}%]")
        print(f"  │  Sharpe (annualised)    : {m['sharpe_ratio']:.4f}"
              f"   ← limited by 5-min noise; DHR ≈ 50% even with low MSE")
        print(f"  │  IRR (skill score)      : {m['investment_return_ratio']*100:.4f} %"
              f"   [+100%=perfect, 0=random]")
        print(f"  │  Max Drawdown           : {m['max_drawdown']*100:.4f} %")
        print(f"  └  Calmar Ratio           : {m['calmar_ratio']:.4f}")

        H_label = key if key != "full" else str(pred_len)
        print(f"  ┌ Endpoint binary  (enter ONCE per event; ±1 position; hold {H_label} bars)")
        print(f"  │  Directional Hit Rate   : {m['ep_directional_hit_rate']*100:.2f} %"
              f"   ← tests overall level-prediction accuracy")
        print(f"  │  Sharpe (annualised)    : {m['ep_sharpe_ratio']:.4f}"
              f"   ← drops as 1/√H because noise ∝ √H")
        print(f"  │  IRR (skill score)      : {m['ep_investment_return_ratio']*100:.4f} %")
        print(f"  │  Max Drawdown           : {m['ep_max_drawdown']*100:.4f} %")
        print(f"  └  Calmar Ratio           : {m['ep_calmar_ratio']:.4f}")

        print(f"  ┌ Endpoint magnitude-weighted  (position ∝ |predicted move|; hold {H_label} bars)")
        print(f"  │  Weighted DHR           : {m['mw_dhr_weighted']*100:.2f} %"
              f"   ← P&L-weighted accuracy on larger moves")
        print(f"  │  Sharpe (annualised)    : {m['mw_sharpe_ratio']:.4f}"
              f"   ← concentrates on high-conviction events → higher Sharpe")
        print(f"  │  IRR (skill score)      : {m['mw_investment_return_ratio']*100:.4f} %")
        print(f"  │  Max Drawdown           : {m['mw_max_drawdown']*100:.4f} %")
        print(f"  └  Calmar Ratio           : {m['mw_calmar_ratio']:.4f}")

        print(f"  ┌ Endpoint confidence-filtered  (top-50% conviction events; hold {H_label} bars)")
        print(f"  │  Events traded          : {m['cf_pct_traded']:.1f} %  of test set")
        print(f"  │  Directional Hit Rate   : {m['cf_directional_hit_rate']*100:.2f} %"
              f"   ← should exceed binary DHR if model is calibrated")
        print(f"  │  Sharpe (annualised)    : {m['cf_sharpe_ratio']:.4f}"
              f"   ← quality-over-quantity improvement")
        print(f"  │  IRR (skill score)      : {m['cf_investment_return_ratio']*100:.4f} %")
        print(f"  │  Max Drawdown           : {m['cf_max_drawdown']*100:.4f} %")
        print(f"  └  Calmar Ratio           : {m['cf_calmar_ratio']:.4f}")
    print(f"\n{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Optional CSV output
# ─────────────────────────────────────────────────────────────────────────────

def save_results(output_dir: str, standard: dict, financial: dict,
                 preds: np.ndarray, targets: np.ndarray,
                 feature_idx: int, split: str):
    import json
    os.makedirs(output_dir, exist_ok=True)

    # ── per-sample forecasting metrics CSV ────────────────────────────────────
    N, d, pred_len = preds.shape
    p = preds[:, feature_idx, :]
    t = targets[:, feature_idx, :]

    per_sample_mse   = np.mean((p - t) ** 2, axis=1)
    per_sample_mae   = np.mean(np.abs(p - t), axis=1)
    per_sample_rmse  = np.sqrt(per_sample_mse)
    denom            = np.abs(p) + np.abs(t)
    per_sample_smape = np.mean(
        2.0 * np.abs(p - t) / np.clip(denom, 1e-6, None), axis=1
    ) * 100.0

    df = pd.DataFrame({
        "sample_idx": np.arange(N),
        "mse":        per_sample_mse,
        "mae":        per_sample_mae,
        "rmse":       per_sample_rmse,
        "smape":      per_sample_smape,
    })
    sample_csv = os.path.join(output_dir, f"{split}_per_sample_metrics.csv")
    df.to_csv(sample_csv, index=False)
    print(f"[save] Per-sample metrics   → {sample_csv}")

    # ── summary JSON (all horizons) ───────────────────────────────────────────
    summary = {
        "split":          split,
        "feature_idx":    feature_idx,
        "bars_per_year":  BARS_PER_YEAR,
        "standard":       standard,
        "financial":      financial,   # keyed by horizon label
    }
    summary_path = os.path.join(output_dir, f"{split}_financial_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] Summary metrics      → {summary_path}")

    # ── flat CSV for easy comparison across runs ──────────────────────────────
    rows = []
    for horizon_key, m in financial.items():
        row = {"split": split, "horizon": horizon_key, **standard, **m}
        rows.append(row)
    flat_df = pd.DataFrame(rows)
    flat_csv = os.path.join(output_dir, f"{split}_metrics_summary.csv")
    flat_df.to_csv(flat_csv, index=False)
    print(f"[save] Flat summary CSV     → {flat_csv}")

    log_path = os.path.join(output_dir, f"{split}_eval.log")
    return log_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── optional dual logging ─────────────────────────────────────────────────
    dual_log = None
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        log_path = os.path.join(args.output_dir, f"{args.split}_eval.log")
        dual_log = DualLogger(log_path)
        sys.stdout = dual_log
        print(f"[log] Writing log to: {log_path}")

    # ── data ──────────────────────────────────────────────────────────────────
    print(f"\n[data] Loading dataset: series={args.series_id}, "
          f"seq_len={args.seq_len}, pred_len={args.pred_len}")
    dataset = EventSet(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        event_id=args.event_id,
        series_id=args.series_id,
        shuffle=False,
        batch_size=args.batch_size,
        scale=True,
        event_dir=args.event_dir,
        series_dir=args.series_dir,
    )

    loader = dataset.test_loader if args.split == "test" else dataset.vali_loader
    d      = dataset.d
    scaler = dataset.scaler   # sklearn StandardScaler; kept for optional inverse-transform

    # ── model ─────────────────────────────────────────────────────────────────
    model = load_model(args, d)
    device = model.device

    # ── inference ─────────────────────────────────────────────────────────────
    print(f"\n[eval] Running inference on {args.split} split …")
    preds, targets, seqs = collect_predictions(model, loader, device)

    print(f"[eval] Collected  preds={preds.shape}  targets={targets.shape}  seqs={seqs.shape}")

    # ── optional inverse transform ────────────────────────────────────────────
    if args.use_inverse_transform:
        print("[eval] Inverse-transforming predictions to original price space …")
        N, d_feat, L = preds.shape
        preds_r   = scaler.inverse_transform(preds.reshape(-1, d_feat)).reshape(N, d_feat, L)
        targets_r = scaler.inverse_transform(targets.reshape(-1, d_feat)).reshape(N, d_feat, L)
        seqs_r    = scaler.inverse_transform(seqs.reshape(-1, d_feat)).reshape(N, d_feat, seqs.shape[2])
        preds_fin, targets_fin, seqs_fin = preds_r, targets_r, seqs_r
    else:
        preds_fin, targets_fin, seqs_fin = preds, targets, seqs

    # ── metrics ───────────────────────────────────────────────────────────────
    standard  = compute_standard_metrics(preds, targets)   # always in scaled space
    financial = compute_financial_metrics(
        preds_fin, targets_fin, seqs_fin,
        feature_idx=args.feature_idx,
        risk_free_rate=args.risk_free_rate,
        annualize_factor=args.annualize_factor,
        eval_horizons=tuple(args.eval_horizons),
    )

    print_metrics(standard, financial, split=args.split, pred_len=args.pred_len)

    # ── save ──────────────────────────────────────────────────────────────────
    if args.output_dir:
        save_results(
            args.output_dir, standard, financial,
            preds, targets,
            feature_idx=args.feature_idx,
            split=args.split,
        )

    if dual_log is not None:
        sys.stdout = dual_log.terminal
        dual_log.close()


if __name__ == "__main__":
    main()
