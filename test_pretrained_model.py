#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a pretrained GS-FUSE checkpoint with compact endpoint metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import warnings
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm

warnings.simplefilter("ignore")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="torch.utils._pytree._register_pytree_node is deprecated",
)

BARS_PER_TRADING_DAY = 78
TRADING_DAYS_PER_YEAR = 252
BARS_PER_YEAR = TRADING_DAYS_PER_YEAR * BARS_PER_TRADING_DAY


def _live_stream():
    """Prefer the controlling terminal so progress survives `conda run` capture."""
    try:
        return open("/dev/tty", "w", encoding="utf-8", buffering=1)
    except OSError:
        return sys.stdout


PROGRESS_STREAM = _live_stream()


def live_print(message: str = "") -> None:
    print(message, file=PROGRESS_STREAM, flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Load a pretrained GS-FUSE checkpoint and report MSE/MAE plus "
            "endpoint binary DHR/Sharpe."
        )
    )

    p.add_argument("--model-path", "--model_path", required=True, help="Path to .pth checkpoint")
    p.add_argument("--split", choices=["test", "vali"], default="test")

    p.add_argument("--series-id", "--series_id", default="NASDAQ")
    p.add_argument("--event-id", "--event_id", type=int, default=0)
    p.add_argument("--seq-len", "--seq_len", type=int, default=35)
    p.add_argument("--pred-len", "--pred_len", type=int, default=140)
    p.add_argument("--batch-size", "--batch_size", type=int, default=16)
    p.add_argument("--event-dir", "--event_dir", default="data/event")
    p.add_argument("--series-dir", "--series_dir", default="data/series")

    p.add_argument("--ts-backbone", choices=["moment", "kronos"], default="moment")
    p.add_argument("--text-backbone", choices=["llama", "phi"], default="llama")
    p.add_argument("--text-model-path", "--text_model_path", required=True)
    p.add_argument("--moment-path", "--moment_path", default=None)
    p.add_argument("--kronos-model-path", "--kronos_model_path", default=None)
    p.add_argument("--kronos-tokenizer-path", "--kronos_tokenizer_path", default=None)

    p.add_argument("--window", type=int, default=500)
    p.add_argument("--stride", type=int, default=400)
    p.add_argument("--decoder-layers", "--decoder_layers", type=int, default=3)
    p.add_argument("--decoder-heads", "--decoder_heads", type=int, default=8)
    p.add_argument("--max-token-num", "--max_token_num", type=int, default=1024)

    p.add_argument(
        "--feature-idx",
        "--feature_idx",
        type=int,
        default=0,
        help="Feature/channel used for endpoint binary DHR and Sharpe.",
    )
    p.add_argument("--annualize-factor", "--annualize_factor", type=float, default=BARS_PER_YEAR)
    p.add_argument("--risk-free-rate", "--risk_free_rate", type=float, default=0.0)
    p.add_argument(
        "--eval-horizons",
        "--eval_horizons",
        type=int,
        nargs="+",
        default=[35, 70, 140],
        help="Forecast sub-horizons to evaluate. Values > pred_len are skipped.",
    )
    p.add_argument(
        "--inverse-transform",
        action="store_true",
        help="Compute endpoint metrics in original scaler space instead of scaled space.",
    )
    p.add_argument("--output-json", "--output_json", default=None)
    p.add_argument("--output-csv", "--output_csv", default=None)
    return p.parse_args()


def compute_forecasting_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """MSE and MAE in the original model scripts' all-horizon style."""
    diff = preds - targets
    return {
        "mse": float(np.mean(diff ** 2)),
        "mae": float(np.mean(np.abs(diff))),
    }


def compute_endpoint_binary_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    seqs: np.ndarray,
    *,
    feature_idx: int,
    annualize_factor: float = BARS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """Endpoint binary long/short metrics for checkpoint evaluation.

    Signal is sign(predicted endpoint - last context value). Each event opens one
    unit long/short position and holds until the forecast endpoint.
    """
    p = np.asarray(preds[:, feature_idx, :], dtype=np.float64)
    t = np.asarray(targets[:, feature_idx, :], dtype=np.float64)
    s = np.asarray(seqs[:, feature_idx, :], dtype=np.float64)

    if p.ndim != 2 or t.ndim != 2 or s.ndim != 2:
        raise ValueError("preds, targets, and seqs must have shape [N, d, L]")
    if p.shape != t.shape:
        raise ValueError(f"preds and targets horizon shapes differ: {p.shape} vs {t.shape}")
    if s.shape[0] != p.shape[0]:
        raise ValueError(f"seqs and preds sample counts differ: {s.shape[0]} vs {p.shape[0]}")

    n_events, horizon = p.shape
    last_seq = s[:, -1]
    pred_move = p[:, -1] - last_seq
    actual_move = t[:, -1] - last_seq

    signal = np.sign(pred_move)
    traded = signal != 0
    hit = (np.sign(pred_move) == np.sign(actual_move)) & traded
    dhr = float(hit[traded].mean()) if traded.any() else float("nan")

    pnl = signal * actual_move
    periods_per_year = annualize_factor / max(horizon, 1)
    rf_per_period = risk_free_rate / max(periods_per_year, 1e-10)
    excess = pnl - rf_per_period
    mu = float(np.mean(excess))
    sigma = float(np.std(excess, ddof=1)) + 1e-10 if n_events > 1 else 1e-10
    sharpe = (mu / sigma) * math.sqrt(periods_per_year)

    return {
        "directional_hit_rate": dhr,
        "sharpe_ratio": float(sharpe),
        "num_events": int(n_events),
        "num_traded": int(traded.sum()),
        "pct_traded": float(traded.mean() * 100.0),
        "endpoint_pred_move_mean": float(np.mean(pred_move)),
        "endpoint_actual_move_mean": float(np.mean(actual_move)),
    }


def compute_financial_strategy_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    seqs: np.ndarray,
    *,
    feature_idx: int,
    annualize_factor: float = BARS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """Financial metrics for four long/short strategy variants.

    Variants:
    - step-wise strategy (trade every forecast step)
    - endpoint binary (unit long/short at endpoint)
    - endpoint magnitude-weighted (position proportional to predicted move)
    - endpoint confidence-filtered (top 50% predicted move magnitude)
    """
    p_h = np.asarray(preds[:, feature_idx, :], dtype=np.float64)
    t_h = np.asarray(targets[:, feature_idx, :], dtype=np.float64)
    last_seq = np.asarray(seqs[:, feature_idx, -1], dtype=np.float64)

    n_events, horizon = p_h.shape
    actual_ret = np.empty((n_events, horizon), dtype=np.float64)
    actual_ret[:, 0] = t_h[:, 0] - last_seq
    actual_ret[:, 1:] = np.diff(t_h, axis=1)

    pred_ret = np.empty((n_events, horizon), dtype=np.float64)
    pred_ret[:, 0] = p_h[:, 0] - last_seq
    pred_ret[:, 1:] = np.diff(p_h, axis=1)

    nonzero = pred_ret != 0
    hit = (np.sign(pred_ret) == np.sign(actual_ret)) & nonzero
    dhr = float(hit[nonzero].mean()) if nonzero.any() else float("nan")

    half = horizon // 2
    m1, m2 = nonzero[:, :half], nonzero[:, half:]
    dhr_first = float(hit[:, :half][m1].mean()) if m1.any() else float("nan")
    dhr_second = float(hit[:, half:][m2].mean()) if m2.any() else float("nan")

    signals = np.sign(pred_ret)
    strategy_ret = (signals * actual_ret).astype(np.float64)
    periods_per_year = annualize_factor / max(horizon, 1)
    rf_per_period = risk_free_rate / max(periods_per_year, 1e-10)
    event_pnl_hp = strategy_ret.sum(axis=1)
    event_excess = event_pnl_hp - rf_per_period
    mu = float(np.mean(event_excess))
    sigma = float(np.std(event_excess, ddof=1)) + 1e-10 if n_events > 1 else 1e-10
    sharpe = (mu / sigma) * math.sqrt(periods_per_year)

    event_max_pnl = np.abs(actual_ret).sum(axis=1)
    total_max_pnl = float(event_max_pnl.sum()) + 1e-10
    irr = float(event_pnl_hp.sum()) / total_max_pnl
    equity = np.concatenate([[1.0], 1.0 + np.cumsum(event_pnl_hp) / total_max_pnl])
    running_peak = np.maximum.accumulate(equity)
    mdd = float(((equity - running_peak) / (running_peak + 1e-10)).min())
    calmar = irr / abs(mdd) if abs(mdd) > 1e-10 else float("inf")

    total_pred_move = p_h[:, -1] - last_seq
    total_actual_move = t_h[:, -1] - last_seq
    ep_rf = risk_free_rate / max(periods_per_year, 1e-10)

    def _ep_stats(pnl_vec: np.ndarray, actual_abs_vec: np.ndarray):
        rf_excess = pnl_vec - ep_rf
        mu_ = float(np.mean(rf_excess))
        sigma_ = float(np.std(rf_excess, ddof=1)) + 1e-10 if pnl_vec.size > 1 else 1e-10
        sharpe_ = (mu_ / sigma_) * math.sqrt(periods_per_year)
        denom_ = float(actual_abs_vec.sum()) + 1e-10
        irr_ = float(pnl_vec.sum()) / denom_
        eq_ = np.concatenate([[1.0], 1.0 + np.cumsum(pnl_vec) / denom_])
        pk_ = np.maximum.accumulate(eq_)
        mdd_ = float(((eq_ - pk_) / (pk_ + 1e-10)).min())
        calmar_ = irr_ / abs(mdd_) if abs(mdd_) > 1e-10 else float("inf")
        return sharpe_, irr_, mdd_, calmar_

    ep_signal = np.sign(total_pred_move)
    ep_nonzero = ep_signal != 0
    ep_hit = (np.sign(total_pred_move) == np.sign(total_actual_move)) & ep_nonzero
    ep_dhr = float(ep_hit[ep_nonzero].mean()) if ep_nonzero.any() else float("nan")
    ep_pnl = ep_signal * total_actual_move
    ep_sharpe, ep_irr, ep_mdd, ep_calmar = _ep_stats(ep_pnl, np.abs(total_actual_move))

    pred_std = float(np.std(total_pred_move)) + 1e-10
    mw_signal = total_pred_move / pred_std
    mw_pnl = mw_signal * total_actual_move
    mw_correct_pnl = np.where(np.sign(mw_signal) == np.sign(total_actual_move), np.abs(mw_pnl), 0.0)
    mw_dhr_weighted = float(mw_correct_pnl.sum() / (np.abs(mw_pnl).sum() + 1e-10))
    mw_sharpe, mw_irr, mw_mdd, mw_calmar = _ep_stats(mw_pnl, np.abs(total_actual_move))

    conf_threshold = float(np.percentile(np.abs(total_pred_move), 50))
    conf_mask = np.abs(total_pred_move) > conf_threshold
    if conf_mask.sum() > 1:
        cf_pred = total_pred_move[conf_mask]
        cf_actual = total_actual_move[conf_mask]
        cf_signal = np.sign(cf_pred)
        cf_hit = (np.sign(cf_pred) == np.sign(cf_actual)) & (cf_signal != 0)
        cf_dhr = float(cf_hit[cf_signal != 0].mean()) if (cf_signal != 0).any() else float("nan")
        cf_pnl = cf_signal * cf_actual
        cf_excess = cf_pnl - ep_rf
        cf_mu = float(np.mean(cf_excess))
        cf_sigma = float(np.std(cf_excess, ddof=1)) + 1e-10
        cf_sharpe = (cf_mu / cf_sigma) * math.sqrt(periods_per_year)
        cf_irr = float(cf_pnl.sum()) / (float(np.abs(cf_actual).sum()) + 1e-10)
        cf_eq = np.concatenate([[1.0], 1.0 + np.cumsum(cf_pnl) / (float(np.abs(cf_actual).sum()) + 1e-10)])
        cf_pk = np.maximum.accumulate(cf_eq)
        cf_mdd = float(((cf_eq - cf_pk) / (cf_pk + 1e-10)).min())
        cf_calmar = cf_irr / abs(cf_mdd) if abs(cf_mdd) > 1e-10 else float("inf")
        cf_pct_traded = float((conf_mask.sum() / max(n_events, 1)) * 100)
    else:
        cf_dhr = cf_sharpe = cf_irr = cf_mdd = cf_calmar = float("nan")
        cf_pct_traded = 0.0

    return {
        "directional_hit_rate": dhr,
        "dhr_first_half": dhr_first,
        "dhr_second_half": dhr_second,
        "sharpe_ratio": sharpe,
        "investment_return_ratio": irr,
        "max_drawdown": mdd,
        "calmar_ratio": calmar,
        "ep_directional_hit_rate": ep_dhr,
        "ep_sharpe_ratio": ep_sharpe,
        "ep_investment_return_ratio": ep_irr,
        "ep_max_drawdown": ep_mdd,
        "ep_calmar_ratio": ep_calmar,
        "mw_dhr_weighted": mw_dhr_weighted,
        "mw_sharpe_ratio": mw_sharpe,
        "mw_investment_return_ratio": mw_irr,
        "mw_max_drawdown": mw_mdd,
        "mw_calmar_ratio": mw_calmar,
        "cf_pct_traded": cf_pct_traded,
        "cf_directional_hit_rate": cf_dhr,
        "cf_sharpe_ratio": cf_sharpe,
        "cf_investment_return_ratio": cf_irr,
        "cf_max_drawdown": cf_mdd,
        "cf_calmar_ratio": cf_calmar,
    }


def _valid_horizons(pred_len: int, horizons) -> list[int]:
    valid = sorted({int(h) for h in horizons if 0 < int(h) <= pred_len})
    if pred_len not in valid:
        valid.append(pred_len)
    return valid


def compute_horizon_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    seqs: np.ndarray,
    *,
    horizons,
    feature_idx: int,
    annualize_factor: float = BARS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> Dict[str, Dict[str, float]]:
    pred_len = preds.shape[-1]
    results = {}
    for horizon in _valid_horizons(pred_len, horizons):
        label = "full" if horizon == pred_len else f"H{horizon}"
        p_h = preds[..., :horizon]
        t_h = targets[..., :horizon]
        forecast = compute_forecasting_metrics(p_h, t_h)
        financial = compute_financial_strategy_metrics(
            p_h,
            t_h,
            seqs,
            feature_idx=feature_idx,
            annualize_factor=annualize_factor,
            risk_free_rate=risk_free_rate,
        )
        results[label] = {
            **forecast,
            **financial,
            # Backward-compatible aliases for the endpoint binary result.
            "endpoint_binary_dhr": financial["ep_directional_hit_rate"],
            "endpoint_binary_sharpe": financial["ep_sharpe_ratio"],
        "endpoint_num_events": int(p_h.shape[0]),
        "endpoint_num_traded": int((np.sign(p_h[:, feature_idx, -1] - seqs[:, feature_idx, -1]) != 0).sum()),
        "endpoint_pct_traded": float((np.sign(p_h[:, feature_idx, -1] - seqs[:, feature_idx, -1]) != 0).mean() * 100.0),
        }
    return results


def collect_predictions(model, loader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    model.eval()
    all_preds, all_targets, all_seqs = [], [], []
    total_batches = len(loader) if hasattr(loader, "__len__") else None

    if total_batches is not None:
        live_print(f"[eval] Running inference over {total_batches} batches")
    else:
        live_print("[eval] Running inference")

    with torch.no_grad():
        progress = tqdm(
            loader,
            desc="Running inference",
            total=total_batches,
            ncols=100,
            dynamic_ncols=True,
            file=PROGRESS_STREAM,
            disable=False,
            leave=True,
        )
        for batch_idx, batch in enumerate(progress, start=1):
            (
                batch_text,
                batch_sent_reports,
                batch_negative_type_reports,
                _batch_seq,
                _batch_pred,
                batch_seq_scale,
                batch_pred_scale,
            ) = batch

            batch_sent_reports = list(map(list, zip(*batch_sent_reports)))
            batch_negative_type_reports = list(map(list, zip(*batch_negative_type_reports)))
            output, *_ = model.predict_batch_contrastive(
                batch_text,
                batch_sent_reports,
                batch_negative_type_reports,
                batch_seq_scale,
                return_analysis=False,
            )

            all_preds.append(output.detach().cpu().numpy())
            all_targets.append(batch_pred_scale.numpy())
            all_seqs.append(batch_seq_scale.numpy())
            if total_batches is not None and (batch_idx % 10 == 0 or batch_idx == total_batches):
                live_print(f"[eval] processed {batch_idx}/{total_batches} batches")

    return (
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_seqs, axis=0),
    )


def load_pretrained_model(args, d: int):
    from model.gs_fuse import GSFuse

    live_print("[load] Initializing GS-FUSE model and backbones ...")
    model = GSFuse(
        text_model_path=args.text_model_path,
        ts_backbone=args.ts_backbone,
        text_backbone=args.text_backbone,
        moment_path=args.moment_path,
        kronos_model_path=args.kronos_model_path,
        kronos_tokenizer_path=args.kronos_tokenizer_path,
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
    live_print("[load] Model initialized. Loading checkpoint weights ...")
    model.load_model_combined(save_path=args.model_path, strict=False)
    live_print("[load] Checkpoint load complete.")
    model.eval()
    return model


def maybe_inverse_transform(preds, targets, seqs, scaler):
    n, d_feat, pred_len = preds.shape
    seq_len = seqs.shape[2]
    preds_r = scaler.inverse_transform(preds.reshape(-1, d_feat)).reshape(n, d_feat, pred_len)
    targets_r = scaler.inverse_transform(targets.reshape(-1, d_feat)).reshape(n, d_feat, pred_len)
    seqs_r = scaler.inverse_transform(seqs.reshape(-1, d_feat)).reshape(n, d_feat, seq_len)
    return preds_r, targets_r, seqs_r


def write_outputs(metrics: Dict[str, float], output_json: str | None, output_csv: str | None) -> None:
    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)


def main():
    from data.dataloader import event_set as EventSet

    args = parse_args()
    if args.ts_backbone == "moment" and not args.moment_path:
        raise SystemExit("--moment-path is required when --ts-backbone moment")
    if args.ts_backbone == "kronos" and (not args.kronos_model_path or not args.kronos_tokenizer_path):
        raise SystemExit("--kronos-model-path and --kronos-tokenizer-path are required for Kronos")

    live_print(
        f"[data] Loading dataset: series={args.series_id}, split={args.split}, "
        f"seq_len={args.seq_len}, pred_len={args.pred_len}"
    )
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
    live_print(
        f"[data] Dataset ready: train={len(dataset.train_set)}, "
        f"test={len(dataset.test_set)}, vali={len(dataset.vali_set)}, "
        f"eval_batches={len(loader)}"
    )

    live_print(f"[GS-FUSE eval] Loading checkpoint: {args.model_path}")
    model = load_pretrained_model(args, d=dataset.d)
    device = model.device

    live_print(f"[GS-FUSE eval] Running {args.split} inference for series={args.series_id}")
    preds, targets, seqs = collect_predictions(model, loader, device)

    endpoint_preds, endpoint_targets, endpoint_seqs = preds, targets, seqs
    if args.inverse_transform:
        endpoint_preds, endpoint_targets, endpoint_seqs = maybe_inverse_transform(
            preds, targets, seqs, dataset.scaler
        )

    horizon_metrics = compute_horizon_metrics(
        endpoint_preds,
        endpoint_targets,
        endpoint_seqs,
        horizons=args.eval_horizons,
        feature_idx=args.feature_idx,
        annualize_factor=args.annualize_factor,
        risk_free_rate=args.risk_free_rate,
    )
    full_metrics = horizon_metrics["full"]

    metrics = {
        "model_path": args.model_path,
        "split": args.split,
        "series_id": args.series_id,
        "ts_backbone": args.ts_backbone,
        "text_backbone": args.text_backbone,
        "feature_idx": int(args.feature_idx),
        "horizons": horizon_metrics,
        # Backward-compatible top-level full-horizon fields.
        **full_metrics,
    }

    live_print("\nGS-FUSE Pretrained Checkpoint Metrics")
    live_print("=" * 42)
    for label, row in horizon_metrics.items():
        live_print(f"[{label}]")
        live_print(f"  MSE                    : {row['mse']:.6f}")
        live_print(f"  MAE                    : {row['mae']:.6f}")
        live_print(
            f"  Step-wise              : DHR={row['directional_hit_rate'] * 100:.2f}% "
            f"Sharpe={row['sharpe_ratio']:.4f}"
        )
        live_print(
            f"  Endpoint binary        : DHR={row['ep_directional_hit_rate'] * 100:.2f}% "
            f"Sharpe={row['ep_sharpe_ratio']:.4f}"
        )
        live_print(
            f"  Endpoint mag-weighted  : DHR={row['mw_dhr_weighted'] * 100:.2f}% "
            f"Sharpe={row['mw_sharpe_ratio']:.4f}"
        )
        live_print(
            f"  Endpoint conf-filtered : traded={row['cf_pct_traded']:.1f}% "
            f"DHR={row['cf_directional_hit_rate'] * 100:.2f}% "
            f"Sharpe={row['cf_sharpe_ratio']:.4f}"
        )

    write_outputs(metrics, args.output_json, args.output_csv)


if __name__ == "__main__":
    main()
