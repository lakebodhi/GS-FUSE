"""
Gate openness experiments (Exp-G1, Exp-G2, Exp-G4) for CAMEF4P19L.

Implements:
  Exp-G1 Test (1): TS-only vs Full (text+TS) performance table for SP500 and NASDAQ.
  Exp-G2 Test (2): min/max/avg gate openness per event type.
  Exp-G2 Test (3): gate–utility followness (correlation + bucket plot).
  Exp-G4: Per-asset gate distribution (boxplots).

Usage:
  python test_gate_openness.py --checkpoint path/to/lastest_model.pth [--g1] [--g2] [--g3] [--g4]
  Omit --g1/--g2/--g3/--g4 to run all experiments.
  Use --data_only to write only JSON data (no PNG plots) to --output_dir.
"""

import argparse
import glob
import json
import os
import warnings
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning, message="torch.utils._pytree._register_pytree_node is deprecated")

from data.dataloader_gate_test import event_set_gate_test as EventSet
from data.sliding_window_dataloader import sliding_window_set
from model.CAMEF4P19L import CAMEF, test, TrainingConfig, smape

# Optional matplotlib for bucket plot
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Optional scipy for Spearman correlation
try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def get_article_count_for_sample(item):
    """
    Return number of event text files (full_summary + report_sent*) for this sample.
    item is a list from test_set.list; item[0] is full_summary path. Used for article-based G2 stats.
    """
    if not item or len(item) < 1:
        return 1
    path = item[0]
    if not isinstance(path, str) or not os.path.isfile(path):
        return 1
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        return 1
    n = 0
    for f in os.listdir(folder):
        if not f.endswith(".txt"):
            continue
        if f.endswith("_full_summary.txt"):
            n += 1
        elif "_report_sent" in f:
            n += 1
    return max(n, 1)


def canonical_event_type(et):
    """
    Map 12 subfolders to 6 event types. Subfolders 1-6 are main types; 7-12 are
    complements (7->1, 8->2, ..., 12->6). Returns "1".."6".
    """
    try:
        n = int(et)
        return str(((n - 1) % 6) + 1)
    except (ValueError, TypeError):
        return et if et else "0"


def run_exp_g1(model, device, series_ids, seq_len, pred_len, batch_size, event_dir, series_dir, event_id):
    """
    Exp-G1 Test (1): TS-only vs Full performance table (SP500, NASDAQ).
    """
    print("\n" + "=" * 60)
    print("Exp-G1: Text contribution — TS-only vs Full (SP500, NASDAQ)")
    print("=" * 60)

    rows = []
    for series_id in series_ids:
        ev = EventSet(
            seq_len, pred_len,
            event_id=event_id,
            series_id=series_id,
            shuffle=False,
            batch_size=batch_size,
            scale=True,
            event_dir=event_dir,
            series_dir=series_dir,
        )
        test_loader = ev.test_loader
        result = test(
            model, test_loader,
            return_analysis=False,
            compute_modality_scores=True,
        )
        # result: (avg_combined_loss, avg_mse_loss, avg_mae_loss, avg_contrastive_loss, avg_rmse_loss, avg_smape_loss, modality_metrics)
        mod = result[6]
        if mod is None:
            print(f"[{series_id}] No modality_metrics (model may not support ablations). Skipping row.")
            continue
        ts_only_rmse = mod.get("ts_only_rmse")
        ts_only_smape = mod.get("ts_only_smape")
        ts_only_mse = mod.get("ts_only_mse")
        ts_only_mae = mod.get("ts_only_mae")
        full_mse = result[1]
        full_rmse = result[4]
        full_smape = result[5]
        full_mae = result[2]
        rows.append({
            "series_id": series_id,
            "ts_only_rmse": ts_only_rmse,
            "ts_only_smape": ts_only_smape,
            "ts_only_mse": ts_only_mse,
            "ts_only_mae": ts_only_mae,
            "full_rmse": full_rmse,
            "full_smape": full_smape,
            "full_mse": full_mse,
            "full_mae": full_mae,
        })

    # Print table (includes TS-only MSE/MAE for metric consistency with paper)
    print("\n--- Table: TS-only vs Full ---")
    print(f"{'Index':<8} {'TS-only RMSE':<12} {'TS-only MSE':<12} {'TS-only MAE':<12} {'TS-only SMAPE':<12} | {'Full RMSE':<10} {'Full MSE':<10} {'Full MAE':<10} {'Full SMAPE':<10}")
    print("-" * 110)
    for r in rows:
        tsm = r.get("ts_only_mse") if r.get("ts_only_mse") is not None else float("nan")
        tma = r.get("ts_only_mae") if r.get("ts_only_mae") is not None else float("nan")
        print(f"{r['series_id']:<8} {r['ts_only_rmse']:<12.6f} {tsm:<12.6f} {tma:<12.6f} {r['ts_only_smape']:<12.4f} | {r['full_rmse']:<10.6f} {r['full_mse']:<10.6f} {r['full_mae']:<10.6f} {r['full_smape']:<10.4f}")
    return rows


def run_exp_g1_print(rows):
    """Print the G1 table from a list of dicts with series_id + ts_only_* / full_* keys."""
    if not rows:
        return
    print("\n" + "=" * 60)
    print("Exp-G1: Text contribution — TS-only vs Full (SP500, NASDAQ)")
    print("=" * 60)
    print("\n--- Table: TS-only vs Full ---")
    print(f"{'Index':<8} {'TS-only RMSE':<12} {'TS-only MSE':<12} {'TS-only MAE':<12} {'TS-only SMAPE':<12} | {'Full RMSE':<10} {'Full MSE':<10} {'Full MAE':<10} {'Full SMAPE':<10}")
    print("-" * 110)
    for r in rows:
        tsm = r.get("ts_only_mse") if r.get("ts_only_mse") is not None else float("nan")
        tma = r.get("ts_only_mae") if r.get("ts_only_mae") is not None else float("nan")
        print(f"{r['series_id']:<8} {r['ts_only_rmse']:<12.6f} {tsm:<12.6f} {tma:<12.6f} {r['ts_only_smape']:<12.4f} | {r['full_rmse']:<10.6f} {r['full_mse']:<10.6f} {r['full_mae']:<10.6f} {r['full_smape']:<10.4f}")


def _collect_g2_data(model, device, test_loader, test_set):
    """Collect G2 data only (backward compat)."""
    return _collect_all_g_data(model, device, test_loader, test_set, do_g1=False, do_g2=True, do_g4=False)


def _collect_all_g_data(model, device, test_loader, test_set, do_g1=False, do_g2=True, do_g4=False):
    """
    Run ONE pass over the test set and collect everything needed for G1, G2, and G4.
    - do_g1: accumulate full/ts_only MSE, MAE, SMAPE -> g1_metrics (one row for this series).
    - do_g2: collect event_types, article_counts, gate_text, err_ts, err_full, gc_score -> g2_data.
    - do_g4: collect gate values list -> g4_gate_list.
    Returns dict with optional g1_metrics, g2_data, g4_gate_list; or None if no data.
    """
    has_et = hasattr(test_set, "list") and test_set.list and len(test_set.list[0]) == 8
    if do_g2 and not has_et:
        return None
    event_types = [canonical_event_type(test_set.list[i][5]) for i in range(len(test_set))] if has_et else []
    article_counts = [get_article_count_for_sample(test_set.list[i]) for i in range(len(test_set))] if has_et else []
    gate_text_list = []
    err_ts_list = []
    err_full_list = []
    gc_score_list = []
    et_list = []
    n_list = []
    if do_g1:
        sum_mse_f, sum_mae_f, sum_smape_f = 0.0, 0.0, 0.0
        sum_mse_ts, sum_mae_ts, sum_smape_ts = 0.0, 0.0, 0.0
        n_batches_g1 = 0
    gate_list_g4 = [] if do_g4 else None
    model.eval()
    device = next(model.parameters()).device
    start_idx = 0
    # Clear cache at the start of data collection to ensure minimal initial memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="G1+G2+G4 single pass", ncols=100)):
            batch_text, batch_sent_reports, batch_negative_type_reprots, batch_seq, batch_pred, batch_seq_scale, batch_pred_scale = batch[:7]
            batch_sent_reports = list(map(list, zip(*batch_sent_reports)))
            batch_negative_type_reprots = list(map(list, zip(*batch_negative_type_reprots)))
            result = model.predict_batch_contrastive(
                batch_text, batch_sent_reports, batch_negative_type_reprots, batch_seq_scale, return_analysis=True
            )
            output = result[0]
            analysis = result[6]
            # Immediately delete unused large tensors from return tuple to free GPU memory
            # These are: fused_emb, batch_series_embeddings, batch_text_embeddings,
            # batch_sent_report_embeddings, batch_negative_type_report_embeddings
            del result
            target = batch_pred_scale.to(device).float()
            output_f = output.float()
            B = len(batch_text)
            batch_et = event_types[start_idx : start_idx + B] if has_et else []
            batch_n = article_counts[start_idx : start_idx + B] if has_et else []
            start_idx += B
            if analysis is None or "gate_softmax" not in analysis or analysis["gate_softmax"] is None:
                if do_g1:
                    sum_mse_f += float(((output_f - target) ** 2).mean().cpu())
                    sum_mae_f += float((output_f - target).abs().mean().cpu())
                    sum_smape_f += float(smape(output_f, target).cpu())
                    n_batches_g1 += 1
                continue
            s_vec_residual = analysis.get("s_vec_residual")
            ts_tokens = analysis.get("ts_tokens")
            ts_mask = analysis.get("ts_mask")
            x_cf = analysis.get("x_cf")
            # Extract gate_softmax before cleanup
            gate_softmax = analysis.get("gate_softmax")
            # Delete ALL unused tensors from analysis dict immediately to free GPU memory
            # We only need: gate_softmax, s_vec_residual, ts_tokens, ts_mask, x_cf
            # Delete everything else immediately
            keys_to_delete = ["attn_t2s", "attn_s2t", "token_align_info", "token_align_loss",
                             "t_vec", "t_vec_residual", "s_vec", "text_tokens", "text_mask",
                             "ts_inst", "last_value_alpha"]
            for key in keys_to_delete:
                if key in analysis:
                    tensor = analysis.pop(key)
                    if isinstance(tensor, torch.Tensor):
                        del tensor
            if s_vec_residual is None or ts_tokens is None or x_cf is None:
                if do_g1:
                    sum_mse_f += float(((output_f - target) ** 2).mean().cpu())
                    sum_mae_f += float((output_f - target).abs().mean().cpu())
                    sum_smape_f += float(smape(output_f, target).cpu())
                    n_batches_g1 += 1
                continue
            ts_only_pred = model.predict_ts_only_full_decoder(ts_tokens, ts_mask, s_vec_residual, x_cf)
            err_ts = ((ts_only_pred - target) ** 2).mean(dim=(1, 2))
            err_full = ((output_f - target) ** 2).mean(dim=(1, 2))
            diff = err_ts - err_full
            if TrainingConfig.GC_USE_ADAPTIVE_SCALE:
                s = torch.clamp(diff.abs().mean(), min=TrainingConfig.GC_MIN_SCALE)
            else:
                s = torch.tensor(TrainingConfig.GC_SIGMOID_SCALE, device=diff.device, dtype=diff.dtype)
            logit = torch.clamp(TrainingConfig.GC_GAIN * diff / s, -TrainingConfig.GC_LOGIT_CLAMP, TrainingConfig.GC_LOGIT_CLAMP)
            gc_score = torch.sigmoid(logit).cpu().numpy()
            # Extract gate values before cleanup
            gate_text_scalar = gate_softmax[:, 0, :].mean(dim=-1).cpu().numpy() if gate_softmax is not None else None
            err_ts_np = err_ts.cpu().numpy()
            err_full_np = err_full.cpu().numpy()
            if do_g1:
                n_batches_g1 += 1
                # Extract values before deleting tensors
                err_full_mean = float(err_full.mean().cpu())
                mae_f = float((output_f - target).abs().mean().cpu())
                smape_f = float(smape(output_f, target).cpu())
                err_ts_mean = float(err_ts.mean().cpu())
                mae_ts = float((ts_only_pred.float() - target).abs().mean().cpu())
                smape_ts = float(smape(ts_only_pred.float(), target).cpu())
                sum_mse_f += err_full_mean
                sum_mae_f += mae_f
                sum_smape_f += smape_f
                sum_mse_ts += err_ts_mean
                sum_mae_ts += mae_ts
                sum_smape_ts += smape_ts
            # Aggressive cleanup: delete analysis dict and all intermediate tensors
            del analysis
            del gate_softmax
            del s_vec_residual
            del ts_tokens
            del ts_mask
            del x_cf
            del ts_only_pred
            del err_ts
            del err_full
            del diff
            del logit
            del s
            if gate_text_scalar is None:
                continue
            if do_g2:
                for i in range(B):
                    et_list.append(batch_et[i])
                    n_list.append(batch_n[i])
                    gate_text_list.append(float(gate_text_scalar[i]))
                    err_ts_list.append(float(err_ts_np[i]))
                    err_full_list.append(float(err_full_np[i]))
                    gc_score_list.append(float(gc_score[i]))
            if do_g4:
                gate_list_g4.extend(gate_text_scalar.tolist())
            # Aggressive cleanup: delete batch tensors and clear cache
            del output
            del output_f
            del target
            del batch_seq_scale
            del batch_pred_scale
            del batch_text
            del batch_sent_reports
            del batch_negative_type_reprots
            del batch_seq
            del batch_pred
            # Clear CUDA cache more frequently to prevent memory accumulation
            if device.type == "cuda":
                if batch_idx % 3 == 0:  # Clear cache every 3 batches (more frequent)
                    torch.cuda.empty_cache()
                if batch_idx % 10 == 0:  # Synchronize and clear every 10 batches
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
    out = {}
    if do_g1 and n_batches_g1 > 0:
        n = n_batches_g1
        out["g1_metrics"] = {
            "full_mse": sum_mse_f / n,
            "full_rmse": np.sqrt(sum_mse_f / n),
            "full_mae": sum_mae_f / n,
            "full_smape": sum_smape_f / n,
            "ts_only_mse": sum_mse_ts / n,
            "ts_only_rmse": np.sqrt(sum_mse_ts / n),
            "ts_only_mae": sum_mae_ts / n,
            "ts_only_smape": sum_smape_ts / n,
        }
    if do_g2 and gate_text_list:
        out["g2_data"] = {
            "event_types": et_list,
            "article_counts": n_list,
            "gate_text": np.array(gate_text_list),
            "err_ts": np.array(err_ts_list),
            "err_full": np.array(err_full_list),
            "gc_score": np.array(gc_score_list),
        }
    if do_g4 and gate_list_g4:
        out["g4_gate_list"] = np.array(gate_list_g4)
    return out if out else None


def run_exp_g2_gate_per_event(model, device, test_loader, test_set, batch_size, seq_len, pred_len, output_dir=None, save_plots=True, data=None):
    """
    Exp-G2 Test (2): min/max/avg gate openness per event type (stats based on ARTICLES).
    If data is provided (from _collect_g2_data), uses it; otherwise runs one pass to collect.
    """
    print("\n" + "=" * 60)
    print("Exp-G2 Test (2): Gate openness per event type (min/max/avg, 6 types) [article-based stats]")
    print("=" * 60)
    if data is None:
        data = _collect_g2_data(model, device, test_loader, test_set)
    if data is None:
        print("Event type not available or no data collected. Skipping Exp-G2 Test (2).")
        return None
    gate_text_by_type = defaultdict(list)
    for i in range(len(data["event_types"])):
        et = data["event_types"][i]
        gate_text_by_type[et].append((data["gate_text"][i], data["article_counts"][i]))
    # Article-based stats: count = sum(articles), avg = sum(gate * n) / sum(n)
    stats = {}
    for et in sorted(gate_text_by_type.keys()):
        pairs = gate_text_by_type[et]
        vals = [p[0] for p in pairs]
        weights = [p[1] for p in pairs]
        total_articles = sum(weights)
        wsum = sum(g * n for g, n in pairs)
        avg_w = wsum / total_articles if total_articles > 0 else np.mean(vals) if vals else float("nan")
        stats[et] = {"min": min(vals), "max": max(vals), "avg": avg_w, "articles": total_articles}
    print("\n--- Gate (text) openness per event type [by articles] ---")
    print(f"{'EventType':<10} {'Min':<10} {'Max':<10} {'Avg':<10} {'Articles':<10}")
    print("-" * 52)
    for et in sorted(stats.keys()):
        s = stats[et]
        print(f"{et:<10} {s['min']:<10.4f} {s['max']:<10.4f} {s['avg']:<10.4f} {s['articles']:<10}")

    if output_dir and save_plots and HAS_MATPLOTLIB and gate_text_by_type:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        labels = sorted(gate_text_by_type.keys())
        data = [np.array([p[0] for p in gate_text_by_type[et]]) for et in labels]
        ax.boxplot(data, labels=labels)
        ax.set_xlabel("Event type")
        ax.set_ylabel("Gate (text) openness")
        ax.set_title("Instance-level gate distribution per event type (article-weighted stats)")
        fig.tight_layout()
        path = os.path.join(output_dir, "gate_per_event_type_boxplot.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Boxplot saved to {path}")
    if output_dir and gate_text_by_type:
        data = {
            et: {"gate_values": [float(p[0]) for p in vals], "articles": stats[et]["articles"], "min": stats[et]["min"], "max": stats[et]["max"], "avg": stats[et]["avg"]}
            for et, vals in gate_text_by_type.items()
        }
        path = os.path.join(output_dir, "gate_per_event_type_data.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Data saved to {path}")
    return stats


def run_exp_g2_single_pass(model, device, test_loader, test_set, batch_size, seq_len, pred_len, output_dir=None, output_dir_g3=None, save_plots=True, run_test2=True, run_test3=True, data=None):
    """
    Report all G2 result tables (Test 2: gate per type + per-type ΔMSE -> output_dir; Test 3: gate–utility -> output_dir_g3).
    If data is provided, uses it; otherwise collects once.
    """
    if data is None:
        data = _collect_g2_data(model, device, test_loader, test_set)
    if data is None:
        print("G2: No data collected (test_set may lack event types). Skipping all G2 reports.")
        return
    dir_g3 = output_dir_g3 if output_dir_g3 is not None else output_dir
    if run_test2:
        run_exp_g2_gate_per_event(model, device, test_loader, test_set, batch_size, seq_len, pred_len, output_dir=output_dir, save_plots=save_plots, data=data)
        run_exp_g2_per_type_delta_mse(model, device, test_loader, test_set, batch_size, seq_len, pred_len, output_dir=output_dir, save_plots=save_plots, data=data)
    if run_test3:
        run_exp_g2_gate_utility(model, device, test_loader, seq_len, pred_len, dir_g3, save_plots=save_plots, data=data)


def run_exp_g2_per_type_delta_mse(model, device, test_loader, test_set, batch_size, seq_len, pred_len, output_dir=None, save_plots=True, data=None):
    """
    Per-event-type ΔMSE and avg gate (article-based stats). If data is provided, uses it; else collects once.
    """
    print("\n" + "=" * 60)
    print("Exp-G2 Test (2) extended: Per-event-type ΔMSE and gate openness [article-based stats]")
    print("=" * 60)
    if data is None:
        data = _collect_g2_data(model, device, test_loader, test_set)
    if data is None:
        print("Event type not available or no data. Skipping per-type ΔMSE.")
        return None
    mse_ts_sum = defaultdict(float)
    mse_full_sum = defaultdict(float)
    gate_sum = defaultdict(float)
    articles_sum = defaultdict(int)
    for i in range(len(data["event_types"])):
        et = data["event_types"][i]
        n = data["article_counts"][i]
        mse_ts_sum[et] += data["err_ts"][i] * n
        mse_full_sum[et] += data["err_full"][i] * n
        articles_sum[et] += n
        gate_sum[et] += data["gate_text"][i] * n
    type_order = sorted(mse_ts_sum.keys())
    if not type_order:
        print("No per-type data collected.")
        return None
    print("\n--- Per-event-type: Avg gate openness and ΔMSE (ts_only − full) [by articles] ---")
    print(f"{'EventType':<10} {'Avg gate':<10} {'MSE_ts_only':<12} {'MSE_full':<12} {'ΔMSE':<12} {'Articles':<10}")
    print("-" * 68)
    out_rows = []
    for et in type_order:
        n = articles_sum[et]
        avg_gate = gate_sum[et] / n if n > 0 and gate_sum[et] != 0 else float("nan")
        mse_ts = mse_ts_sum[et] / n if n > 0 else float("nan")
        mse_full = mse_full_sum[et] / n if n > 0 else float("nan")
        delta_mse = mse_ts - mse_full if n > 0 else float("nan")
        out_rows.append({"event_type": et, "avg_gate": avg_gate, "mse_ts": mse_ts, "mse_full": mse_full, "delta_mse": delta_mse, "articles": n})
        print(f"{et:<10} {avg_gate:<10.4f} {mse_ts:<12.6f} {mse_full:<12.6f} {delta_mse:<12.6f} {n:<10}")

    if output_dir and save_plots and HAS_MATPLOTLIB and out_rows:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
        x = np.arange(len(out_rows))
        width = 0.35
        labels = [r["event_type"] for r in out_rows]
        avg_gates = [r["avg_gate"] if not np.isnan(r["avg_gate"]) else 0 for r in out_rows]
        delta_mses = [r["delta_mse"] for r in out_rows]
        ax1.bar(x - width / 2, avg_gates, width, label="Avg gate openness", color="steelblue", alpha=0.8)
        ax1.set_xlabel("Event type")
        ax1.set_ylabel("Avg gate (text) openness", color="steelblue")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_ylim(0, 1.05)
        ax1.tick_params(axis="y", labelcolor="steelblue")
        ax2 = ax1.twinx()
        ax2.bar(x + width / 2, delta_mses, width, label=r"$\Delta$MSE (ts_only − full)", color="coral", alpha=0.8)
        ax2.set_ylabel(r"$\Delta$MSE (ts_only − full)", color="coral")
        ax2.tick_params(axis="y", labelcolor="coral")
        fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        fig.suptitle("Per-event-type: gate openness vs ΔMSE (Granger story; article-based stats)")
        fig.tight_layout()
        path = os.path.join(output_dir, "gate_vs_delta_mse_per_event_type.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Bar chart saved to {path}")
    if output_dir and out_rows:
        def _to_json_val(v):
            if isinstance(v, (int, np.integer)):
                return int(v)
            if isinstance(v, (float, np.floating)):
                return float(v) if not np.isnan(v) else None
            return v
        data = [{k: _to_json_val(v) for k, v in r.items()} for r in out_rows]
        path = os.path.join(output_dir, "gate_vs_delta_mse_per_event_type_data.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Data saved to {path}")
    return out_rows


def run_exp_g2_gate_utility(model, device, test_loader, seq_len, pred_len, output_dir, save_plots=True, data=None):
    """
    Exp-G2 Test (3): Gate–utility followness (correlation + bucket plot).
    If data is provided (from _collect_g2_data), uses it; otherwise runs one pass to collect.
    """
    print("\n" + "=" * 60)
    print("Exp-G2 Test (3): Gate–utility followness (correlation + bucket plot)")
    print("=" * 60)
    if data is None and hasattr(test_loader, "dataset"):
        data = _collect_g2_data(model, device, test_loader, test_loader.dataset)
    if data is None:
        print("No data for gate–utility (need test_set with event types). Skipping Exp-G2 Test (3).")
        return None
    gate_all = data["gate_text"]
    gc_all = data["gc_score"]
    delta_mse_all = data["err_ts"] - data["err_full"]
    corr_pearson = np.corrcoef(gate_all, gc_all)[0, 1] if len(gate_all) > 1 else float("nan")
    corr_spearman = float("nan")
    if HAS_SCIPY and len(gate_all) > 1:
        try:
            rho, _ = spearmanr(gate_all, gc_all)
            corr_spearman = float(rho)
        except Exception:
            pass
    print(f"\nCorrelation(gate_text, gc_score): Pearson {corr_pearson:.4f}" + (f"  Spearman {corr_spearman:.4f}" if not np.isnan(corr_spearman) else ""))
    print(f"  Samples: {len(gate_all)}")

    # Bucket: bin by gate_text, mean gc_score and mean ΔMSE per bin
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_means = []
    bin_means_delta_mse = []
    bin_centers = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (gate_all >= lo) & (gate_all < hi) if i < n_bins - 1 else (gate_all >= lo) & (gate_all <= hi)
        if mask.sum() > 0:
            bin_means.append(gc_all[mask].mean())
            bin_means_delta_mse.append(delta_mse_all[mask].mean())
            bin_centers.append((lo + hi) / 2)
        else:
            bin_means.append(np.nan)
            bin_means_delta_mse.append(np.nan)
            bin_centers.append((lo + hi) / 2)
    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)
    bin_means_delta_mse = np.array(bin_means_delta_mse)

    print("\n--- Bucket summary (gate_text bin -> mean gc_score, mean ΔMSE) ---")
    for i in range(n_bins):
        dm = bin_means_delta_mse[i] if not np.isnan(bin_means_delta_mse[i]) else float("nan")
        print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): gc_score={bin_means[i]:.4f}  mean_ΔMSE={dm:.6f}")
    if output_dir and save_plots and HAS_MATPLOTLIB:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.bar(bin_centers, bin_means, width=0.08, align="center", edgecolor="gray", alpha=0.8, label="mean gc_score")
        ax.set_xlabel("Gate (text) openness")
        ax.set_ylabel("Mean utility (gc_score)")
        title = f"Gate–utility followness (Pearson={corr_pearson:.3f}"
        if not np.isnan(corr_spearman):
            title += f", Spearman={corr_spearman:.3f}"
        title += ")"
        ax.set_title(title)
        fig.tight_layout()
        path = os.path.join(output_dir, "gate_utility_bucket_plot.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Bucket plot saved to {path}")
    if output_dir:
        data = {
            "gate_text": gate_all.tolist(),
            "gc_score": gc_all.tolist(),
            "delta_mse": delta_mse_all.tolist(),
            "correlation_pearson": float(corr_pearson),
            "correlation_spearman": float(corr_spearman),
            "n_samples": int(len(gate_all)),
            "bin_centers": bin_centers.tolist(),
            "bin_means_gc_score": bin_means.tolist(),
            "bin_means_delta_mse": [float(x) if not np.isnan(x) else None for x in bin_means_delta_mse],
        }
        path = os.path.join(output_dir, "gate_utility_data.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Data saved to {path}")
    return {"correlation_pearson": corr_pearson, "correlation_spearman": corr_spearman, "n_samples": len(gate_all), "bin_means": bin_means, "bin_means_delta_mse": bin_means_delta_mse, "bin_centers": bin_centers}


def _resolve_checkpoint_path(series_id, checkpoint_map, checkpoint_dir, default_checkpoint):
    """
    Resolve checkpoint path for a series_id.
    checkpoint_map: "SP500:path1,NASDAQ:path2,..." or None
    checkpoint_dir: base dir, glob *_{series_id}_*/best_model.pth or None
    Prefers CAMEF4P19L (MOMENT) checkpoints since test_gate_openness uses CAMEF4P19L.
    """
    if checkpoint_map:
        for pair in checkpoint_map.split(","):
            pair = pair.strip()
            if ":" in pair:
                k, v = pair.split(":", 1)
                if k.strip().upper() == series_id.upper():
                    return v.strip()
    if checkpoint_dir:
        for sid in (series_id, series_id.upper(), series_id.lower()):
            pattern = os.path.join(checkpoint_dir, f"*_{sid}_*", "best_model.pth")
            matches = glob.glob(pattern)
            if matches:
                # Prefer MOMENT/CAMEF4P19L checkpoints; test_gate_openness uses CAMEF4P19L
                moment_matches = [m for m in matches if "moment" in m.lower() and "camef4p19l" in m.lower()]
                return moment_matches[0] if moment_matches else matches[0]
    return default_checkpoint


def run_exp_g4_per_asset_gate_boxplot(
    series_ids,
    checkpoint_map,
    checkpoint_dir,
    default_checkpoint,
    device,
    seq_len,
    pred_len,
    batch_size,
    event_dir,
    series_dir,
    event_id,
    output_dir,
    moment_model,
    llama_name,
    window,
    stride,
    max_token_num,
    save_plots=True,
):
    """
    Exp-G4: Instance-level per-asset gate distribution (boxplots).
    Shows that gate varies at instance level; e.g., USGG1M has more open gate than equities.
    Each asset uses its own checkpoint (per-asset trained model).
    """
    print("\n" + "=" * 60)
    print("Exp-G4: Per-asset gate distribution (instance-level boxplots)")
    print("=" * 60)

    gate_by_asset = {}
    for series_id in series_ids:
        ckpt = _resolve_checkpoint_path(series_id, checkpoint_map, checkpoint_dir, default_checkpoint)
        if not ckpt or not os.path.isfile(ckpt):
            print(f"[{series_id}] Checkpoint not found, skipping.")
            continue
        ev = EventSet(
            seq_len, pred_len,
            event_id=event_id,
            series_id=series_id,
            shuffle=False,
            batch_size=batch_size,
            scale=True,
            event_dir=event_dir,
            series_dir=series_dir,
        )
        d = ev.d
        model = CAMEF(
            llama_name=llama_name,
            moment=moment_model,
            seq_len=seq_len,
            pred_len=pred_len,
            d=d,
            window=window,
            stride=stride,
            batch_size=batch_size,
            decoder_layers=3,
            decoder_heads=8,
            use_ts_memory=False,
            max_token_num=max_token_num,
        )
        model.load_model_combined(ckpt, strict=False)
        model = model.to(device)
        model.eval()
        # Clear CUDA cache immediately after model loading
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        gate_values = []
        with torch.no_grad():
            for batch in tqdm(ev.test_loader, desc=f"Gate {series_id}", ncols=100):
                batch_text, batch_sent_reports, batch_negative_type_reprots, batch_seq, batch_pred, batch_seq_scale, batch_pred_scale = batch[:7]
                batch_sent_reports = list(map(list, zip(*batch_sent_reports)))
                batch_negative_type_reprots = list(map(list, zip(*batch_negative_type_reprots)))
                (output, _, _, _, _, _, analysis) = model.predict_batch_contrastive(
                    batch_text, batch_sent_reports, batch_negative_type_reprots, batch_seq_scale, return_analysis=True
                )
                if analysis is not None and "gate_softmax" in analysis and analysis["gate_softmax"] is not None:
                    g = analysis["gate_softmax"].detach().cpu()
                    gate_text_scalar = g[:, 0, :].mean(dim=-1)
                    gate_values.extend(gate_text_scalar.tolist())
        if gate_values:
            gate_by_asset[series_id] = np.array(gate_values)
            print(f"[{series_id}] {len(gate_values)} samples, gate mean={np.mean(gate_values):.4f}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not gate_by_asset:
        print("No gate data collected. Skipping Exp-G4.")
        return None

    run_exp_g4_report_only(gate_by_asset, output_dir, save_plots)
    return gate_by_asset


def run_exp_g4_report_only(gate_by_asset, output_dir, save_plots=True):
    """Write G4 JSON and boxplot from pre-collected gate_by_asset {series_id: gate_array}."""
    if not gate_by_asset:
        return
    if output_dir:
        data = {asset: values.tolist() if hasattr(values, "tolist") else values for asset, values in gate_by_asset.items()}
        path = os.path.join(output_dir, "gate_per_asset_data.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Data saved to {path}")

    if output_dir and save_plots and HAS_MATPLOTLIB:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        labels = list(gate_by_asset.keys())
        data = [gate_by_asset[k] for k in labels]
        ax.boxplot(data, labels=labels)
        ax.set_xlabel("Asset")
        ax.set_ylabel("Gate (text) openness")
        ax.set_title("Instance-level gate distribution per asset")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        path = os.path.join(output_dir, "gate_per_asset_boxplot.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Boxplot saved to {path}")


def main():
    p = argparse.ArgumentParser(description="Gate openness experiments for CAMEF4P19L")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (required for G1/G2/G3; fallback for G4)")
    p.add_argument("--series_ids", type=str, nargs="+", default=["SP500", "NASDAQ"], help="Series for G1 table")
    p.add_argument("--seq_len", type=int, default=35)
    p.add_argument("--pred_len", type=int, default=140)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--event_id", type=int, default=0)
    p.add_argument("--event_dir", type=str, default="data/event")
    p.add_argument("--series_dir", type=str, default="data/series")
    p.add_argument("--output_dir", type=str, default=None, help="Directory for JSON data (and optional plots)")
    p.add_argument("--g1", action="store_true", help="Run only Exp-G1 (TS-only vs Full table)")
    p.add_argument("--g2", action="store_true", help="Run only Exp-G2 (gate per event type + gate–utility)")
    p.add_argument("--g3", action="store_true", help="Run only Exp-G2 Test (3) gate–utility")
    p.add_argument("--g4", action="store_true", help="Run only Exp-G4 (per-asset gate boxplot)")
    p.add_argument("--checkpoint_map", type=str, default=None, help="SP500:path1,NASDAQ:path2,... for per-asset checkpoints")
    p.add_argument("--checkpoint_dir", type=str, default=None, help="Base dir to glob *_{series_id}_*/best_model.pth")
    p.add_argument("--data_only", action="store_true", help="Only output JSON data; do not save PNG plots")
    # Model build (must match training)
    p.add_argument("--moment_model", type=str, default="/home/yang/Research/CAMEF/baselines/moment/MOMETN-1-large/")
    p.add_argument("--llama_name", type=str, default="/home/yang/Research/CAMEF/llama3-2-3B/")
    p.add_argument("--window", type=int, default=500)
    p.add_argument("--stride", type=int, default=400)
    p.add_argument("--max_token_num", type=int, default=1024)
    args = p.parse_args()

    run_all = not (args.g1 or args.g2 or args.g3 or args.g4)
    do_g1 = run_all or args.g1
    do_g2_test2 = run_all or args.g2
    do_g2_test3 = run_all or args.g2 or args.g3
    do_g4 = run_all or args.g4

    per_asset_g4 = bool(args.checkpoint_map or args.checkpoint_dir)
    need_main_model = do_g1 or do_g2_test2 or do_g2_test3 or (do_g4 and not per_asset_g4)
    if need_main_model and not args.checkpoint:
        raise SystemExit("--checkpoint is required for G1/G2/G3")
    if do_g4 and not (args.checkpoint_map or args.checkpoint_dir or args.checkpoint):
        raise SystemExit("For G4, provide --checkpoint_map, --checkpoint_dir, or --checkpoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set CUDA memory allocation strategy to reduce initial memory spike
    if device.type == "cuda":
        # Use memory-efficient allocation (PyTorch 1.10+)
        # This helps prevent CUDA from reserving too much memory upfront
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # Clear any existing cache before starting
        torch.cuda.empty_cache()

    model = None
    ev = None

    save_plots = not args.data_only

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        g1_dir = os.path.join(args.output_dir, "g1")
        g2_dir = os.path.join(args.output_dir, "g2")
        g3_dir = os.path.join(args.output_dir, "g3")
        g4_dir = os.path.join(args.output_dir, "g4")
        for d in (g1_dir, g2_dir, g3_dir, g4_dir):
            os.makedirs(d, exist_ok=True)
    else:
        g1_dir = g2_dir = g3_dir = g4_dir = None

    if need_main_model:
        g1_rows = []
        g2_data = None
        g2_test_loader = None
        g2_test_set = None
        gate_by_asset_unified = {}
        for series_id in args.series_ids:
            ev = EventSet(
                args.seq_len, args.pred_len,
                event_id=args.event_id,
                series_id=series_id,
                shuffle=False,
                batch_size=args.batch_size,
                scale=True,
                event_dir=args.event_dir,
                series_dir=args.series_dir,
            )
            if model is None:
                d = ev.d
                model = CAMEF(
                    llama_name=args.llama_name,
                    moment=args.moment_model,
                    seq_len=args.seq_len,
                    pred_len=args.pred_len,
                    d=d,
                    window=args.window,
                    stride=args.stride,
                    batch_size=args.batch_size,
                    decoder_layers=3,
                    decoder_heads=8,
                    use_ts_memory=False,
                    max_token_num=args.max_token_num,
                )
                model.load_model_combined(args.checkpoint, strict=False)
                model = model.to(device)
                model.eval()
                # Clear CUDA cache immediately after model loading to free reserved memory
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # Set memory fraction to prevent over-allocation (optional, can be removed if causes issues)
                    # torch.cuda.set_per_process_memory_fraction(0.9)
            test_loader = ev.test_loader
            test_set = ev.test_set
            is_first_series = (series_id == args.series_ids[0])
            # Clear cache before starting data collection to ensure clean state
            if device.type == "cuda":
                torch.cuda.empty_cache()
            data = _collect_all_g_data(
                model, device, test_loader, test_set,
                do_g1=do_g1,
                do_g2=(do_g2_test2 or do_g2_test3) and is_first_series,
                do_g4=do_g4 and not per_asset_g4,
            )
            if data and data.get("g1_metrics"):
                m = data["g1_metrics"]
                g1_rows.append({
                    "series_id": series_id,
                    "ts_only_rmse": m["ts_only_rmse"], "ts_only_mse": m["ts_only_mse"],
                    "ts_only_mae": m["ts_only_mae"], "ts_only_smape": m["ts_only_smape"],
                    "full_rmse": m["full_rmse"], "full_mse": m["full_mse"],
                    "full_mae": m["full_mae"], "full_smape": m["full_smape"],
                })
            if data and data.get("g2_data") and is_first_series:
                g2_data = data["g2_data"]
                g2_test_loader = test_loader
                g2_test_set = test_set
            if data and data.get("g4_gate_list") is not None:
                gate_by_asset_unified[series_id] = data["g4_gate_list"]
            # Cleanup EventSet and test_loader to free memory
            del ev
            del test_loader
            if device.type == "cuda":
                torch.cuda.empty_cache()
            import gc
            gc.collect()

        if g1_rows:
            run_exp_g1_print(g1_rows)
            if g1_dir:
                path = os.path.join(g1_dir, "ts_only_vs_full_table.json")
                with open(path, "w") as f:
                    json.dump(g1_rows, f, indent=2)
                print(f"G1 table saved to {path}")
        if g2_data:
            run_exp_g2_single_pass(
                model, device, g2_test_loader, g2_test_set,
                args.batch_size, args.seq_len, args.pred_len,
                output_dir=g2_dir,
                output_dir_g3=g3_dir,
                save_plots=save_plots,
                run_test2=do_g2_test2,
                run_test3=do_g2_test3,
                data=g2_data,
            )
        if do_g4 and not per_asset_g4 and gate_by_asset_unified:
            run_exp_g4_report_only(gate_by_asset_unified, g4_dir, save_plots)

    if do_g4 and per_asset_g4:
        run_exp_g4_per_asset_gate_boxplot(
            series_ids=args.series_ids,
            checkpoint_map=args.checkpoint_map,
            checkpoint_dir=args.checkpoint_dir,
            default_checkpoint=args.checkpoint,
            device=device,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            batch_size=args.batch_size,
            event_dir=args.event_dir,
            series_dir=args.series_dir,
            event_id=args.event_id,
            output_dir=g4_dir,
            moment_model=args.moment_model,
            llama_name=args.llama_name,
            window=args.window,
            stride=args.stride,
            max_token_num=args.max_token_num,
            save_plots=save_plots,
        )

    print("\nGate openness experiments done.")


if __name__ == "__main__":
    main()
