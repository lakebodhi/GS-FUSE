import math

import numpy as np

from test_pretrained_model import (
    compute_endpoint_binary_metrics,
    compute_forecasting_metrics,
    compute_horizon_metrics,
    compute_financial_strategy_metrics,
)


def test_compute_forecasting_metrics_matches_original_mse_mae_style():
    preds = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    targets = np.array([[[2.0, 0.0], [1.0, 4.0]]])

    metrics = compute_forecasting_metrics(preds, targets)

    assert metrics["mse"] == 2.25
    assert metrics["mae"] == 1.25


def test_compute_endpoint_binary_metrics_uses_endpoint_signal_only():
    preds = np.array(
        [
            [[11.0, 12.0]],
            [[9.5, 8.0]],
            [[10.0, 10.0]],
        ]
    )
    targets = np.array(
        [
            [[10.5, 13.0]],
            [[9.5, 7.0]],
            [[9.0, 8.0]],
        ]
    )
    seqs = np.array(
        [
            [[9.0, 10.0]],
            [[11.0, 10.0]],
            [[9.5, 10.0]],
        ]
    )

    metrics = compute_endpoint_binary_metrics(
        preds,
        targets,
        seqs,
        feature_idx=0,
        annualize_factor=4.0,
        risk_free_rate=0.0,
    )

    assert metrics["num_events"] == 3
    assert metrics["num_traded"] == 2
    assert metrics["directional_hit_rate"] == 1.0
    assert metrics["endpoint_pred_move_mean"] == 0.0
    assert metrics["endpoint_actual_move_mean"] == -2.0 / 3.0
    assert math.isfinite(metrics["sharpe_ratio"])


def test_compute_horizon_metrics_reports_requested_sub_horizons_and_full():
    preds = np.array([[[1.0, 2.0, 3.0, 4.0]]])
    targets = np.array([[[1.0, 0.0, 5.0, 8.0]]])
    seqs = np.array([[[0.0, 1.0]]])

    metrics = compute_horizon_metrics(
        preds,
        targets,
        seqs,
        horizons=(2, 3),
        feature_idx=0,
        annualize_factor=4.0,
        risk_free_rate=0.0,
    )

    assert set(metrics) == {"H2", "H3", "full"}
    assert metrics["H2"]["mse"] == 2.0
    assert metrics["H2"]["mae"] == 1.0
    assert metrics["H3"]["mse"] == 8.0 / 3.0
    assert metrics["full"]["mse"] == 6.0


def test_compute_financial_strategy_metrics_reports_four_financial_variants():
    preds = np.array(
        [
            [[11.0, 12.0, 13.0, 14.0]],
            [[9.0, 8.0, 7.0, 6.0]],
            [[11.0, 9.0, 12.0, 8.0]],
        ]
    )
    targets = np.array(
        [
            [[11.0, 12.0, 13.0, 14.0]],
            [[9.0, 8.0, 7.0, 6.0]],
            [[9.0, 11.0, 8.0, 12.0]],
        ]
    )
    seqs = np.array(
        [
            [[9.0, 10.0]],
            [[11.0, 10.0]],
            [[9.5, 10.0]],
        ]
    )

    metrics = compute_financial_strategy_metrics(
        preds,
        targets,
        seqs,
        feature_idx=0,
        annualize_factor=4.0,
        risk_free_rate=0.0,
    )

    expected_keys = {
        "directional_hit_rate",
        "sharpe_ratio",
        "ep_directional_hit_rate",
        "ep_sharpe_ratio",
        "mw_dhr_weighted",
        "mw_sharpe_ratio",
        "cf_pct_traded",
        "cf_directional_hit_rate",
        "cf_sharpe_ratio",
    }
    assert expected_keys <= set(metrics)
    assert metrics["ep_directional_hit_rate"] == 2.0 / 3.0
    assert 0.0 <= metrics["mw_dhr_weighted"] <= 1.0
