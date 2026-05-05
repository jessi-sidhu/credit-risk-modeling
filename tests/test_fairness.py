import numpy as np
import pandas as pd

from src.fairness import four_fifths_rule, group_metrics, parity_ratios


def test_group_metrics_columns_and_shape():
    rng = np.random.default_rng(0)
    n = 1_000
    y_true = rng.binomial(1, 0.2, size=n)
    y_prob = rng.uniform(0, 1, size=n)
    group = rng.choice(["A", "B"], size=n)

    out = group_metrics(y_true, y_prob, group, threshold=0.5)
    assert set(out.columns) >= {
        "n", "base_rate", "approval_rate", "tpr", "fpr", "fnr", "ppv",
        "tp", "fp", "fn", "tn",
    }
    assert sorted(out.index.tolist()) == ["A", "B"]
    assert (out["n"] > 0).all()


def test_parity_ratios_unity_when_predictions_ignore_group():
    """If group is independent of (X, y) and predictions are random,
    parity ratios should hover near 1.0 with enough samples."""
    rng = np.random.default_rng(1)
    n = 60_000
    y_true = rng.binomial(1, 0.18, size=n)
    y_prob = rng.uniform(0, 1, size=n)
    group = rng.choice(["A", "B", "C"], size=n)

    m = group_metrics(y_true, y_prob, group, threshold=0.5)
    r = parity_ratios(m)
    # ignore the reference row's self-ratio of 1; tolerance accounts for sampling noise
    for col in ("approval_rate_ratio", "tpr_ratio", "fpr_ratio"):
        non_ref = r[col][r[col].index != r["reference_group"].iloc[0]]
        assert (non_ref - 1).abs().max() < 0.05, f"{col} not near 1: {non_ref.to_dict()}"


def test_four_fifths_passes_when_approvals_equal():
    metrics = pd.DataFrame({
        "approval_rate": [0.85, 0.84, 0.86],
        "tpr": [0.5, 0.5, 0.5], "fpr": [0.1, 0.1, 0.1],
        "fnr": [0.5, 0.5, 0.5], "ppv": [0.5, 0.5, 0.5],
        "n": [100, 100, 100],
    }, index=["A", "B", "C"])
    out = four_fifths_rule(metrics)
    assert out["passes"] is True
    assert out["ratio"] >= 0.97


def test_four_fifths_fails_on_skewed_approval():
    metrics = pd.DataFrame({
        "approval_rate": [0.95, 0.40, 0.85],
        "tpr": [0.5, 0.5, 0.5], "fpr": [0.1, 0.1, 0.1],
        "fnr": [0.5, 0.5, 0.5], "ppv": [0.5, 0.5, 0.5],
        "n": [100, 100, 100],
    }, index=["A", "B", "C"])
    out = four_fifths_rule(metrics)
    assert out["passes"] is False
    assert out["min_group"] == "B"
    assert out["max_group"] == "A"
    assert out["ratio"] < 0.5
