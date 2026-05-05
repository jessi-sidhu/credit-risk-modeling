import numpy as np

from src.evaluate import (
    CostMatrix,
    cost_optimal_threshold,
    cost_sensitivity,
    evaluate_at_threshold,
    expected_cost,
    threshold_sweep,
)


def test_pr_auc_of_perfect_classifier_is_one():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    out = evaluate_at_threshold(y_true, y_prob, 0.5)
    assert out["pr_auc"] == 1.0
    assert out["roc_auc"] == 1.0
    assert out["precision"] == 1.0 and out["recall"] == 1.0


def test_expected_cost_uses_5_to_1_default():
    """One FN and one FP under the default cost matrix => 5 + 1 = 6."""
    y_true = np.array([1, 0])
    y_prob = np.array([0.1, 0.9])  # FN on idx 0, FP on idx 1, threshold 0.5
    assert expected_cost(y_true, y_prob, 0.5) == 6.0


def test_cost_optimal_threshold_matches_brute_force():
    """The function-returned threshold must produce the minimum cost on
    the same sweep grid (no off-by-one between sweep and selection)."""
    rng = np.random.default_rng(0)
    n = 2_000
    y_true = rng.binomial(1, 0.18, size=n)
    # noisy but informative scores
    y_prob = np.clip(0.18 + 0.4 * y_true + rng.normal(0, 0.25, n), 1e-4, 1 - 1e-4)
    cost = CostMatrix()

    sweep = threshold_sweep(y_true, y_prob, cost)
    brute_min = float(sweep.loc[sweep["expected_cost"].idxmin(), "threshold"])
    chosen = cost_optimal_threshold(y_true, y_prob, cost)
    assert chosen == brute_min


def test_cost_optimal_threshold_below_half_when_fn_is_expensive():
    """Heavy FN penalty should push the threshold below 0.5 (catch more
    positives at the cost of more FPs)."""
    rng = np.random.default_rng(1)
    n = 5_000
    y_true = rng.binomial(1, 0.18, size=n)
    y_prob = np.clip(0.18 + 0.35 * y_true + rng.normal(0, 0.3, n), 1e-4, 1 - 1e-4)
    cost = CostMatrix(fn_cost=20.0, fp_cost=1.0)
    thr = cost_optimal_threshold(y_true, y_prob, cost)
    assert thr < 0.5


def test_cost_sensitivity_threshold_decreases_with_fn_cost():
    """Higher FN penalty => lower optimal threshold (catch more positives)."""
    rng = np.random.default_rng(2)
    n = 5_000
    y_true = rng.binomial(1, 0.18, size=n)
    y_prob = np.clip(0.18 + 0.35 * y_true + rng.normal(0, 0.3, n), 1e-4, 1 - 1e-4)
    out = cost_sensitivity(y_true, y_prob, ratios=(1.0, 3.0, 5.0, 10.0))
    thresholds = out["optimal_threshold"].to_numpy()
    # monotonically non-increasing as FN penalty grows
    assert (np.diff(thresholds) <= 1e-9).all(), thresholds


def test_threshold_sweep_shape_and_columns():
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.4, 0.2, 0.6, 0.7, 0.3, 0.5, 0.8])
    sweep = threshold_sweep(y_true, y_prob)
    assert set(sweep.columns) == {"threshold", "precision", "recall", "f1", "expected_cost"}
    assert len(sweep) == 91
    assert sweep["threshold"].is_monotonic_increasing
