import numpy as np
import pytest

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


# --- KS / Brier / headline / bootstrap ---------------------------------------

from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluate import (
    _sorted_for_bootstrap,
    _weighted_auc_ap,
    bootstrap_ci,
    brier_score,
    headline_metrics,
    ks_statistic,
)


def _scores(n=3_000, seed=0, round_to=None):
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.15, size=n)
    p = np.clip(0.15 + 0.3 * y + rng.normal(0, 0.2, n), 1e-4, 1 - 1e-4)
    if round_to is not None:          # force many tied scores
        p = np.round(p, round_to)
    return y, p


def test_ks_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert ks_statistic(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.0


def test_brier_known_value():
    assert brier_score([1, 0], [0.8, 0.4]) == pytest.approx((0.04 + 0.16) / 2)


def test_headline_metrics_gini_is_2auc_minus_1():
    y, p = _scores()
    m = headline_metrics(y, p)
    assert set(m) == {"roc_auc", "gini", "ks", "pr_auc", "brier", "ece"}
    assert m["gini"] == pytest.approx(2 * m["roc_auc"] - 1)


@pytest.mark.parametrize("round_to", [None, 2])
def test_weighted_auc_ap_matches_sklearn_with_unit_weights(round_to):
    y, p = _scores(round_to=round_to)
    order, y_s, ends = _sorted_for_bootstrap(y, p)
    auc, ap = _weighted_auc_ap(y_s, np.ones(len(y)), ends)
    assert auc == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    assert ap == pytest.approx(average_precision_score(y, p), abs=1e-12)


def test_weighted_auc_ap_matches_sklearn_on_a_resample():
    y, p = _scores(round_to=2)
    idx = np.random.default_rng(3).integers(0, len(y), len(y))
    w = np.bincount(idx, minlength=len(y)).astype(float)
    order, y_s, ends = _sorted_for_bootstrap(y, p)
    auc, ap = _weighted_auc_ap(y_s, w[order], ends)
    assert auc == pytest.approx(roc_auc_score(y[idx], p[idx]), abs=1e-12)
    assert ap == pytest.approx(average_precision_score(y[idx], p[idx]), abs=1e-12)


def test_bootstrap_ci_brackets_estimate_and_diff_is_paired():
    y, p_good = _scores(seed=1)
    rng = np.random.default_rng(2)
    p_weak = np.clip(p_good + rng.normal(0, 0.25, len(y)), 1e-4, 1 - 1e-4)
    ci = bootstrap_ci(y, {"good": p_good, "weak": p_weak}, diff=("good", "weak"), n_boot=200)
    for key in ("good", "weak", "good - weak"):
        for metric in ("roc_auc", "pr_auc"):
            c = ci[key][metric]
            assert c["ci_low"] <= c["estimate"] <= c["ci_high"]
    d = ci["good - weak"]["roc_auc"]
    assert d["estimate"] == pytest.approx(
        ci["good"]["roc_auc"]["estimate"] - ci["weak"]["roc_auc"]["estimate"])
    assert d["ci_low"] > 0      # clearly better model -> CI excludes zero
