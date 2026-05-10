import numpy as np
import pandas as pd
import pytest

from src.stacking import fit_meta, predict_meta, stack_weights


def _two_correlated_probs(n: int, seed: int):
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.2, size=n)
    p_a = np.clip(y * 0.5 + 0.2 + rng.normal(0, 0.15, n), 0.01, 0.99)
    p_b = np.clip(y * 0.6 + 0.15 + rng.normal(0, 0.20, n), 0.01, 0.99)
    return y, {"a": p_a, "b": p_b}


def test_fit_and_predict_returns_unit_interval():
    y, probs = _two_correlated_probs(2_000, seed=0)
    meta, names = fit_meta(probs, y)
    out = predict_meta(meta, probs, names)
    assert out.shape == (2_000,)
    assert (out >= 0).all() and (out <= 1).all()


def test_names_in_alphabetical_order():
    y, probs = _two_correlated_probs(500, seed=1)
    meta, names = fit_meta(probs, y)
    assert names == sorted(probs.keys())


def test_predict_raises_when_base_model_missing():
    y, probs = _two_correlated_probs(500, seed=2)
    meta, names = fit_meta(probs, y)
    with pytest.raises(KeyError):
        predict_meta(meta, {"a": probs["a"]}, names)


def test_weights_have_expected_shape():
    y, probs = _two_correlated_probs(500, seed=3)
    meta, names = fit_meta(probs, y)
    w = stack_weights(meta, names)
    assert isinstance(w, pd.Series)
    assert sorted(w.index.tolist()) == names


def test_length_mismatch_raises():
    y, probs = _two_correlated_probs(500, seed=4)
    bad = {"a": probs["a"][:100], "b": probs["b"]}
    with pytest.raises(ValueError):
        fit_meta(bad, y)


def test_stack_beats_or_matches_each_base_on_pr_auc():
    """A reasonable sanity check: the stacked predictor on its training
    set should be at least as good as each individual base model on
    that same set. (This is a weaker claim than 'stack wins on
    held-out', which doesn't always hold for correlated bases, but
    in-sample dominance should always hold.)"""
    from sklearn.metrics import average_precision_score
    y, probs = _two_correlated_probs(3_000, seed=5)
    meta, names = fit_meta(probs, y)
    stacked = predict_meta(meta, probs, names)
    pr_stack = average_precision_score(y, stacked)
    for name in names:
        pr_base = average_precision_score(y, probs[name])
        assert pr_stack >= pr_base - 1e-6, f"stack {pr_stack:.4f} < base {name} {pr_base:.4f}"
