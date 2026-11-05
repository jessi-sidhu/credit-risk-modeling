"""Out-of-time training path, run on the synthetic fixture (2014-2018)."""
import numpy as np
import pandas as pd
import pytest

from src.features import engineer_features
from src.synth import generate
from src.train import fit_and_evaluate_oot, temporal_xy


@pytest.fixture(scope="module")
def oot():
    df = generate(n=3_000, seed=42)
    return fit_and_evaluate_oot(df=df, train_until="2016-12", test_from="2018-01", n_iter=2)


def test_temporal_xy_is_ordered_and_disjoint():
    df = engineer_features(generate(n=2_000, seed=0))
    splits = temporal_xy(df, train_until="2016-12", test_from="2018-01")
    tr, va, te = (splits[k][0]["issue_d"] for k in ("train", "val", "test"))
    assert tr.max() < va.min() and va.max() < te.min()
    assert sum(len(splits[k][1]) for k in splits) == len(df)


def test_temporal_xy_raises_on_empty_split():
    df = engineer_features(generate(n=500, seed=0))
    with pytest.raises(ValueError, match="test split is empty"):
        temporal_xy(df, train_until="2016-12", test_from="2030-01")


def test_oot_reports_both_models_on_test(oot):
    results, splits = oot
    n_test = len(splits["test"][1])
    for name in ("logistic_regression", "lightgbm_tuned"):
        r = results[name]
        assert len(r["y_prob"]) == n_test
        assert 0.0 < r["metrics_optimal"]["roc_auc"] <= 1.0
    assert results["meta"]["n_test"] == n_test


def test_oot_threshold_chosen_on_validation(oot):
    from src.evaluate import cost_optimal_threshold
    results, splits = oot
    _, y_va = splits["val"]
    for name in ("logistic_regression", "lightgbm_tuned"):
        r = results[name]
        expected = cost_optimal_threshold(y_va, r["y_prob_val"])
        assert r["metrics_optimal"]["threshold"] == pytest.approx(expected)


def test_lgbm_early_stopping_sets_n_estimators(oot):
    results, _ = oot
    lgbm = results["lightgbm_tuned"]
    assert lgbm["best_params"]["n_estimators"] == lgbm["best_iteration"] >= 1
    assert len(lgbm["trials"]) == 2
    assert np.isclose(lgbm["trials"]["val_pr_auc"].max(), lgbm["best_val_pr_auc"])
