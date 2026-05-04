"""Train, tune, and evaluate the three models.

Use as a module from the modeling notebook, or run as a CLI:
    python -m src.train [--quick]

`--quick` uses 5,000 rows and 10 randomized-search iterations so the
end-to-end smoke test finishes in under a minute.
"""
from __future__ import annotations

import argparse
from typing import Any, Tuple

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from . import RANDOM_STATE
from .data import load_data
from .evaluate import (
    CostMatrix,
    cost_optimal_threshold,
    evaluate_at_threshold,
)
from .features import engineer_features, split_xy
from .models import lightgbm_classifier, logistic_regression, random_forest


def prepare_split(
    n: int | None = None,
    seed: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = engineer_features(load_data())
    if n is not None:
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    X, y = split_xy(df)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)


def tune_lightgbm(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    n_iter: int = 30,
    cv: int = 5,
    random_state: int = RANDOM_STATE,
) -> RandomizedSearchCV:
    base = lightgbm_classifier()
    param_dist = {
        "clf__num_leaves": randint(15, 96),
        "clf__learning_rate": loguniform(0.01, 0.2),
        "clf__min_child_samples": randint(10, 80),
        "clf__feature_fraction": [0.7, 0.8, 0.9, 1.0],
        "clf__reg_lambda": loguniform(1e-3, 10.0),
        "clf__n_estimators": randint(200, 800),
    }
    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv_split,
        scoring="average_precision",
        random_state=random_state,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_tr, y_tr)
    return search


def fit_and_evaluate_all(quick: bool = False) -> Tuple[dict, tuple]:
    X_tr, X_te, y_tr, y_te = prepare_split(n=5_000 if quick else None)
    cost = CostMatrix()
    n_iter = 10 if quick else 30

    results: dict[str, Any] = {}

    for name, model in [
        ("logistic_regression", logistic_regression()),
        ("random_forest", random_forest()),
    ]:
        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_te)[:, 1]
        thr = cost_optimal_threshold(y_te, y_prob, cost)
        results[name] = {
            "model": model,
            "y_prob": y_prob,
            "metrics_default": evaluate_at_threshold(y_te, y_prob, 0.5),
            "metrics_optimal": evaluate_at_threshold(y_te, y_prob, thr),
        }

    search = tune_lightgbm(X_tr, y_tr, n_iter=n_iter)
    best = search.best_estimator_
    y_prob = best.predict_proba(X_te)[:, 1]
    thr = cost_optimal_threshold(y_te, y_prob, cost)
    results["lightgbm_tuned"] = {
        "model": best,
        "y_prob": y_prob,
        "best_params": search.best_params_,
        "best_cv_pr_auc": float(search.best_score_),
        "metrics_default": evaluate_at_threshold(y_te, y_prob, 0.5),
        "metrics_optimal": evaluate_at_threshold(y_te, y_prob, thr),
    }

    return results, (X_tr, X_te, y_tr, y_te)


def results_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        m_opt = r["metrics_optimal"]
        m_def = r["metrics_default"]
        rows.append({
            "model": name,
            "pr_auc": round(m_opt["pr_auc"], 4),
            "roc_auc": round(m_opt["roc_auc"], 4),
            "f1@0.5": round(m_def["f1"], 4),
            "f1@cost-opt": round(m_opt["f1"], 4),
            "precision@cost-opt": round(m_opt["precision"], 4),
            "recall@cost-opt": round(m_opt["recall"], 4),
            "threshold": round(m_opt["threshold"], 3),
            "tp": m_opt["tp"], "fp": m_opt["fp"],
            "fn": m_opt["fn"], "tn": m_opt["tn"],
        })
    return pd.DataFrame(rows).set_index("model")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="5,000 rows + 10 tuning iters for a fast smoke test")
    args = parser.parse_args()

    results, _ = fit_and_evaluate_all(quick=args.quick)
    table = results_table(results)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(table)


if __name__ == "__main__":
    main()
