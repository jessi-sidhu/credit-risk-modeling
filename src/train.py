"""Train, tune, and evaluate the three models.

Use as a module from the modeling notebook, or run as a CLI:
    python -m src.train [--quick]

`--quick` uses 5,000 rows and 10 randomized-search iterations so the
end-to-end smoke test finishes in under a minute.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from . import RANDOM_STATE
from .data import load_data, temporal_split
from .evaluate import (
    CostMatrix,
    cost_optimal_threshold,
    evaluate_at_threshold,
)
from .features import engineer_features, split_xy
from .models import lightgbm_classifier, logistic_regression, random_forest
from .synth import generate


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


def _load_tuned_lgbm_params() -> dict:
    """Use the best params from the random-split tuning run if cached;
    otherwise fall back to the package defaults. Lets the temporal
    train comparison reuse the random-split tune budget."""
    import json
    p = (Path(__file__).resolve().parent.parent
         / "reports" / "artifacts" / "best_params.json")
    if not p.exists():
        return {}
    raw = json.load(open(p))
    return {k.replace("clf__", ""): v for k, v in raw.items()}


def _fit_one(model, X_tr, y_tr, X_te, y_te, cost: CostMatrix) -> dict:
    model.fit(X_tr, y_tr)
    y_prob = model.predict_proba(X_te)[:, 1]
    thr = cost_optimal_threshold(y_te, y_prob, cost)
    return {
        "model": model,
        "y_prob": y_prob,
        "metrics_default": evaluate_at_threshold(y_te, y_prob, 0.5),
        "metrics_optimal": evaluate_at_threshold(y_te, y_prob, thr),
    }


def fit_and_evaluate_temporal(quick: bool = False) -> dict:
    """Train all three models on temporal-trend data, evaluating both a
    random split and a time-aware split. Same hyperparameters for both
    so the only thing that varies is the validation methodology.

    Returns:
        {'random_split': {..per-model..}, 'temporal_split': {...},
         'meta': {...}}
    """
    n = 5_000 if quick else 50_000
    df = engineer_features(generate(n=n, seed=RANDOM_STATE, temporal_trend=True))
    cost = CostMatrix()

    # --- Random split baseline (ignores time) -----------------------------
    X, y = split_xy(df)
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )

    # --- Temporal split ---------------------------------------------------
    train_until, test_from = "2017-06", "2018-01"
    tr_df, _, te_df = temporal_split(df, train_until=train_until, test_from=test_from)
    Xt_tr, yt_tr = split_xy(tr_df)
    Xt_te, yt_te = split_xy(te_df)

    tuned_params = _load_tuned_lgbm_params()

    def _models():
        return {
            "logistic_regression": logistic_regression(),
            "random_forest": random_forest(),
            "lightgbm_tuned": lightgbm_classifier(**tuned_params) if tuned_params else lightgbm_classifier(),
        }

    rand_results = {
        name: _fit_one(m, Xr_tr, yr_tr, Xr_te, yr_te, cost)
        for name, m in _models().items()
    }
    temp_results = {
        name: _fit_one(m, Xt_tr, yt_tr, Xt_te, yt_te, cost)
        for name, m in _models().items()
    }

    return {
        "random_split": rand_results,
        "temporal_split": temp_results,
        "meta": {
            "n_train_random": int(len(Xr_tr)), "n_test_random": int(len(Xr_te)),
            "n_train_temporal": int(len(Xt_tr)), "n_test_temporal": int(len(Xt_te)),
            "train_until": train_until, "test_from": test_from,
            "tuned_params_loaded": bool(tuned_params),
        },
        "splits": {
            "random": (Xr_tr, Xr_te, yr_tr, yr_te),
            "temporal": (Xt_tr, Xt_te, yt_tr, yt_te),
        },
    }


def temporal_results_table(results: dict) -> pd.DataFrame:
    """One row per (model × split) with the headline metrics."""
    rows = []
    for split in ("random_split", "temporal_split"):
        for name, r in results[split].items():
            m = r["metrics_optimal"]
            rows.append({
                "split": split.replace("_split", ""),
                "model": name,
                "pr_auc": round(m["pr_auc"], 4),
                "roc_auc": round(m["roc_auc"], 4),
                "f1@cost-opt": round(m["f1"], 4),
                "threshold": round(m["threshold"], 3),
            })
    return pd.DataFrame(rows).set_index(["split", "model"])


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


def save_artifacts(results: dict, X_te, y_te, table: pd.DataFrame, out_dir) -> None:
    """Persist trained model + test predictions + results table + best
    params + threshold for the downstream notebooks."""
    import json
    from pathlib import Path
    import joblib

    from .evaluate import (
        CostMatrix, cost_optimal_threshold, threshold_sweep, expected_cost,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    lgbm = results["lightgbm_tuned"]
    joblib.dump(lgbm["model"], out / "lightgbm_tuned.joblib")
    X_te.to_csv(out / "X_test.csv", index=False)
    y_te.to_frame("default").to_csv(out / "y_test.csv", index=False)
    np.save(out / "y_prob_lightgbm.npy", lgbm["y_prob"])
    np.save(out / "y_prob_logreg.npy", results["logistic_regression"]["y_prob"])
    np.save(out / "y_prob_rf.npy", results["random_forest"]["y_prob"])
    table.to_csv(out / "results_table.csv")

    bp = lgbm["best_params"]
    json.dump({k: (float(v) if hasattr(v, "item") else v) for k, v in bp.items()},
              open(out / "best_params.json", "w"), indent=2)

    cost = CostMatrix()
    sweep = threshold_sweep(y_te, lgbm["y_prob"], cost)
    thr_opt = float(sweep.loc[sweep["expected_cost"].idxmin(), "threshold"])
    cost_05 = expected_cost(y_te, lgbm["y_prob"], 0.5, cost)
    cost_opt = float(sweep["expected_cost"].min())
    json.dump({"cost_optimal": thr_opt,
               "cost_at_default": cost_05,
               "cost_at_optimal": cost_opt},
              open(out / "thresholds.json", "w"), indent=2)


def save_temporal_artifacts(results: dict, table: pd.DataFrame, out_dir) -> None:
    """Persist per-split predictions and the comparison table for the
    temporal notebook."""
    import json
    import joblib

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split in ("random_split", "temporal_split"):
        for name, r in results[split].items():
            np.save(out / f"y_prob_{split}_{name}.npy", r["y_prob"])

    # save the temporal-test X and y for the notebook to slice by group/cohort
    Xr_tr, Xr_te, yr_tr, yr_te = results["splits"]["random"]
    Xt_tr, Xt_te, yt_tr, yt_te = results["splits"]["temporal"]
    Xr_te.to_csv(out / "X_test_random.csv", index=False)
    Xt_te.to_csv(out / "X_test_temporal.csv", index=False)
    yr_te.to_frame("default").to_csv(out / "y_test_random.csv", index=False)
    yt_te.to_frame("default").to_csv(out / "y_test_temporal.csv", index=False)

    table.to_csv(out / "temporal_results_table.csv")
    json.dump(results["meta"], open(out / "temporal_meta.json", "w"), indent=2)
    joblib.dump(results["temporal_split"]["lightgbm_tuned"]["model"],
                out / "lightgbm_temporal.joblib")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="5,000 rows + 10 tuning iters for a fast smoke test")
    parser.add_argument("--save", action="store_true",
                        help="persist artifacts to reports/artifacts/ "
                             "(skipped by default so --quick stays disposable)")
    parser.add_argument("--temporal", action="store_true",
                        help="run the temporal-validation comparison instead "
                             "of the random-split tune; saves to "
                             "reports/artifacts/temporal/")
    args = parser.parse_args()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    if args.temporal:
        results = fit_and_evaluate_temporal(quick=args.quick)
        table = temporal_results_table(results)
        print(table)
        if args.save:
            out = Path(__file__).resolve().parent.parent / "reports" / "artifacts" / "temporal"
            save_temporal_artifacts(results, table, out)
            print(f"\nArtifacts saved to {out}")
        return

    results, splits = fit_and_evaluate_all(quick=args.quick)
    table = results_table(results)
    print(table)

    if args.save:
        _, X_te, _, y_te = splits
        out = Path(__file__).resolve().parent.parent / "reports" / "artifacts"
        save_artifacts(results, X_te, y_te, table, out)
        print(f"\nArtifacts saved to {out}")


if __name__ == "__main__":
    main()
