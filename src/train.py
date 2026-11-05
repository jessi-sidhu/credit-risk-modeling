"""Train, tune, and evaluate logistic regression (benchmark) and LightGBM.

Use as a module from the modeling notebook, or run as a CLI:
    python -m src.train [--quick] [--save] [--random-split]

Default: out-of-time split on `issue_d` (train <= TRAIN_UNTIL, validation
in between, test >= TEST_FROM). LightGBM hyperparameters are searched on
train with early stopping on validation; the cost-optimal threshold is
also picked on validation. The test set is only used for reporting.

`--random-split` runs the old stratified random 80/20 split with 5-fold
CV tuning. It ignores time and is kept for comparison only.

`--quick` uses 5,000 rows and 10 search iterations for a fast smoke test.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from lightgbm import early_stopping
from scipy.stats import loguniform, randint
from sklearn.metrics import average_precision_score
from sklearn.model_selection import (
    ParameterSampler,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)

from . import RANDOM_STATE
from .data import load_data, temporal_split
from .calibration import plot_reliability
from .evaluate import (
    CostMatrix,
    bootstrap_ci,
    cost_optimal_threshold,
    evaluate_at_threshold,
    expected_cost,
    headline_metrics,
)
from .features import engineer_features, make_preprocessor, split_xy
from .models import lightgbm_classifier, logistic_regression

logger = logging.getLogger(__name__)

_REPORTS = Path(__file__).resolve().parent.parent / "reports"
_ARTIFACTS = _REPORTS / "artifacts"
N_BOOTSTRAP = 1_000

# Out-of-time split on the matured cohorts (2007-06 .. 2016-03).
TRAIN_UNTIL = "2013-12"
TEST_FROM = "2015-01"

# Search space shared by both tuning paths. n_estimators is only searched
# on the random split; the out-of-time path picks it by early stopping.
LGBM_PARAM_DIST = {
    "num_leaves": randint(15, 96),
    "learning_rate": loguniform(0.01, 0.2),
    "min_child_samples": randint(10, 80),
    "feature_fraction": [0.7, 0.8, 0.9, 1.0],
    "reg_lambda": loguniform(1e-3, 10.0),
}
MAX_TREES = 2_000
EARLY_STOPPING_ROUNDS = 50


# --- Out-of-time split (default) ---------------------------------------------

def temporal_xy(
    df: pd.DataFrame,
    train_until: str = TRAIN_UNTIL,
    test_from: str = TEST_FROM,
) -> dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """Split an engineered frame by `issue_d` into train/val/test (X, y)
    and log the size, date range and default rate of each part."""
    parts = dict(zip(("train", "val", "test"),
                     temporal_split(df, train_until=train_until, test_from=test_from)))
    for name, part in parts.items():
        if part.empty:
            raise ValueError(f"{name} split is empty for train_until={train_until}, "
                             f"test_from={test_from}")
        logger.info(
            "%-5s %8d rows  %s -> %s  default rate %.4f",
            name, len(part),
            part["issue_d"].min().strftime("%Y-%m"),
            part["issue_d"].max().strftime("%Y-%m"),
            part["default"].mean(),
        )
    return {name: split_xy(part) for name, part in parts.items()}


def tune_lightgbm_early_stopping(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    n_iter: int = 30,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Random search over LGBM_PARAM_DIST. Each candidate is fit on train
    and early-stopped on validation PR-AUC; the candidate with the best
    validation PR-AUC wins. The preprocessor is fit on train only.

    Returns {'model': fitted Pipeline, 'best_params', 'best_val_pr_auc',
    'best_iteration', 'trials': DataFrame}.
    """
    pre = make_preprocessor(scale_numeric=False).fit(X_tr)
    Z_tr, Z_va = pre.transform(X_tr), pre.transform(X_va)

    best: dict[str, Any] = {"best_val_pr_auc": -np.inf}
    trials = []
    for params in ParameterSampler(LGBM_PARAM_DIST, n_iter=n_iter, random_state=random_state):
        pipe = lightgbm_classifier(n_estimators=MAX_TREES, **params)
        clf = pipe.named_steps["clf"]
        clf.fit(
            Z_tr, y_tr,
            eval_set=[(Z_va, y_va)],
            eval_metric="average_precision",
            callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        score = float(average_precision_score(y_va, clf.predict_proba(Z_va)[:, 1]))
        trials.append({**params, "best_iteration": clf.best_iteration_, "val_pr_auc": score})
        if score > best["best_val_pr_auc"]:
            pipe.steps[0] = ("pre", pre)
            best = {
                "model": pipe,
                "best_params": {**params, "n_estimators": int(clf.best_iteration_)},
                "best_val_pr_auc": score,
                "best_iteration": int(clf.best_iteration_),
            }
    best["trials"] = pd.DataFrame(trials).sort_values("val_pr_auc", ascending=False)
    return best


def _evaluate_on_test(model, splits: dict, cost: CostMatrix) -> dict:
    """Pick the cost-optimal threshold on validation, report on test."""
    X_va, y_va = splits["val"]
    X_te, y_te = splits["test"]
    y_prob_val = model.predict_proba(X_va)[:, 1]
    y_prob = model.predict_proba(X_te)[:, 1]
    thr = cost_optimal_threshold(y_va, y_prob_val, cost)
    return {
        "model": model,
        "y_prob_val": y_prob_val,
        "y_prob": y_prob,
        "metrics_default": evaluate_at_threshold(y_te, y_prob, 0.5),
        "metrics_optimal": evaluate_at_threshold(y_te, y_prob, thr),
    }


def fit_and_evaluate_oot(
    quick: bool = False,
    df: pd.DataFrame | None = None,
    train_until: str = TRAIN_UNTIL,
    test_from: str = TEST_FROM,
    n_iter: int | None = None,
) -> Tuple[dict, dict]:
    """Out-of-time pipeline. Returns (results, splits)."""
    if df is None:
        df = load_data()
    df = engineer_features(df)
    if quick:
        df = df.sample(n=5_000, random_state=RANDOM_STATE).reset_index(drop=True)
    if n_iter is None:
        n_iter = 10 if quick else 30

    splits = temporal_xy(df, train_until=train_until, test_from=test_from)
    X_tr, y_tr = splits["train"]
    X_va, y_va = splits["val"]
    cost = CostMatrix()

    results: dict[str, Any] = {}
    results["logistic_regression"] = _evaluate_on_test(
        logistic_regression().fit(X_tr, y_tr), splits, cost,
    )

    tuned = tune_lightgbm_early_stopping(X_tr, y_tr, X_va, y_va, n_iter=n_iter)
    results["lightgbm_tuned"] = {
        **_evaluate_on_test(tuned["model"], splits, cost),
        "best_params": tuned["best_params"],
        "best_val_pr_auc": tuned["best_val_pr_auc"],
        "best_iteration": tuned["best_iteration"],
        "trials": tuned["trials"],
    }
    _, y_te = splits["test"]
    probs = {name: results[name]["y_prob"] for name in ("logistic_regression", "lightgbm_tuned")}
    for name, y_prob in probs.items():
        results[name]["headline"] = headline_metrics(y_te, y_prob)
    results["bootstrap"] = bootstrap_ci(
        y_te, probs, diff=("lightgbm_tuned", "logistic_regression"),
        n_boot=N_BOOTSTRAP, seed=RANDOM_STATE,
    )
    results["meta"] = {
        "split": "out_of_time",
        "train_until": train_until,
        "test_from": test_from,
        **{f"n_{k}": int(len(y)) for k, (_, y) in splits.items()},
        **{f"default_rate_{k}": float(y.mean()) for k, (_, y) in splits.items()},
    }
    return results, splits


# --- Random split (comparison only) ------------------------------------------

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
    param_dist = {f"clf__{k}": v for k, v in LGBM_PARAM_DIST.items()}
    param_dist["clf__n_estimators"] = randint(200, 800)
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


def fit_and_evaluate_all(quick: bool = False) -> Tuple[dict, tuple]:
    """Random-split pipeline (comparison only). The threshold is picked on
    the test fold here, so its test metrics are optimistic."""
    X_tr, X_te, y_tr, y_te = prepare_split(n=5_000 if quick else None)
    cost = CostMatrix()
    n_iter = 10 if quick else 30

    results: dict[str, Any] = {}

    results["logistic_regression"] = _fit_one(
        logistic_regression(), X_tr, y_tr, X_te, y_te, cost,
    )

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


# --- Reporting ----------------------------------------------------------------

_MODELS = ("logistic_regression", "lightgbm_tuned")


def headline_table(results: dict) -> pd.DataFrame:
    """Test-set ROC-AUC, Gini, KS, PR-AUC, Brier, ECE with bootstrap CIs."""
    rows = []
    for name in _MODELS:
        h, ci = results[name]["headline"], results["bootstrap"][name]
        rows.append({
            "model": name,
            "roc_auc": f"{h['roc_auc']:.4f} [{ci['roc_auc']['ci_low']:.4f}, {ci['roc_auc']['ci_high']:.4f}]",
            "gini": round(h["gini"], 4),
            "ks": round(h["ks"], 4),
            "pr_auc": f"{h['pr_auc']:.4f} [{ci['pr_auc']['ci_low']:.4f}, {ci['pr_auc']['ci_high']:.4f}]",
            "brier": round(h["brier"], 4),
            "ece": round(h["ece"], 4),
        })
    return pd.DataFrame(rows).set_index("model")


def metrics_summary(results: dict) -> dict:
    """Everything the README/report may quote, as plain JSON types."""
    def _plain(d):
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in d.items()}

    return {
        "meta": results["meta"],
        "models": {
            name: {
                "test": results[name]["headline"],
                "cost_threshold": {
                    "threshold_chosen_on": "validation",
                    **_plain(results[name]["metrics_optimal"]),
                },
                "at_0.5": _plain(results[name]["metrics_default"]),
            }
            for name in _MODELS
        },
        "bootstrap_95ci": {"n_resamples": N_BOOTSTRAP, **results["bootstrap"]},
        "lightgbm": {
            "best_params": _plain(results["lightgbm_tuned"]["best_params"]),
            "best_val_pr_auc": results["lightgbm_tuned"]["best_val_pr_auc"],
        },
        "cost_matrix": {"fn_cost": CostMatrix().fn_cost, "fp_cost": CostMatrix().fp_cost},
    }


def results_table(results: dict) -> pd.DataFrame:
    rows = []
    for name in _MODELS:
        r = results[name]
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
    import joblib

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    lgbm = results["lightgbm_tuned"]
    joblib.dump(lgbm["model"], out / "lightgbm_tuned.joblib")
    X_te.to_csv(out / "X_test.csv", index=False)
    y_te.to_frame("default").to_csv(out / "y_test.csv", index=False)
    np.save(out / "y_prob_lightgbm.npy", lgbm["y_prob"])
    np.save(out / "y_prob_logreg.npy", results["logistic_regression"]["y_prob"])
    table.to_csv(out / "results_table.csv")

    bp = lgbm["best_params"]
    json.dump({k: (v.item() if hasattr(v, "item") else v) for k, v in bp.items()},
              open(out / "best_params.json", "w"), indent=2)

    cost = CostMatrix()
    thr = lgbm["metrics_optimal"]["threshold"]
    json.dump({"cost_optimal": thr,
               "cost_at_default": expected_cost(y_te, lgbm["y_prob"], 0.5, cost),
               "cost_at_optimal": expected_cost(y_te, lgbm["y_prob"], thr, cost)},
              open(out / "thresholds.json", "w"), indent=2)

    if "meta" in results:
        json.dump(results["meta"], open(out / "split_meta.json", "w"), indent=2)
    if "trials" in lgbm:
        lgbm["trials"].to_csv(out / "lgbm_trials.csv", index=False)
    if "bootstrap" in results:
        json.dump(metrics_summary(results), open(out / "metrics.json", "w"), indent=2)
        plot_reliability(
            y_te,
            {name: results[name]["y_prob"] for name in _MODELS},
            _REPORTS / "figures" / "reliability.png",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="5,000 rows + 10 tuning iters for a fast smoke test")
    parser.add_argument("--save", action="store_true",
                        help="persist artifacts to reports/artifacts/ "
                             "(skipped by default so --quick stays disposable)")
    parser.add_argument("--random-split", action="store_true",
                        help="stratified random split instead of out-of-time "
                             "(comparison only); saves to "
                             "reports/artifacts/random_split/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    if args.random_split:
        results, (_, X_te, _, y_te) = fit_and_evaluate_all(quick=args.quick)
        out = _ARTIFACTS / "random_split"
    else:
        results, splits = fit_and_evaluate_oot(quick=args.quick)
        X_te, y_te = splits["test"]
        out = _ARTIFACTS
        lgbm = results["lightgbm_tuned"]
        print(f"LightGBM best params: {lgbm['best_params']}  "
              f"(val PR-AUC {lgbm['best_val_pr_auc']:.4f})")
        print("\nOut-of-time test metrics (95% bootstrap CI):")
        print(headline_table(results))
        d = results["bootstrap"]["lightgbm_tuned - logistic_regression"]
        for m in ("roc_auc", "pr_auc"):
            print(f"LightGBM - LR {m}: {d[m]['estimate']:+.4f} "
                  f"[{d[m]['ci_low']:+.4f}, {d[m]['ci_high']:+.4f}]")
        print()

    table = results_table(results)
    print(table)

    if args.save:
        save_artifacts(results, X_te, y_te, table, out)
        print(f"\nArtifacts saved to {out}")


if __name__ == "__main__":
    main()
