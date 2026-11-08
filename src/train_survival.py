"""Train the Phase 2 discrete-time competing-risks hazard models.

    python -m src.train_survival [--quick]

Same out-of-time split on `issue_d` as Phase 1 (train <= 2013-12,
validation 2014, test >= 2015-01), but now with ALL loans: both terms,
finished or not (unfinished loans are censored).

- Benchmark: multinomial logistic regression on person-quarter rows.
- Main: LightGBM multiclass; small random search, each candidate early-
  stopped on validation multi-class log loss, best validation log loss wins.

Writes to reports/artifacts/survival/:
    hazard_lr.joblib, hazard_lgbm.joblib   fitted models
    hazards_{lr,lgbm}.npy                  test hazards (n_test, 20, 2): default, prepay
    test_loans.csv.gz                      per-loan test table with CIF_default at 12/24/36/60 months
    train_meta.json                        split sizes, row counts, chosen params, validation losses
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.metrics import log_loss
from sklearn.model_selection import ParameterSampler

from . import RANDOM_STATE
from .features import engineer_features
from .survival import (
    MAX_TRAIN_LOANS,
    HazardModel,
    cif_at,
    cumulative_incidence,
    expand_person_period,
    sample_loans,
)
from .survival_data import EVENT_NAMES, load_survival_data
from .train import TEST_FROM, TRAIN_UNTIL

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).resolve().parent.parent / "reports" / "artifacts" / "survival"
HORIZONS = (12, 24, 36, 60)
N_SEARCH = 6
LGBM_PARAM_DIST = {
    "num_leaves": randint(15, 96),
    "learning_rate": loguniform(0.02, 0.2),
    "min_child_samples": randint(50, 400),
    "feature_fraction": [0.7, 0.8, 0.9, 1.0],
    "reg_lambda": loguniform(1e-3, 10.0),
}


def split_masks(features: pd.DataFrame) -> dict[str, np.ndarray]:
    d = features["issue_d"]
    end_train = pd.Period(TRAIN_UNTIL, freq="M").to_timestamp("M")
    start_test = pd.Period(TEST_FROM, freq="M").to_timestamp()
    return {
        "train": (d <= end_train).to_numpy(),
        "val": ((d > end_train) & (d < start_test)).to_numpy(),
        "test": (d >= start_test).to_numpy(),
    }


def _val_logloss(model: HazardModel, pp: pd.DataFrame) -> float:
    return float(log_loss(pp["y"], model.predict_rows(pp), labels=[0, 1, 2]))


def tune_lgbm(pp_tr, pp_va, n_iter: int) -> tuple[HazardModel, dict, list]:
    best, trials = None, []
    for i, params in enumerate(ParameterSampler(LGBM_PARAM_DIST, n_iter=n_iter, random_state=RANDOM_STATE)):
        t0 = time.time()
        m = HazardModel("lgbm", **params).fit(pp_tr, eval_pp=pp_va)
        loss = _val_logloss(m, pp_va)
        trials.append({**{k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()},
                       "best_iteration": m.best_iteration_, "val_logloss": loss,
                       "seconds": round(time.time() - t0, 1)})
        logger.info("lgbm trial %d/%d: val logloss %.5f, %s trees, %.0fs",
                    i + 1, n_iter, loss, m.best_iteration_, time.time() - t0)
        if best is None or loss < best[1]:
            best = (m, loss, params)
    m, loss, params = best
    chosen = {**{k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()},
              "n_estimators": m.best_iteration_}
    return m, chosen, trials


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="20k train / 20k val / 50k test loans, 2 search iterations")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    features, outcomes = load_survival_data()
    features = engineer_features(features)
    masks = split_masks(features)
    parts = {k: (features[m], outcomes[m]) for k, m in masks.items()}
    data_attrs = dict(features.attrs)
    del features, outcomes; gc.collect()

    caps = {"train": 20_000, "val": 20_000, "test": 50_000} if args.quick else \
           {"train": MAX_TRAIN_LOANS, "val": MAX_TRAIN_LOANS, "test": None}
    meta = {"quick": args.quick, "train_until": TRAIN_UNTIL, "test_from": TEST_FROM,
            "data": data_attrs, "splits": {}}
    for k, (f, o) in parts.items():
        if caps[k] is not None:
            parts[k] = sample_loans(f, o, caps[k])
        f, o = parts[k]
        meta["splits"][k] = {
            "loans": int(len(f)),
            "events": {EVENT_NAMES[e]: int(n) for e, n in o["event"].value_counts().sort_index().items()},
            "by_term": {str(t): int(n) for t, n in f["term"].value_counts().sort_index().items()},
            "issue_range": [f["issue_d"].min().strftime("%Y-%m"), f["issue_d"].max().strftime("%Y-%m")],
        }
        logger.info("%s: %d loans %s", k, len(f), meta["splits"][k]["events"])

    pp = {}
    for k in ("train", "val"):
        pp[k] = expand_person_period(*parts[k])
        meta["splits"][k]["person_period_rows"] = int(len(pp[k]))
        meta["splits"][k]["person_period_mb"] = round(pp[k].memory_usage(deep=True).sum() / 1e6, 1)
        meta["splits"][k]["row_targets"] = {EVENT_NAMES[e]: int(n)
                                            for e, n in pp[k]["y"].value_counts().sort_index().items()}

    t0 = time.time()
    lr = HazardModel("lr").fit(pp["train"])
    meta["lr"] = {"val_logloss": _val_logloss(lr, pp["val"]), "seconds": round(time.time() - t0, 1),
                  "n_iter": int(np.max(lr.clf.n_iter_))}
    logger.info("lr: val logloss %.5f (%.0fs)", meta["lr"]["val_logloss"], meta["lr"]["seconds"])

    lgbm, chosen, trials = tune_lgbm(pp["train"], pp["val"], n_iter=2 if args.quick else N_SEARCH)
    meta["lgbm"] = {"best_params": chosen, "val_logloss": min(t["val_logloss"] for t in trials),
                    "trials": trials}
    # a model that predicts the training base rates per row, for reference
    base = pp["train"]["y"].value_counts(normalize=True).sort_index().to_numpy()
    meta["base_rate_val_logloss"] = float(log_loss(pp["val"]["y"], np.tile(base, (len(pp["val"]), 1)),
                                                   labels=[0, 1, 2]))
    del pp; gc.collect()

    f_te, o_te = parts["test"]
    table = pd.DataFrame({
        "issue_d": f_te["issue_d"].dt.strftime("%Y-%m"), "term": f_te["term"], "grade": f_te["grade"],
        "int_rate": f_te["int_rate"], "loan_amount": f_te["loan_amount"],
        "duration": o_te["duration"], "event": o_te["event"], "loan_status": o_te["loan_status"],
    }, index=f_te.index)
    for name, model in (("lr", lr), ("lgbm", lgbm)):
        t0 = time.time()
        h = model.hazards(f_te)
        cif = cumulative_incidence(h)["cif_default"]
        for hz in HORIZONS:
            table[f"cif_default_{hz}_{name}"] = cif_at(cif, hz).astype(np.float32)
        np.save(OUT_DIR / f"hazards_{name}.npy", h[..., 1:].astype(np.float32))
        joblib.dump(model, OUT_DIR / f"hazard_{name}.joblib")
        logger.info("%s: test hazards for %d loans in %.0fs", name, len(f_te), time.time() - t0)
        del h, cif; gc.collect()

    table.index.name = "id"
    table.to_csv(OUT_DIR / "test_loans.csv.gz", compression="gzip")
    json.dump(meta, open(OUT_DIR / "train_meta.json", "w"), indent=2, default=int)
    print(json.dumps({k: meta[k] for k in ("lr", "base_rate_val_logloss")}, indent=2))
    print("lgbm:", json.dumps(meta["lgbm"]["best_params"]), "val logloss", round(meta["lgbm"]["val_logloss"], 5))
    print(table.filter(like="cif_default").describe().round(4).to_string())
    print(f"saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
