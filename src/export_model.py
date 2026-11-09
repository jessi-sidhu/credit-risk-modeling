"""Export the Phase 2 LightGBM hazard model as a self-contained bundle.

    python -m src.export_model          (make export-model)

Copies the fitted model and, from the saved Phase 2 artifacts, the level
offsets, LGD table and headline metrics into models/pd_survival_v1/.
Reference distributions for drift monitoring are computed from the Phase 2
training loans (issued up to 2013-12), so this step needs data/archive/;
nothing in the bundle contains raw loan records.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .drift import _bin_numeric, _hist
from .features import CATEGORICAL, engineer_features
from .serve.bundle import BUNDLE_FILES, DEFAULT_BUNDLE, load_bundle
from .survival_data import load_survival_data
from .train_survival import split_masks

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "reports" / "artifacts" / "survival"
MODEL_VERSION = "pd_survival_v1"
MAX_BUNDLE_MB = 20
PSI_BINS = 10

# Application-time inputs the API accepts (the raw columns behind the Phase 1
# feature list; engineered ratios are computed from them). `valid` is a hard
# sanity range: requests outside it are rejected. The training 0.1% / 99.9%
# caps (from the fitted model) are softer: values beyond them are scored at
# the cap and the response carries a warning.
INPUTS = {
    "loan_amount":    {"dtype": "float", "valid": [1, 1_000_000], "label": "Loan amount"},
    "term":           {"dtype": "int",   "allowed": [36, 60],     "label": "Loan term"},
    "int_rate":       {"dtype": "float", "valid": [0, 100],       "label": "Interest rate"},
    "grade":          {"dtype": "category",                       "label": "LC grade"},
    "emp_length":     {"dtype": "int",   "valid": [0, 10],        "label": "Years employed"},
    "home_ownership": {"dtype": "category",                       "label": "Home ownership"},
    "annual_income":  {"dtype": "float", "valid": [0, 1e9],       "label": "Annual income"},
    "purpose":        {"dtype": "category",                       "label": "Loan purpose"},
    "dti":            {"dtype": "float", "valid": [-1, 1000],     "label": "Debt-to-income ratio"},
    "fico":           {"dtype": "float", "valid": [300, 850],     "label": "FICO score"},
    "open_acc":       {"dtype": "float", "valid": [0, 200],       "label": "Open credit lines", "nullable": True},
    "revol_util":     {"dtype": "float", "valid": [0, 1000],      "label": "Revolving utilization", "nullable": True},
}
ENGINEERED_LABELS = {
    "loan_to_income": "Loan amount relative to income",
    "installment_estimate": "Monthly installment",
    "installment_to_income": "Installment relative to income",
    "emp_x_homeowner": "Years employed (homeowners)",
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def build_schema(model, train: pd.DataFrame) -> dict:
    """`train`: engineered training loans, for the medians reason codes use
    to say whether a value is high or low."""
    ohe = model.pre.named_transformers_["cat"].named_steps["ohe"]
    categories = {c: [str(v) for v in cats] for c, cats in zip(CATEGORICAL, ohe.categories_)}
    fields = {}
    for name, spec in INPUTS.items():
        f = dict(spec)
        if spec["dtype"] == "category":
            f["allowed"] = categories[name]
        if name in model.clip_:
            f["training_cap"] = list(model.clip_[name])
        if spec["dtype"] != "category":
            f["training_median"] = float(train[name].median())
        fields[name] = f
    return {
        "fields": fields,
        "engineered": {name: {"label": label, "training_cap": list(model.clip_[name]),
                              "training_median": float(train[name].replace([np.inf, -np.inf], np.nan).median())}
                       for name, label in ENGINEERED_LABELS.items()},
        "optional": {"issue_d": "issue month (YYYY-MM), used for monitoring only",
                     "loan_id": "client-supplied id, used to match outcomes later"},
    }


def reference_stats(train: pd.DataFrame) -> dict:
    """Binned training distributions per input, in the format drift monitoring reads."""
    out = {}
    for name, spec in INPUTS.items():
        s = train[name]
        if spec["dtype"] == "category" or name == "term":
            shares = s.astype(str).value_counts(normalize=True)
            out[name] = {"kind": "categorical", "shares": {k: float(v) for k, v in shares.items()}}
        else:
            vals = s.to_numpy(dtype=float)
            edges = _bin_numeric(vals[~np.isnan(vals)], PSI_BINS)
            out[name] = {"kind": "numeric", "edges": [float(e) for e in edges],
                         "shares": [float(p) for p in _hist(vals, edges)]}   # last share = missing
    return {"source": "Phase 2 training loans, issued 2007-06 to 2013-12",
            "n_loans": int(len(train)), "features": out}


def model_card(meta: dict, metrics: dict, level: dict, ecl: dict) -> dict:
    lgbm = level["results"]["lgbm"]
    return {
        "model_version": MODEL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "model": "Discrete-time competing-risks hazard model (LightGBM multiclass: no event / default / prepay per quarter)",
        "training_window": {k: meta["splits"][k]["issue_range"] for k in ("train", "val")},
        "training_loans": meta["splits"]["train"]["loans"],
        "lgbm_params": meta["lgbm"]["best_params"],
        "validation_logloss": meta["lgbm"]["val_logloss"],
        "test": {
            "issue_range": meta["splits"]["test"]["issue_range"],
            "loans": metrics["n_test_loans"],
            "auc_default_by_horizon": {h: v["lgbm"]["roc_auc"] for h, v in metrics["horizon_auc"].items()},
        },
        "known_under_prediction": {
            "description": ("Predicted / observed cumulative default probability on 2015-2018 cohorts, "
                            "observed with Late (31-120 days) counted as default. 1.0 = calibrated."),
            "raw": {k: v["ratio_arrears_as_default"] for k, v in lgbm["base"]["cells"].items()},
            "calibrated": {k: v["ratio_arrears_as_default"] for k, v in lgbm["recalibrated"]["cells"].items()},
            "matured_loss_predicted_over_realized": {"raw": lgbm["base"]["matured_loss"]["ratio"],
                                                     "calibrated": lgbm["recalibrated"]["matured_loss"]["ratio"]},
        },
        "calibration": "level offsets fit on the 2014 validation loans (level_offsets.json); applied by default",
        "lgd": ecl["note"],
        "intended_use": "Portfolio illustration. Not a regulatory model and not for credit decisions.",
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    meta = json.load(open(ART / "train_meta.json"))
    metrics = json.load(open(ART / "metrics.json"))
    level = json.load(open(ART / "level_experiments.json"))
    ecl = json.load(open(ART / "ecl.json"))
    if meta.get("quick") or level.get("quick"):
        raise SystemExit("Phase 2 artifacts come from a --quick run; run `make train-survival` first")

    out = DEFAULT_BUNDLE
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ART / "hazard_lgbm.joblib", out / "hazard_lgbm.joblib")
    import joblib
    model = joblib.load(out / "hazard_lgbm.joblib")

    offsets = level["fit"]["lgbm"]["recalibrated"]["offsets_log_odds"]
    json.dump({"default": offsets["default"], "prepay": offsets["prepay"],
               "source": "level_experiments.json: log-odds shifts fit on 2014 validation loan-quarters"},
              open(out / "level_offsets.json", "w"), indent=2)
    json.dump({"lgd": {g: v["lgd"] for g, v in ecl["lgd_by_grade"].items()},
               "n_train_chargeoffs": {g: v["n_train_chargeoffs"] for g, v in ecl["lgd_by_grade"].items()},
               "source": "ecl.json: training-period (2007-2013) charge-offs"},
              open(out / "lgd_by_grade.json", "w"), indent=2)
    features, _ = load_survival_data()
    train = features[split_masks(features)["train"]]
    json.dump(build_schema(model, engineer_features(train)), open(out / "feature_schema.json", "w"), indent=2)
    json.dump(reference_stats(train), open(out / "reference_stats.json", "w"), indent=2)
    del features, train

    json.dump(model_card(meta, metrics, level, ecl), open(out / "model_card.json", "w"), indent=2)

    load_bundle(out)                                   # round-trip check
    sizes = {f: (out / f).stat().st_size for f in BUNDLE_FILES}
    total_mb = sum(sizes.values()) / 1e6
    for f, b in sizes.items():
        print(f"  {f:24s} {b / 1e3:8.1f} KB")
    print(f"bundle {out}: {total_mb:.2f} MB")
    if total_mb > MAX_BUNDLE_MB:
        raise SystemExit(f"bundle is {total_mb:.1f} MB, over the {MAX_BUNDLE_MB} MB limit")


if __name__ == "__main__":
    main()
