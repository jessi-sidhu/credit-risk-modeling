"""Evaluate the Phase 2 hazard models on the out-of-time test set.

    python -m src.survival_eval            (after python -m src.train_survival)

1. Discrimination at fixed horizons (12, 24, 36 months): AUC of CIF_default(h)
   on loans whose status at h is known, with bootstrap 95% CIs.
2. Calibration: mean predicted CIF_default(h) vs observed (Aalen-Johansen)
   by issue year and by grade, with 2016-2018 originations reported
   separately.
3. Apples-to-apples vs Phase 1: on the matured 36-month test loans Phase 1
   was scored on, Phase 1 PD vs Phase 2 CIF_default(36).
4. Example default curves per grade.

Writes reports/artifacts/survival/metrics.json and two figures.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from . import RANDOM_STATE
from .calibration import expected_calibration_error
from .evaluate import bootstrap_ci
from .features import engineer_features
from .survival import MAX_PERIODS, PERIOD_MONTHS, cumulative_incidence
from .survival_data import (
    BAND_COLORS,
    EVENT_DEFAULT,
    EVENT_PREPAY,
    GRADE_BANDS,
    load_survival_data,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "reports" / "artifacts" / "survival"
FIG = ROOT / "reports" / "figures"
MODELS = ("lr", "lgbm")
AUC_HORIZONS = (12, 24, 36)
N_BOOTSTRAP = 1_000
RECENT_YEARS = (2016, 2017, 2018)


# --- 1. Discrimination at a horizon --------------------------------------------

def known_status_at(t: pd.DataFrame, h: int) -> tuple[np.ndarray, np.ndarray]:
    """(mask, label) for "defaulted by month h".

    Positive: defaulted at or before h. Negative: still observed after h
    (any event type), or finished without default by h (prepaid, or paid
    at term). Loans censored before h while still open are excluded:
    whether they default by h is unknown.
    """
    d, e, fin = t["duration"].to_numpy(), t["event"].to_numpy(), t["finished"].to_numpy()
    pos = (e == EVENT_DEFAULT) & (d <= h)
    neg = (d > h) | (fin & (e != EVENT_DEFAULT) & (d <= h))
    return pos | neg, pos[pos | neg].astype(int)


def horizon_auc(t: pd.DataFrame) -> dict:
    out = {}
    for h in AUC_HORIZONS:
        mask, y = known_status_at(t, h)
        probs = {m: t.loc[mask, f"cif_default_{h}_{m}"].to_numpy() for m in MODELS}
        ci = bootstrap_ci(y, probs, diff=("lgbm", "lr"), n_boot=N_BOOTSTRAP, seed=RANDOM_STATE)
        out[f"{h}m"] = {"n_loans": int(mask.sum()), "default_rate": float(y.mean()), **ci}
        logger.info("AUC@%dm on %d loans: lr %.4f, lgbm %.4f", h, mask.sum(),
                    ci["lr"]["roc_auc"]["estimate"], ci["lgbm"]["roc_auc"]["estimate"])
    return out


# --- 2. Calibration vs Aalen-Johansen --------------------------------------------

def aalen_johansen_cif(duration, event, h: int) -> float:
    """Observed P(default by h) accounting for censoring and prepayment.
    NaN if no loan in the group is followed beyond h - 1 months."""
    from lifelines import AalenJohansenFitter

    duration, event = np.asarray(duration), np.asarray(event)
    if len(duration) == 0 or duration.max() < h:
        return float("nan")
    aj = AalenJohansenFitter(calculate_variance=False, seed=RANDOM_STATE)
    aj.fit(duration, event, event_of_interest=EVENT_DEFAULT)
    cd = aj.cumulative_density_.iloc[:, 0]
    return float(cd[cd.index <= h].iloc[-1]) if (cd.index <= h).any() else 0.0


def calibration_table(t: pd.DataFrame, by: pd.Series, horizons) -> pd.DataFrame:
    rows = []
    for g, idx in t.groupby(by).groups.items():
        sub = t.loc[idx]
        for h in horizons:
            obs = aalen_johansen_cif(sub["duration"], sub["event"], h)
            row = {"group": str(g), "horizon": h, "loans": len(sub), "observed": obs}
            for m in MODELS:
                row[f"pred_{m}"] = float(sub[f"cif_default_{h}_{m}"].mean())
                row[f"ratio_{m}"] = row[f"pred_{m}"] / obs if obs and np.isfinite(obs) else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def plot_calibration(by_year: pd.DataFrame, by_grade: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"observed": "#3d3d3a", "lr": "#2a78d6", "lgbm": "#eb6834"}
    labels = {"observed": "observed (Aalen-Johansen)", "lr": "multinomial LR", "lgbm": "LightGBM"}
    markers = {"observed": "s", "lr": "o", "lgbm": "^"}
    panels = [(by_year, 12, "by issue year, 12 months"), (by_year, 24, "by issue year, 24 months"),
              (by_grade, 36, "by grade, 36 months")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (tbl, h, title) in zip(axes, panels):
        d = tbl[(tbl["horizon"] == h) & tbl["observed"].notna()]
        x = np.arange(len(d))
        for key, col in (("observed", "observed"), ("lr", "pred_lr"), ("lgbm", "pred_lgbm")):
            ax.plot(x, d[col], marker=markers[key], markersize=8, lw=2 if key != "observed" else 1.5,
                    color=colors[key], label=labels[key], markeredgecolor="#fcfcfb", markeredgewidth=1.5)
        ax.set_xticks(x); ax.set_xticklabels(d["group"])
        ax.set_title(f"P(default) {title}", loc="left")
        ax.set_facecolor("#fcfcfb"); ax.grid(color="#e6e6e1", lw=0.8); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("cumulative default probability")
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle("Phase 2 calibration on the out-of-time test set (loans issued 2015-2018)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


# --- 3. Apples-to-apples vs Phase 1 ------------------------------------------------

def _weighted_brier_draws(y, probs: dict, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    sq = {k: (np.asarray(p) - y) ** 2 for k, p in probs.items()}
    draws = {k: np.empty(n_boot) for k in probs}
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n, size=n), minlength=n)
        for k in probs:
            draws[k][b] = (w * sq[k]).sum() / n
    return draws


def phase1_comparison(t: pd.DataFrame, features: pd.DataFrame) -> dict:
    """Phase 1 test population: 36-month loans issued 2015-01..2016-03 that
    reached a final status. Phase 1's saved model is re-scored on Phase 2's
    features for exactly these loans; the result must reproduce Phase 1's
    published AUC."""
    p1_metrics = json.load(open(ROOT / "reports" / "artifacts" / "metrics.json"))
    issue = pd.to_datetime(t["issue_d"])
    m = (t["term"] == 36) & (issue <= "2016-03-31") & t["finished"]
    sub = t[m]
    y = (sub["event"] == EVENT_DEFAULT).astype(int).to_numpy()

    out = {"n_loans": int(m.sum()), "default_rate": float(y.mean()), "models": {}}
    for p2_name, p1_file, p1_key in (("lgbm", "lightgbm_tuned.joblib", "lightgbm_tuned"),
                                     ("lr", None, "logistic_regression")):
        if p1_file is None:
            continue
        p1_model = joblib.load(ROOT / "reports" / "artifacts" / p1_file)
        X = features.loc[sub.index.astype(features.index.dtype)]
        p1 = p1_model.predict_proba(X)[:, 1]
        p2 = sub[f"cif_default_36_{p2_name}"].to_numpy()
        reproduced = float(roc_auc_score(y, p1))
        published = p1_metrics["models"][p1_key]["test"]["roc_auc"]
        ci = bootstrap_ci(y, {"phase1": p1, "phase2": p2}, diff=("phase2", "phase1"),
                          n_boot=N_BOOTSTRAP, seed=RANDOM_STATE)
        bd = _weighted_brier_draws(y, {"phase1": p1, "phase2": p2}, N_BOOTSTRAP, RANDOM_STATE)
        brier = {k: float(brier_score_loss(y, p)) for k, p in (("phase1", p1), ("phase2", p2))}
        diff = bd["phase2"] - bd["phase1"]
        out["models"][p2_name] = {
            "phase1_auc_reproduced": reproduced,
            "phase1_auc_published": published,
            "auc": ci,
            "brier": {**brier, "phase2 - phase1": {
                "estimate": brier["phase2"] - brier["phase1"],
                "ci_low": float(np.percentile(diff, 2.5)), "ci_high": float(np.percentile(diff, 97.5))}},
            "ece": {"phase1": expected_calibration_error(y, p1), "phase2": expected_calibration_error(y, p2)},
            "mean_predicted": {"phase1": float(p1.mean()), "phase2": float(p2.mean())},
        }
        logger.info("Phase 1 comparison (%s): reproduced AUC %.4f (published %.4f), Phase 2 AUC %.4f",
                    p2_name, reproduced, published, ci["phase2"]["roc_auc"]["estimate"])
    return out


# --- 4. Example curves -------------------------------------------------------------

def plot_cif_examples(t: pd.DataFrame, cif: np.ndarray, path) -> dict:
    """For each term and grade band, the loan with the median LightGBM
    CIF_default at the end of its term."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands = t["grade"].map(GRADE_BANDS).to_numpy()
    terms = t["term"].to_numpy()
    picked = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, term in zip(axes, (36, 60)):
        n_q = term // PERIOD_MONTHS
        months = np.arange(0, n_q + 1) * PERIOD_MONTHS
        for band, color in BAND_COLORS.items():
            pos = np.flatnonzero((terms == term) & (bands == band))
            if not len(pos):
                continue
            end = cif[pos, n_q - 1]
            i = pos[np.argsort(end)[len(end) // 2]]
            curve = np.r_[0.0, cif[i, :n_q]]
            ax.plot(months, curve, color=color, lw=2, marker="o", markersize=4, label=band)
            ax.text(months[-1] + 1, curve[-1], band, color="#3d3d3a", va="center", fontsize=10)
            picked[f"{term}m_{band}"] = {"id": str(t.index[i]), "cif_end_of_term": float(curve[-1])}
        ax.set_title(f"{term}-month loans", loc="left"); ax.set_xlabel("Months since issue")
        ax.set_xlim(0, term + 6); ax.set_facecolor("#fcfcfb")
        ax.grid(color="#e6e6e1", lw=0.8); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Predicted cumulative default probability")
    axes[0].legend(frameon=False, title="grade", fontsize=9, loc="upper left")
    fig.suptitle("Predicted default curves (LightGBM): median test loan per grade", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return picked


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t = pd.read_csv(ART / "test_loans.csv.gz", index_col="id")
    meta = json.load(open(ART / "train_meta.json"))
    if meta.get("quick"):
        logger.warning("train_meta.json comes from a --quick run; numbers are not final")

    metrics = {"quick": meta.get("quick", False), "n_test_loans": int(len(t))}
    metrics["horizon_auc"] = horizon_auc(t)

    year = pd.to_datetime(t["issue_d"]).dt.year
    by_year = calibration_table(t, year, horizons=(12, 24, 36))
    by_grade = calibration_table(t, t["grade"].map(GRADE_BANDS), horizons=(12, 36))
    metrics["calibration_by_issue_year"] = by_year.to_dict(orient="records")
    metrics["calibration_by_grade"] = by_grade.to_dict(orient="records")
    recent = by_year[by_year["group"].astype(int).isin(RECENT_YEARS) & by_year["observed"].notna()]
    metrics["recent_cohorts_2016_2018"] = recent.to_dict(orient="records")
    plot_calibration(by_year, by_grade, FIG / "survival_calibration.png")

    features, _ = load_survival_data()
    features = engineer_features(features)
    metrics["phase1_comparison"] = phase1_comparison(t, features)
    del features

    h = np.load(ART / "hazards_lgbm.npy")
    full = np.concatenate([np.zeros(h.shape[:2] + (1,), np.float32), h], axis=2)
    cif = cumulative_incidence(np.where(np.isnan(full), np.nan, full))["cif_default"]
    metrics["cif_examples"] = plot_cif_examples(t, cif, FIG / "cif_examples.png")

    json.dump(metrics, open(ART / "metrics.json", "w"), indent=2, default=float)
    print(json.dumps({k: metrics[k] for k in ("horizon_auc",)}, indent=1, default=float)[:3000])
    print(recent.round(4).to_string())
    print(json.dumps(metrics["phase1_comparison"], indent=1, default=float))
    print(f"saved {ART / 'metrics.json'}, figures survival_calibration.png and cif_examples.png")


if __name__ == "__main__":
    main()
