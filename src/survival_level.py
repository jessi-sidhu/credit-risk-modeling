"""Can the default LEVEL be fixed? Experiments on the Phase 2 hazard model.

    python -m src.survival_level          (after python -m src.train_survival)

Phase 2 ranks loans as well as Phase 1 but predicts only ~70-87% of the
defaults that happened in the 2015-16 cohorts. Three ways to move the level,
each fit under the same out-of-time protocol and scored on the same test
loans as the Phase 2 base model:

    recalibrated  base model + log-odds shifts of the default and prepay
                  hazards, fit on the 2014 validation loan-quarters
    vintage       issue date as a feature (years since 2007); a linear
                  model can extrapolate the cohort trend, trees cannot
    macro         US unemployment in each loan-quarter's calendar month
                  (FRED UNRATE). Test loans use the REALIZED path, i.e.
                  perfect foresight: an upper bound for a macro scenario

Metrics per variant and model: predicted / observed cumulative default
probability by issue year (observed = Aalen-Johansen, with and without
Late (31-120 days) loans counted as defaults), horizon AUC (point
estimates), the Phase 1 population's mean CIF(36) vs its default rate,
and predicted / realized lifetime loss on matured test loans.

Writes reports/artifacts/survival/level_experiments.json and
reports/figures/level_experiments.png.
"""
from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from .ecl import (
    discount_factors,
    ead_schedule,
    expected_loss,
    lgd_by_grade,
    read_recovery_columns,
    realized_loss,
)
from .features import engineer_features
from .survival import (
    HazardModel,
    apply_level_offsets,
    cif_at,
    cumulative_incidence,
    expand_person_period,
    fit_level_offsets,
)
from .survival_data import EVENT_DEFAULT, load_survival_data
from .survival_eval import aalen_johansen_cif, known_status_at
from .train_survival import split_masks

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "reports" / "artifacts" / "survival"
FIG = ROOT / "reports" / "figures"
VARIANTS = ("base", "recalibrated", "vintage", "macro")
TIME_FEATURES = {"base": (), "vintage": ("vintage",), "macro": ("unrate",)}
# (issue year, horizon in months) cells with an observable Aalen-Johansen estimate
CELLS = [(2015, 36), (2016, 36), (2016, 12), (2017, 12), (2017, 24), (2018, 12)]
COMPLETE_CELLS = [(2015, 36), (2016, 36)]          # outcomes essentially complete


def _test_frame(features, outcomes, alt) -> pd.DataFrame:
    return pd.DataFrame({
        "issue_d": features["issue_d"], "term": features["term"], "grade": features["grade"],
        "int_rate": features["int_rate"], "loan_amount": features["loan_amount"],
        "duration": outcomes["duration"], "event": outcomes["event"], "finished": outcomes["finished"],
        "duration_alt": alt.loc[features.index, "duration"],
        "event_alt": alt.loc[features.index, "event"],
        "finished_alt": alt.loc[features.index, "finished"],
    }, index=features.index)


def observed_cells(t: pd.DataFrame) -> dict:
    year = t["issue_d"].dt.year
    out = {}
    for y, h in CELLS:
        m = (year == y).to_numpy()
        out[f"{y}@{h}m"] = {
            "observed": aalen_johansen_cif(t.loc[m, "duration"], t.loc[m, "event"], h),
            "observed_arrears_as_default": aalen_johansen_cif(t.loc[m, "duration_alt"], t.loc[m, "event_alt"], h),
        }
    return out


def evaluate(h: np.ndarray, t: pd.DataFrame, obs: dict, ecl_inputs: dict) -> dict:
    curves = cumulative_incidence(h)
    cif, mpd = curves["cif_default"], curves["marginal_pd_default"]
    year = t["issue_d"].dt.year.to_numpy()
    res = {"cells": {}}
    for y, hz in CELLS:
        key = f"{y}@{hz}m"
        pred = float(cif_at(cif, hz)[year == y].mean())
        o, oa = obs[key]["observed"], obs[key]["observed_arrears_as_default"]
        res["cells"][key] = {"predicted": pred, "ratio": pred / o, "ratio_arrears_as_default": pred / oa}
    for hz in (12, 36):
        mask, y = known_status_at(t, hz)
        res[f"auc_{hz}m"] = float(roc_auc_score(y, cif_at(cif, hz)[mask]))
    p1 = ecl_inputs["phase1_mask"]
    p1_rate = float((t.loc[p1, "event"] == EVENT_DEFAULT).mean())
    res["phase1_population"] = {"mean_cif36": float(cif_at(cif, 36)[p1].mean()), "default_rate": p1_rate}
    res["phase1_population"]["ratio"] = res["phase1_population"]["mean_cif36"] / p1_rate
    mat = ecl_inputs["matured_mask"]
    el = expected_loss(mpd[mat], ecl_inputs["ead"], ecl_inputs["lgd"], ecl_inputs["disc"])
    predicted = float(el["el_lifetime_undiscounted"].sum())
    res["matured_loss"] = {"predicted": predicted, "realized": ecl_inputs["realized"],
                           "ratio": predicted / ecl_inputs["realized"]}
    return res


def plot_ratios(results: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"base": "#3d3d3a", "recalibrated": "#2a78d6", "vintage": "#eb6834", "macro": "#1baf7a"}
    markers = {"base": "s", "recalibrated": "o", "vintage": "^", "macro": "D"}
    labels = [k for k in results["lgbm"]["base"]["cells"]]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (model, title) in zip(axes, (("lr", "Multinomial LR"), ("lgbm", "LightGBM"))):
        ax.set_facecolor("#fcfcfb")
        ax.axhline(1.0, color="#8a8a85", lw=1, ls="--")
        ax.axvspan(1.5, len(labels) - 0.5, color="#e6e6e1", alpha=0.5, lw=0)
        ax.text(len(labels) - 0.55, 0.9, "recent cohorts: outcomes still incomplete",
                transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=9, color="#3d3d3a")
        for v in VARIANTS:
            # observed with Late (31-120 days) counted as default, so recent cohorts
            # aren't flattered by charge-offs that hadn't been recorded yet
            ratios = [results[model][v]["cells"][k]["ratio_arrears_as_default"] for k in labels]
            ax.plot(x, ratios, marker=markers[v], markersize=8, lw=2, color=colors[v], label=v,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5)
        ax.set_xticks(x); ax.set_xticklabels([l.replace("@", "\n") for l in labels])
        ax.set_title(title, loc="left")
        ax.grid(color="#e6e6e1", lw=0.8); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("predicted / observed P(default)\n(Late 31-120 days counted as default)")
    axes[1].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("Level experiments: predicted / observed by issue year and horizon (1.0 = calibrated)",
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="20k train / 20k val / 60k test loans")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    meta = json.load(open(ART / "train_meta.json"))
    lgbm_params = {k: v for k, v in meta["lgbm"]["best_params"].items() if k != "n_estimators"}

    features, outcomes = load_survival_data()
    _, alt = load_survival_data(arrears_as_default=True)
    features = engineer_features(features)
    for df in (features, outcomes, alt):
        df.index = df.index.astype(str)
    masks = split_masks(features)
    parts = {k: (features[m], outcomes[m]) for k, m in masks.items()}
    if args.quick:
        from .survival import sample_loans
        parts = {k: sample_loans(*v, {"train": 20_000, "val": 20_000, "test": 60_000}[k]) for k, v in parts.items()}
        masks = {k: features.index.isin(parts[k][0].index) for k in masks}
    t = _test_frame(*parts["test"], alt)
    end_train = features.loc[masks["train"], "issue_d"].max()

    # Expected-loss inputs (same definitions as src/ecl.py)
    rec = read_recovery_columns()
    rec.index = rec.index.astype(str)
    train_def = masks["train"] & (outcomes["event"] == EVENT_DEFAULT).to_numpy()
    td = rec.loc[features.index[train_def]].assign(grade=features.loc[train_def, "grade"].to_numpy())
    lgd = lgd_by_grade(td)
    matured = ((t["term"] == 36) & (t["issue_d"] <= "2016-03-31")).to_numpy()
    tm = t[matured]
    realized = float(np.where(tm["event"] == EVENT_DEFAULT, realized_loss(rec.loc[tm.index]), 0.0).sum())
    ecl_inputs = {
        "matured_mask": matured,
        "phase1_mask": matured & t["finished"].to_numpy(),
        "ead": ead_schedule(tm["loan_amount"], tm["int_rate"], tm["term"], 20),
        "disc": discount_factors(tm["int_rate"], 20),
        "lgd": tm["grade"].map(lgd["lgd"]).fillna(lgd["lgd"].mean()).to_numpy(),
        "realized": realized,
    }
    del features, outcomes, alt, rec; gc.collect()

    obs = observed_cells(t)
    logger.info("observed cells: %s", {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in obs.items()})
    pp_tr = expand_person_period(*parts["train"])
    pp_va = expand_person_period(*parts["val"])
    f_te = parts["test"][0]

    results = {"lr": {}, "lgbm": {}}
    details = {"lr": {}, "lgbm": {}}
    for model_kind in ("lr", "lgbm"):
        params = lgbm_params if model_kind == "lgbm" else {}
        for variant in ("base", "vintage", "macro"):
            t0 = time.time()
            m = HazardModel(model_kind, time_features=TIME_FEATURES[variant], **params)
            m.fit(pp_tr, eval_pp=pp_va if model_kind == "lgbm" else None)
            p_va = m.predict_rows(pp_va)
            loss = float(log_loss(pp_va["y"], p_va, labels=[0, 1, 2]))
            details[model_kind][variant] = {"val_logloss": loss, "best_iteration": m.best_iteration_}
            h = m.hazards(f_te)
            results[model_kind][variant] = evaluate(h, t, obs, ecl_inputs)
            logger.info("%s/%s: val logloss %.5f, %.0fs, cells %s", model_kind, variant, loss,
                        time.time() - t0,
                        {k: round(v["ratio"], 3) for k, v in results[model_kind][variant]["cells"].items()})
            if variant == "base":
                ref = meta["lr"]["val_logloss"] if model_kind == "lr" else meta["lgbm"]["val_logloss"]
                details[model_kind]["base"]["matches_phase2_run"] = bool(abs(loss - ref) < 1e-6)
                offsets = fit_level_offsets(p_va, pp_va["y"].to_numpy())
                adj_va = apply_level_offsets(p_va[None, :, :], offsets)[0]
                details[model_kind]["recalibrated"] = {
                    "offsets_log_odds": {"default": float(offsets[0]), "prepay": float(offsets[1])},
                    "val_logloss": float(log_loss(pp_va["y"], adj_va, labels=[0, 1, 2])),
                }
                results[model_kind]["recalibrated"] = evaluate(apply_level_offsets(h, offsets), t, obs, ecl_inputs)
                logger.info("%s/recalibrated: offsets %s, cells %s", model_kind, np.round(offsets, 4),
                            {k: round(v["ratio"], 3) for k, v in results[model_kind]["recalibrated"]["cells"].items()})
            del h, m, p_va; gc.collect()

    out = {
        "quick": args.quick,
        "note": ("macro uses the realized unemployment path for test loans (perfect foresight); "
                 "AUCs are point estimates"),
        "train_until": str(end_train.date()),
        "observed": obs,
        "complete_cells": [f"{y}@{h}m" for y, h in COMPLETE_CELLS],
        "results": results,
        "fit": details,
    }
    json.dump(out, open(ART / "level_experiments.json", "w"), indent=2)
    plot_ratios(results, FIG / "level_experiments.png")
    for model_kind in ("lr", "lgbm"):
        rows = []
        for v in VARIANTS:
            r = results[model_kind][v]
            rows.append({"variant": v, **{k: round(c["ratio"], 3) for k, c in r["cells"].items()},
                         "auc12": round(r["auc_12m"], 4), "auc36": round(r["auc_36m"], 4),
                         "p1_ratio": round(r["phase1_population"]["ratio"], 3),
                         "loss_ratio": round(r["matured_loss"]["ratio"], 3)})
        print(model_kind); print(pd.DataFrame(rows).set_index("variant").to_string())
    print(json.dumps(details, indent=1))
    print(f"saved {ART / 'level_experiments.json'} and reports/figures/level_experiments.png")


if __name__ == "__main__":
    main()
