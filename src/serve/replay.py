"""Replay the 2015-2018 test loans through the API, month by month.

    python -m src.serve.replay          (make replay; needs data/archive/)

For each simulated month (by issue date) from 2015-01 to 2018-12:
  1. score that month's loans through POST /v1/score/batch (logged to a
     fresh SQLite database),
  2. post the outcomes that would be known by then: a loan's 12-month
     outcome becomes available 12 months after issue (Phase 2 outcome
     table with Late (31-120 days) loans counted as defaults),
  3. run the monitor and record when each alert first fires.

Run twice, raw and calibrated. Writes reports/monitoring/replay_summary.json
and reports/figures/replay_*.png.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..survival_data import EVENT_DEFAULT, EVENT_PREPAY, load_survival_data
from ..train_survival import split_masks
from .monitoring import CAL_HIGH, CAL_LOW, HORIZON_MONTHS, PSI_ALERT, monitoring_summary
from .scoring import INPUT_FIELDS, LoanApplication

logger = logging.getLogger("src.serve.replay")

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "reports" / "monitoring"
FIG = ROOT / "reports" / "figures"
MAX_REPLAY_LOANS = int(os.environ.get("MAX_REPLAY_LOANS", 50_000))
BATCH = 1_000
N_LATENCY_PROBES = 300
SEED = 42


def load_replay_loans(max_loans: int = MAX_REPLAY_LOANS) -> pd.DataFrame:
    features, outcomes = load_survival_data(arrears_as_default=True)
    test = split_masks(features)["test"]
    f, o = features[test], outcomes[test]
    if len(f) > max_loans:
        idx = f.sample(n=max_loans, random_state=SEED).index
        f, o = f.loc[idx], o.loc[idx]
    df = f.assign(duration=o["duration"], event=o["event"], finished=o["finished"])
    df["month"] = df["issue_d"].dt.strftime("%Y-%m")
    df.index = df.index.astype(str)
    return df.sort_values("issue_d", kind="mergesort")


def to_application(loan_id: str, row: pd.Series) -> dict:
    d = {}
    for f in INPUT_FIELDS:
        v = row[f]
        d[f] = None if (isinstance(v, float) and np.isnan(v)) else (int(v) if f in ("term", "emp_length") else
                                                                    float(v) if not isinstance(v, str) else v)
    d["loan_id"], d["issue_d"] = loan_id, row["month"]
    return d


def outcome_at_12m(loan_id: str, row: pd.Series) -> dict:
    """The loan's status 12 months after issue, as the lender would know it."""
    d, e = int(row["duration"]), int(row["event"])
    if e == EVENT_DEFAULT and d <= HORIZON_MONTHS:
        return {"loan_id": loan_id, "event": "default", "duration_months": d}
    if e == EVENT_PREPAY and d <= HORIZON_MONTHS:
        return {"loan_id": loan_id, "event": "prepay", "duration_months": d}
    if d >= HORIZON_MONTHS or row["finished"]:
        # still active at 12 months; nothing later is known yet
        return {"loan_id": loan_id, "event": "censored", "duration_months": HORIZON_MONTHS}
    return {"loan_id": loan_id, "event": "censored", "duration_months": d}       # unknown at 12m


def _month_add(month: str, k: int) -> str:
    return str(pd.Period(month, freq="M") + k)


def replay(loans: pd.DataFrame, calibrated: bool) -> dict:
    from fastapi.testclient import TestClient
    from .app import app, state

    tmp = tempfile.mkdtemp(prefix="replay-")
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp}/replay.db"          # start from an empty database
    q = f"?calibrated={'true' if calibrated else 'false'}"
    months = sorted(loans["month"].unique())
    timeline, first_alert, rejected = [], {}, 0

    apps_by_month, rows_by_month = {}, {}
    for m, g in loans.groupby("month"):
        ok, ok_rows = [], []
        for lid, row in g.iterrows():
            a = to_application(lid, row)
            try:
                LoanApplication(**a)
                ok.append(a); ok_rows.append(lid)
            except ValidationError:
                rejected += 1
        apps_by_month[m], rows_by_month[m] = ok, ok_rows

    with TestClient(app) as client:
        t_start = time.time()
        for m in months:
            apps = apps_by_month[m]
            for i in range(0, len(apps), BATCH):
                r = client.post("/v1/score/batch" + q, json={"applications": apps[i:i + BATCH]})
                r.raise_for_status()
            matured = _month_add(m, -HORIZON_MONTHS)            # its 12-month outcomes are known now
            if matured in rows_by_month and rows_by_month[matured]:
                outs = [outcome_at_12m(lid, loans.loc[lid]) for lid in rows_by_month[matured]]
                for i in range(0, len(outs), 10_000):
                    client.post("/v1/outcomes", json={"outcomes": outs[i:i + 10_000]}).raise_for_status()
            s = monitoring_summary(state.Session, state.bundle, month_from=m, month_to=m)
            cal = s["calibration"]
            entry = {"month": m, "n_scored": s["window"]["n"],
                     "psi": {d["feature"]: d["psi"] for d in s["input_drift"]},
                     "calibration_status": cal["status"], "calibration_loans": cal["n_loans"],
                     "pooled_ratio": (cal["pooled"] or {}).get("ratio"),
                     "outcome_months": sum(1 for mm in months if mm <= matured)}
            timeline.append(entry)
            if cal["status"] == "alert" and "calibration" not in first_alert:
                first_alert["calibration"] = {"month": m, "outcome_months": entry["outcome_months"],
                                              "loans_with_outcomes": cal["n_loans"],
                                              "ratio": entry["pooled_ratio"]}
            for d in s["input_drift"]:
                if d["psi"] > PSI_ALERT and f"psi_{d['feature']}" not in first_alert:
                    first_alert[f"psi_{d['feature']}"] = {"month": m, "psi": d["psi"]}
            logger.info("%s %s: scored %d, cal %s ratio %s (%d loans)", "calibrated" if calibrated else "raw",
                        m, entry["n_scored"], cal["status"],
                        f"{entry['pooled_ratio']:.3f}" if entry["pooled_ratio"] else "-", cal["n_loans"])
        elapsed = time.time() - t_start

        final = monitoring_summary(state.Session, state.bundle)
        # per-request latency: single-loan calls through the full HTTP stack
        probe = [a for m in months for a in apps_by_month[m]]
        rng = np.random.default_rng(SEED)
        lat = []
        for a in (probe[i] for i in rng.choice(len(probe), size=min(N_LATENCY_PROBES, len(probe)), replace=False)):
            t0 = time.perf_counter()
            client.post("/v1/score" + q, json={**a, "loan_id": None}).raise_for_status()
            lat.append((time.perf_counter() - t0) * 1000)

    return {
        "calibrated": calibrated,
        "loans_scored": int(final["n_predictions"]) - len(lat),
        "loans_rejected_by_validation": rejected,
        "first_alerts": first_alert,
        "final_calibration": final["calibration"]["pooled"],
        "final_calibration_by_month": final["calibration"]["by_month"],
        "psi_full_period": final["input_drift"],
        "latency_ms_single_request": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                                      "n": len(lat)},
        "throughput_loans_per_s": float((int(final["n_predictions"]) - len(lat)) / elapsed),
        "timeline": timeline,
    }


def plot_replay(runs: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, grid = "#fcfcfb", "#e6e6e1"
    colors = {"raw": "#3d3d3a", "calibrated": "#2a78d6"}
    fig, ax = plt.subplots(figsize=(10, 4.8)); fig.patch.set_facecolor(surface); ax.set_facecolor(surface)
    ax.axhspan(CAL_LOW, CAL_HIGH, color="#1baf7a", alpha=0.10, lw=0, label=f"no-alert band {CAL_LOW}-{CAL_HIGH}")
    ax.axhline(1.0, color="#8a8a85", lw=1, ls="--")
    for name, run in runs.items():
        tl = pd.DataFrame(run["timeline"]).dropna(subset=["pooled_ratio"])
        x = pd.to_datetime(tl["month"])
        ax.plot(x, tl["pooled_ratio"], color=colors[name], lw=2, label=f"{name} model")
        fa = run["first_alerts"].get("calibration")
        if fa:
            xa = pd.to_datetime(fa["month"])
            ax.scatter([xa], [fa["ratio"]], color="#eb6834", s=70, zorder=5, edgecolor=surface, linewidth=1.5)
            ax.annotate(f"{name}: alert {fa['month']}\n({fa['outcome_months']} months of outcomes)",
                        (xa, fa["ratio"]), textcoords="offset points", xytext=(10, -28 if name == "raw" else 12),
                        fontsize=9, color="#3d3d3a")
    ax.set_ylabel("pooled predicted / observed\n12-month default rate")
    ax.set_title("Replay: calibration monitor as 12-month outcomes arrive (simulated month)", loc="left")
    ax.grid(color=grid); ax.legend(frameon=False, fontsize=9, loc="upper right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "replay_calibration.png", dpi=120, facecolor=surface); plt.close(fig)

    run = runs["raw"]
    psi = pd.DataFrame([{"month": t["month"], **t["psi"]} for t in run["timeline"]]).set_index("month")
    top = [d["feature"] for d in run["psi_full_period"][:4]]
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, ax = plt.subplots(figsize=(10, 4.5)); fig.patch.set_facecolor(surface); ax.set_facecolor(surface)
    for f, c in zip(top, palette):
        ax.plot(pd.to_datetime(psi.index), psi[f], color=c, lw=2, label=f)
    ax.axhline(0.10, color="#8a8a85", ls=":", lw=1, label="watch (0.10)")
    ax.axhline(PSI_ALERT, color="#3d3d3a", ls="--", lw=1, label="alert (0.25)")
    ax.set_ylabel("PSI vs 2007-2013 training reference")
    ax.set_title("Replay: monthly input drift, top features over the whole period", loc="left")
    ax.grid(color=grid); ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "replay_drift.png", dpi=120, facecolor=surface); plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    loans = load_replay_loans()
    logger.info("replaying %d loans, %s to %s", len(loans), loans["month"].min(), loans["month"].max())
    gc.collect()
    runs = {"raw": replay(loans, calibrated=False), "calibrated": replay(loans, calibrated=True)}
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "max_replay_loans": MAX_REPLAY_LOANS,
        "loans_sampled": int(len(loans)),
        "months": [loans["month"].min(), loans["month"].max()],
        "outcome_rule": ("12-month outcome posted 12 months after issue; Late (31-120 days) counted as default"),
        "alert_rule": {"calibration_bounds": [CAL_LOW, CAL_HIGH], "psi_alert": PSI_ALERT},
        **{name: {k: v for k, v in run.items()} for name, run in runs.items()},
    }
    json.dump(summary, open(OUT / "replay_summary.json", "w"), indent=2, default=str)
    plot_replay(runs)
    for name, run in runs.items():
        fc = run["final_calibration"]
        print(f"{name}: first calibration alert {run['first_alerts'].get('calibration')}; "
              f"final pooled ratio {fc['ratio']:.3f} ({fc['n']} loans); "
              f"latency p50 {run['latency_ms_single_request']['p50']:.1f} ms, "
              f"p95 {run['latency_ms_single_request']['p95']:.1f} ms")
    print("top drift (full period):", [(d["feature"], round(d["psi"], 3)) for d in runs["raw"]["psi_full_period"][:5]])
    print(f"saved {OUT / 'replay_summary.json'} and reports/figures/replay_*.png")


if __name__ == "__main__":
    main()
