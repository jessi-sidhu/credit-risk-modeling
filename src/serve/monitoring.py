"""Monitoring for the scoring service.

    python -m src.serve.monitoring       (make monitor-report)
    GET /v1/monitoring

1. Input drift: PSI of each input in the logged requests vs the training
   reference distributions in the bundle (reference_stats.json), using the
   binning and smoothing of src/drift.py. PSI > 0.10 = watch, > 0.25 = alert.
2. Score drift: distribution of pd_12m by month.
3. Calibration: for loans whose 12-month outcome is known, mean predicted
   pd_12m vs the observed 12-month default rate, by issue month and pooled.
   Alert when the pooled predicted / observed ratio leaves [0.85, 1.15]
   (once at least MIN_CALIBRATION_LOANS loans have outcomes). Clients
   should report loans more than 30 days past due as `default`, matching
   Phase 2's arrears_as_default finding.

"Month" is the loan's issue month when the client sent `issue_d`,
otherwise the month the request was received.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select

from ..drift import _EPS, _hist
from .bundle import ModelBundle
from .db import Outcome, Prediction

PSI_WATCH, PSI_ALERT = 0.10, 0.25
CAL_LOW, CAL_HIGH = 0.85, 1.15
MIN_CALIBRATION_LOANS = 1_000
HORIZON_MONTHS = 12
REPORT_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "monitoring"


# --- data access ---------------------------------------------------------------------

def load_predictions(Session, calibrated: Optional[bool] = None) -> pd.DataFrame:
    with Session() as s:
        q = select(Prediction)
        if calibrated is not None:
            q = q.where(Prediction.calibrated == calibrated)
        rows = s.scalars(q).all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([{**r.features, "request_id": r.request_id, "loan_id_logged": r.loan_id,
                        "received_at": r.received_at, "pd_12m": r.pd_12m, "pd_lifetime": r.pd_lifetime,
                        "calibrated": r.calibrated, "latency_ms": r.latency_ms} for r in rows])
    received = pd.to_datetime(df["received_at"], utc=True).dt.strftime("%Y-%m")
    df["month"] = df["issue_d"].fillna(received) if "issue_d" in df else received
    return df


def load_outcomes(Session) -> pd.DataFrame:
    """Latest reported outcome per loan."""
    with Session() as s:
        rows = s.scalars(select(Outcome)).all()
    if not rows:
        return pd.DataFrame(columns=["loan_id", "event", "duration_months"])
    df = pd.DataFrame([{"loan_id": r.loan_id, "observed_at": r.observed_at, "event": r.event,
                        "duration_months": r.duration_months} for r in rows])
    return df.sort_values("observed_at").groupby("loan_id").tail(1)


# --- 1. input drift --------------------------------------------------------------------

def _status(psi: float) -> str:
    return "alert" if psi > PSI_ALERT else "watch" if psi > PSI_WATCH else "ok"


def psi_vs_reference(values: pd.Series, ref: dict) -> float:
    if ref["kind"] == "numeric":
        p = np.asarray(ref["shares"], float)
        q = _hist(values.to_numpy(dtype=float), np.asarray(ref["edges"], float))
    else:
        cur = values.astype(str).value_counts(normalize=True)
        cats = sorted(set(ref["shares"]) | set(cur.index))
        p = np.array([ref["shares"].get(c, 0.0) for c in cats])
        q = np.array([cur.get(c, 0.0) for c in cats])
        p, q = np.where(p == 0, _EPS, p), np.where(q == 0, _EPS, q)
    return float(np.sum((q - p) * np.log(q / p)))


def input_drift(preds: pd.DataFrame, reference: dict) -> list[dict]:
    rows = []
    for name, ref in reference["features"].items():
        if name not in preds:
            continue
        v = psi_vs_reference(preds[name], ref)
        rows.append({"feature": name, "psi": v, "status": _status(v)})
    return sorted(rows, key=lambda r: -r["psi"])


# --- 2. score drift ----------------------------------------------------------------------

def score_drift(preds: pd.DataFrame) -> list[dict]:
    g = preds.groupby("month")["pd_12m"]
    out = pd.DataFrame({"n": g.size(), "mean": g.mean(), "p10": g.quantile(0.1),
                        "p50": g.median(), "p90": g.quantile(0.9)}).reset_index()
    return out.to_dict(orient="records")


# --- 3. calibration ----------------------------------------------------------------------

def twelve_month_labels(outcomes: pd.DataFrame) -> pd.DataFrame:
    """1 = defaulted within 12 months; 0 = known not to have (observed past
    12 months, or prepaid / paid off without default by then). Loans with
    shorter follow-up and no default are dropped (status unknown)."""
    d, e = outcomes["duration_months"], outcomes["event"]
    pos = (e == "default") & (d <= HORIZON_MONTHS)
    neg = ((d >= HORIZON_MONTHS) & ~pos) | ((e == "prepay") & (d <= HORIZON_MONTHS))
    known = outcomes[pos | neg].copy()
    known["default_12m"] = pos[pos | neg].astype(int)
    return known[["loan_id", "default_12m"]]


def calibration(preds: pd.DataFrame, outcomes: pd.DataFrame) -> dict:
    labels = twelve_month_labels(outcomes)
    m = preds.dropna(subset=["loan_id_logged"]).merge(labels, left_on="loan_id_logged", right_on="loan_id")
    if m.empty:
        return {"n_loans": 0, "by_month": [], "pooled": None, "status": "no outcomes yet"}
    g = m.groupby("month")
    by_month = pd.DataFrame({"n": g.size(), "predicted": g["pd_12m"].mean(),
                             "observed": g["default_12m"].mean()}).reset_index()
    by_month["ratio"] = by_month["predicted"] / by_month["observed"].replace(0, np.nan)
    pooled = {"n": int(len(m)), "predicted": float(m["pd_12m"].mean()),
              "observed": float(m["default_12m"].mean())}
    pooled["ratio"] = pooled["predicted"] / pooled["observed"] if pooled["observed"] > 0 else None
    if len(m) < MIN_CALIBRATION_LOANS or pooled["ratio"] is None:
        status = "insufficient outcomes"
    else:
        status = "alert" if not CAL_LOW <= pooled["ratio"] <= CAL_HIGH else "ok"
    return {"n_loans": int(len(m)), "by_month": by_month.to_dict(orient="records"),
            "pooled": pooled, "status": status, "bounds": [CAL_LOW, CAL_HIGH],
            "min_loans": MIN_CALIBRATION_LOANS}


# --- report --------------------------------------------------------------------------------

def monitoring_summary(Session, bundle: ModelBundle, month_from: Optional[str] = None,
                       month_to: Optional[str] = None, calibrated: Optional[bool] = None) -> dict:
    preds = load_predictions(Session, calibrated=calibrated)
    if preds.empty:
        return {"n_predictions": 0, "status": "no predictions logged"}
    window = preds
    if month_from:
        window = window[window["month"] >= month_from]
    if month_to:
        window = window[window["month"] <= month_to]
    drift = input_drift(window, bundle.reference) if len(window) else []
    cal = calibration(preds[preds["month"] <= month_to] if month_to else preds, load_outcomes(Session))
    alerts = [f"PSI {d['feature']} = {d['psi']:.3f}" for d in drift if d["status"] == "alert"]
    if cal["status"] == "alert":
        alerts.append(f"calibration ratio {cal['pooled']['ratio']:.3f} outside [{CAL_LOW}, {CAL_HIGH}]")
    return {
        "model_version": bundle.version,
        "n_predictions": int(len(preds)),
        "window": {"month_from": month_from, "month_to": month_to, "n": int(len(window))},
        "input_drift": drift,
        "score_drift": score_drift(window) if len(window) else [],
        "calibration": cal,
        "latency_ms": {"p50": float(preds["latency_ms"].median()), "p95": float(preds["latency_ms"].quantile(0.95))},
        "alerts": alerts,
        "status": "alert" if alerts else "ok",
    }


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    return base64.b64encode(buf.getvalue()).decode()


def report_figures(summary: dict) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = {}
    surface, grid = "#fcfcfb", "#e6e6e1"
    drift = pd.DataFrame(summary["input_drift"])
    if len(drift):
        fig, ax = plt.subplots(figsize=(7, 4.5)); fig.patch.set_facecolor(surface); ax.set_facecolor(surface)
        colors = drift["status"].map({"alert": "#eb6834", "watch": "#eda100", "ok": "#2a78d6"})
        ax.barh(drift["feature"][::-1], drift["psi"][::-1], color=colors[::-1])
        ax.axvline(PSI_WATCH, color="#8a8a85", ls=":", label="watch (0.10)")
        ax.axvline(PSI_ALERT, color="#3d3d3a", ls="--", label="alert (0.25)")
        ax.set_xlabel("PSI vs training reference"); ax.set_title("Input drift", loc="left")
        ax.legend(frameon=False, fontsize=8); ax.grid(axis="x", color=grid)
        figs["drift"] = _fig_to_base64(fig); plt.close(fig)
    sd = pd.DataFrame(summary["score_drift"])
    if len(sd):
        fig, ax = plt.subplots(figsize=(8, 4)); fig.patch.set_facecolor(surface); ax.set_facecolor(surface)
        x = np.arange(len(sd))
        ax.fill_between(x, sd["p10"], sd["p90"], color="#2a78d6", alpha=0.18, lw=0, label="p10-p90")
        ax.plot(x, sd["p50"], color="#2a78d6", lw=2, label="median")
        ax.plot(x, sd["mean"], color="#2a78d6", lw=1.5, ls="--", label="mean")
        step = max(1, len(sd) // 12)
        ax.set_xticks(x[::step]); ax.set_xticklabels(sd["month"][::step], rotation=45, ha="right")
        ax.set_ylabel("predicted 12-month PD"); ax.set_title("Score distribution by month", loc="left")
        ax.legend(frameon=False, fontsize=8); ax.grid(color=grid)
        figs["scores"] = _fig_to_base64(fig); plt.close(fig)
    cal = pd.DataFrame(summary["calibration"].get("by_month", []))
    if len(cal):
        fig, ax = plt.subplots(figsize=(8, 4)); fig.patch.set_facecolor(surface); ax.set_facecolor(surface)
        x = np.arange(len(cal))
        ax.plot(x, cal["observed"], color="#3d3d3a", marker="s", lw=1.5, label="observed 12-month default rate")
        ax.plot(x, cal["predicted"], color="#eb6834", marker="o", lw=2, label="mean predicted pd_12m")
        step = max(1, len(cal) // 12)
        ax.set_xticks(x[::step]); ax.set_xticklabels(cal["month"][::step], rotation=45, ha="right")
        ax.set_ylim(bottom=0); ax.set_title("Calibration by issue month (12-month outcomes)", loc="left")
        ax.legend(frameon=False, fontsize=8); ax.grid(color=grid)
        figs["calibration"] = _fig_to_base64(fig); plt.close(fig)
    return figs


def write_report(summary: dict, out_dir: Path = REPORT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2, default=str)
    figs = report_figures(summary)
    cal = summary.get("calibration", {})
    pooled = cal.get("pooled") or {}
    rows = "".join(f"<tr><td>{d['feature']}</td><td>{d['psi']:.3f}</td><td>{d['status']}</td></tr>"
                   for d in summary.get("input_drift", []))
    imgs = "".join(f'<h2>{k.title()}</h2><img alt="{k}" src="data:image/png;base64,{v}">' for k, v in figs.items())
    alerts = "".join(f"<li>{a}</li>" for a in summary.get("alerts", [])) or "<li>none</li>"
    ratio = f"{pooled['ratio']:.3f}" if pooled.get("ratio") else "n/a"
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Monitoring report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1f1f1e;background:#fcfcfb}}
table{{border-collapse:collapse}}td,th{{padding:4px 10px;border-bottom:1px solid #e6e6e1;text-align:left}}img{{max-width:100%}}</style>
</head><body><h1>Monitoring report: {summary.get('model_version', '')}</h1>
<p>Predictions logged: {summary.get('n_predictions', 0):,}. Status: <b>{summary.get('status')}</b>.</p>
<h2>Alerts</h2><ul>{alerts}</ul>
<h2>Calibration (pooled)</h2><p>Loans with 12-month outcomes: {cal.get('n_loans', 0):,};
predicted / observed = {ratio} (alert outside {CAL_LOW}-{CAL_HIGH}).</p>
<h2>Input drift (PSI)</h2><table><tr><th>feature</th><th>PSI</th><th>status</th></tr>{rows}</table>
{imgs}</body></html>"""
    (out_dir / "report.html").write_text(html)
    return out_dir / "report.html"


def main() -> None:
    import argparse
    from .bundle import load_bundle
    from .db import init_db, make_engine

    parser = argparse.ArgumentParser()
    parser.add_argument("--month-from"); parser.add_argument("--month-to")
    args = parser.parse_args()
    Session = init_db(make_engine())
    summary = monitoring_summary(Session, load_bundle(), args.month_from, args.month_to)
    path = write_report(summary)
    print(json.dumps({k: summary.get(k) for k in ("n_predictions", "status", "alerts")}, indent=2))
    print(f"wrote {path} and {path.parent / 'summary.json'}")


if __name__ == "__main__":
    main()
