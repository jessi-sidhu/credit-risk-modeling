"""Group-conditional metrics for disparate-impact / fairness audits.

Conventions (credit-domain framing):
- The model predicts P(default). The *favorable* outcome for the
  applicant is therefore Ŷ=0 ("approved").
- `approval_rate` = P(Ŷ=0 | A=a). Demographic parity == approval-rate
  parity across groups.
- `fnr` (missed defaults) is the lender's costly error.
- `fpr` (good loans denied) is the borrower's costly error — the one
  fairness audits typically focus on alongside approval-rate parity.

`group_metrics(...)` returns a DataFrame indexed by group with rates
and raw confusion counts; `parity_ratios(...)` reduces it to ratios
versus a reference group; `four_fifths_rule(...)` is the standard
US-EEOC disparate-impact heuristic, also widely cited in lending.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def group_metrics(
    y_true,
    y_prob,
    group,
    threshold: float = 0.5,
) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    group = np.asarray(group)

    rows = []
    for g in sorted(np.unique(group)):
        mask = group == g
        yt, yp = y_true[mask], y_pred[mask]
        if len(yt) == 0:
            continue
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        rows.append({
            "group": g,
            "n": int(len(yt)),
            "base_rate": float(yt.mean()),
            "approval_rate": float(1 - yp.mean()),
            "prediction_rate": float(yp.mean()),
            "tpr": float(tp / max(tp + fn, 1)),
            "fpr": float(fp / max(fp + tn, 1)),
            "fnr": float(fn / max(tp + fn, 1)),
            "ppv": float(tp / max(tp + fp, 1)),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        })
    return pd.DataFrame(rows).set_index("group")


def parity_ratios(metrics: pd.DataFrame, reference: Optional[str] = None) -> pd.DataFrame:
    """Per-metric ratio against a reference group. Default reference is
    the highest-approval-rate group ("most favorably treated")."""
    cols = ("approval_rate", "tpr", "fpr", "fnr", "ppv")
    if reference is None:
        reference = str(metrics["approval_rate"].idxmax())
    ref = metrics.loc[reference]
    out = pd.DataFrame(index=metrics.index)
    for c in cols:
        denom = ref[c]
        out[f"{c}_ratio"] = metrics[c] / denom if denom > 0 else np.nan
    out["reference_group"] = reference
    return out


def four_fifths_rule(metrics: pd.DataFrame) -> dict:
    """The lowest approval rate must be at least 80% of the highest."""
    ar = metrics["approval_rate"]
    if ar.max() <= 0:
        return {"passes": True, "ratio": np.nan, "min_group": None, "max_group": None}
    ratio = float(ar.min() / ar.max())
    return {
        "passes": ratio >= 0.8,
        "ratio": ratio,
        "min_group": str(ar.idxmin()),
        "max_group": str(ar.idxmax()),
    }
