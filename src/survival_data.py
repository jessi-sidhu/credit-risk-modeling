"""Outcome table for the time-to-default (survival) model.

Phase 1 kept only finished, matured, 36-month loans. Here every accepted
loan stays in, both terms, finished or not. Each loan gets:

    duration  months from issue_d to its event or censoring time, in [1, term]
    event     0 = censored, 1 = default, 2 = prepay

Features and outcomes are returned as two separate frames sharing the LC
loan `id` index. Outcome columns (`loan_status`, `last_pymnt_d`) are read
only here and never enter the feature frame; tests/test_features.py checks
this.

Event rules (month counts are calendar-month differences, since LC dates
are month-level):

    Charged Off / Default (+ credit-policy variant)   event 1, duration = months to last_pymnt_d + 1
                                                       (the first missed payment); 1 if last_pymnt_d missing
    Fully Paid (+ credit-policy variant), early        event 2, duration = months to last_pymnt_d
    Fully Paid at or near maturity                     event 0, duration = term
    Current / Late / In Grace Period                   event 0, duration = months to last_pymnt_d
                                                       (SNAPSHOT_DATE if missing)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    KEY_FIELDS,
    _is_matured,
    LC_USECOLS,
    RAW_DIR,
    SNAPSHOT_DATE,
    find_accepted_file,
    parse_lc_features,
)

logger = logging.getLogger(__name__)

EVENT_CENSORED, EVENT_DEFAULT, EVENT_PREPAY = 0, 1, 2
EVENT_NAMES = {EVENT_CENSORED: "censored", EVENT_DEFAULT: "default", EVENT_PREPAY: "prepay"}

# Outcome-only columns. Read here for the outcome frame, never as features.
OUTCOME_COLS = ["loan_status", "last_pymnt_d"]

DEFAULT_STATUSES = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
}
PAID_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}
OPEN_STATUSES = {
    "Current",
    "Late (16-30 days)",
    "Late (31-120 days)",
    "In Grace Period",
}

# Open loans that have missed payments at the snapshot.
ARREARS_STATUSES = {"Late (16-30 days)", "Late (31-120 days)", "In Grace Period"}

# A Fully Paid loan whose last payment falls within this many months of the
# scheduled term counts as paid at maturity (censored at `term`), not as a
# prepayment. Payment dates jitter by a month or two around the schedule
# (a final payment a few days late lands in the next calendar month), and
# paying off the last one or two instalments early is not the behaviour a
# prepayment model is meant to capture.
NEAR_MATURITY_MONTHS = 2

# Durations outside [1, term] are clipped and kept (and counted in the log):
# payoffs in the issue month give 0, and loans on hardship plans or
# modifications can have their last payment after the scheduled term.
# Set to True to drop them instead.
DROP_CLIPPED = False


def _months_between(start: pd.Series, end: pd.Series) -> pd.Series:
    return (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)


def build_outcomes(raw: pd.DataFrame, snapshot_date: str = SNAPSHOT_DATE) -> pd.DataFrame:
    """Compute duration and event from raw LC columns.

    `raw` needs `term` (int), `issue_d` (datetime), `loan_status` and
    `last_pymnt_d` (datetime, may be NaT). Returns a frame with
    `duration`, `event`, `loan_status`, `finished`, `clipped` and `impossible`.
    """
    term = raw["term"].astype(int)
    status = raw["loan_status"]
    last_pay = raw["last_pymnt_d"]
    m_last = _months_between(raw["issue_d"], last_pay)
    m_snap = _months_between(raw["issue_d"], pd.Series(pd.Timestamp(snapshot_date), index=raw.index))

    is_default = status.isin(DEFAULT_STATUSES)
    is_paid = status.isin(PAID_STATUSES)
    is_open = status.isin(OPEN_STATUSES)
    unknown = ~(is_default | is_paid | is_open)
    prepaid = is_paid & (m_last < term - NEAR_MATURITY_MONTHS)

    event = pd.Series(EVENT_CENSORED, index=raw.index, dtype="int8")
    event[is_default] = EVENT_DEFAULT
    event[prepaid] = EVENT_PREPAY

    duration = pd.Series(np.nan, index=raw.index)
    duration[is_default] = (m_last + 1).where(last_pay.notna(), 1)[is_default]
    duration[prepaid] = m_last[prepaid]
    duration[is_paid & ~prepaid] = term[is_paid & ~prepaid]
    duration[is_open] = m_last.where(last_pay.notna(), m_snap)[is_open]

    impossible = (
        unknown
        | (last_pay.notna() & (last_pay < raw["issue_d"]))
        | (is_paid & last_pay.isna())
        | duration.isna()
    )
    clipped = ~impossible & ((duration < 1) | (duration > term))
    duration = duration.clip(lower=1, upper=term)

    return pd.DataFrame({
        "duration": duration.astype("Int64"),
        "event": event,
        "loan_status": status,
        # reached a final status (default or paid off, early or at term);
        # lets downstream code avoid touching loan_status
        "finished": is_default | is_paid,
        "clipped": clipped,
        "impossible": impossible,
    })


def load_survival_data(
    raw_dir: Path = RAW_DIR,
    snapshot_date: str = SNAPSHOT_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(features, outcomes) for every accepted loan, both indexed by LC `id`.

    `features` has the Phase 1 feature schema (no `default` column);
    `outcomes` has `duration`, `event`, `loan_status`, `finished`, `clipped`.
    """
    path = find_accepted_file(raw_dir)
    usecols = ["id"] + LC_USECOLS + [c for c in OUTCOME_COLS if c not in LC_USECOLS]
    raw = pd.read_csv(path, usecols=usecols, low_memory=False)
    # summary/footer lines are empty in every column we read (see data.py)
    raw = raw.dropna(subset=["loan_amnt", "term"]).set_index("id")
    counts = {"rows_read": len(raw)}
    logger.info("LC rows read: %d", len(raw))

    features = parse_lc_features(raw.drop(columns=OUTCOME_COLS))
    keep = features[KEY_FIELDS].notna().all(axis=1)
    features, raw = features[keep], raw[keep]
    counts["key_fields_present"] = len(features)
    logger.info("after dropping rows missing key fields: %d", len(features))

    outcomes = build_outcomes(
        pd.DataFrame({
            "term": features["term"],
            "issue_d": features["issue_d"],
            "loan_status": raw["loan_status"],
            "last_pymnt_d": pd.to_datetime(raw["last_pymnt_d"], format="%b-%Y", errors="coerce"),
        }),
        snapshot_date=snapshot_date,
    )

    n_imp, n_clip = int(outcomes["impossible"].sum()), int(outcomes["clipped"].sum())
    logger.info("impossible dates or unknown status (dropped): %d", n_imp)
    if n_imp:
        logger.info("  by status: %s", outcomes.loc[outcomes["impossible"], "loan_status"].value_counts().to_dict())
    logger.info("duration clipped to [1, term] (%s): %d", "dropped" if DROP_CLIPPED else "kept", n_clip)
    if n_clip:
        logger.info("  by status: %s", outcomes.loc[outcomes["clipped"], "loan_status"].value_counts().to_dict())
    drop = outcomes["impossible"] | (outcomes["clipped"] & DROP_CLIPPED)
    features, outcomes = features[~drop], outcomes[~drop]
    counts.update(impossible=n_imp, clipped=n_clip, final=len(features))

    table = pd.crosstab(features["term"], outcomes["event"].map(EVENT_NAMES), margins=True)
    logger.info("events by term:\n%s", table.to_string())

    outcomes = outcomes.drop(columns=["impossible"]).astype({"duration": int})
    attrs = {
        "row_counts": counts,
        "events_by_term": {str(k): {str(c): int(v) for c, v in row.items()} for k, row in table.iterrows()},
        "snapshot_date": snapshot_date,
        "near_maturity_months": NEAR_MATURITY_MONTHS,
        "drop_clipped": DROP_CLIPPED,
        "source_file": path.name,
    }
    features = features.copy(); outcomes = outcomes.copy()
    features.attrs, outcomes.attrs = dict(attrs), dict(attrs)
    return features, outcomes


# --- Kaplan-Meier "no default yet" by grade ------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
KM_HORIZONS = (12, 24, 36, 60)
# Grade is ordinal, so one hue light -> dark. A single ramp only has room
# for five clearly distinct steps, so the small E, F, G grades share a band.
GRADE_BANDS = {"A": "A", "B": "B", "C": "C", "D": "D", "E": "E-G", "F": "E-G", "G": "E-G"}
BAND_COLORS = {"A": "#86b6ef", "B": "#5598e7", "C": "#2a78d6", "D": "#1c5cab", "E-G": "#104281"}


def km_no_default(features: pd.DataFrame, outcomes: pd.DataFrame) -> dict:
    """Kaplan-Meier estimate of P(no default by t) per term and grade band.
    Prepayment and censoring are both treated as censored here (the
    competing-risks view comes with the hazard model)."""
    from lifelines import KaplanMeierFitter

    bands = features["grade"].map(GRADE_BANDS)
    out = {}
    for term in sorted(features["term"].unique()):
        for band in BAND_COLORS:
            m = (features["term"] == term) & (bands == band)
            if not m.any():
                continue
            kmf = KaplanMeierFitter().fit(
                outcomes.loc[m, "duration"], outcomes.loc[m, "event"] == EVENT_DEFAULT,
            )
            out[(int(term), band)] = (kmf, int(m.sum()))
    return out


def plot_km_by_grade(km: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    terms = sorted({t for t, _ in km})
    fig, axes = plt.subplots(1, len(terms), figsize=(6.5 * len(terms), 5.5), sharey=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, term in zip(axes, terms):
        ax.set_facecolor("#fcfcfb")
        for band, color in BAND_COLORS.items():
            if (term, band) not in km:
                continue
            kmf, n = km[(term, band)]
            sf = kmf.survival_function_.iloc[:, 0]
            ax.step(sf.index, sf.values, where="post", color=color, lw=2, label=f"{band} (n={n:,})")
            ax.text(sf.index[-1] + 0.8, sf.values[-1], band, color="#3d3d3a", va="center", fontsize=10)
        ax.set_title(f"{term}-month loans", loc="left")
        ax.set_xlabel("Months since issue")
        ax.set_xlim(0, term + 5)
        ax.grid(color="#e6e6e1", lw=0.8); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, loc="lower left", fontsize=9, title="grade")
    axes[0].set_ylabel("Share not yet defaulted (Kaplan-Meier)")
    fig.suptitle("Time to default by LC grade, all accepted loans 2007-2018", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    import json

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    features, outcomes = load_survival_data()

    finished = outcomes["event"] != EVENT_CENSORED
    paid_at_maturity = (outcomes["event"] == EVENT_CENSORED) & outcomes["loan_status"].isin(PAID_STATUSES)
    km = km_no_default(features, outcomes)
    summary = {
        **features.attrs,
        "events_by_status": pd.crosstab(outcomes["loan_status"], outcomes["event"].map(EVENT_NAMES))
                              .to_dict(orient="index"),
        "default_share_of_finished": float((outcomes["event"] == EVENT_DEFAULT).sum()
                                           / (finished | paid_at_maturity).sum()),
        # Same population as Phase 1 (finished loans from cohorts matured by the
        # snapshot), by term. For 36-month loans this should match Phase 1.
        "default_share_of_finished_matured_by_term": {
            str(t): float((outcomes.loc[m, "event"] == EVENT_DEFAULT).mean())
            for t in sorted(features["term"].unique())
            for m in [(features["term"] == t)
                      & _is_matured(features["issue_d"], features["term"], features.attrs["snapshot_date"])
                      & outcomes["loan_status"].isin(DEFAULT_STATUSES | PAID_STATUSES)]
        },
        # Loans in arrears at the snapshot are censored, but many will end in
        # default. A high share for recent cohorts means their observed
        # default rates are understated (charge-off lags the missed payment).
        "in_arrears_at_snapshot_by_issue_year": {
            str(y): {"loans": int(len(g)),
                     "in_arrears": int(g.isin(ARREARS_STATUSES).sum()),
                     "share": float(g.isin(ARREARS_STATUSES).mean()),
                     "defaulted": int(g.isin(DEFAULT_STATUSES).sum())}
            for y, g in outcomes["loan_status"].groupby(features["issue_d"].dt.year)
        },
        "duration_median_by_event": outcomes.groupby(outcomes["event"].map(EVENT_NAMES))["duration"]
                                            .median().to_dict(),
        "km_no_default": {
            f"{term}m_{band}": {"n": n, **{f"S({h})": float(kmf.predict(h)) for h in KM_HORIZONS if h <= term}}
            for (term, band), (kmf, n) in km.items()
        },
    }
    out_dir = _PROJECT_ROOT / "reports" / "artifacts" / "survival"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out_dir / "outcomes_summary.json", "w"), indent=2, default=int)
    plot_km_by_grade(km, _PROJECT_ROOT / "reports" / "figures" / "km_by_grade.png")
    print(json.dumps({k: summary[k] for k in ("row_counts", "default_share_of_finished",
                                              "default_share_of_finished_matured_by_term",
                                              "duration_median_by_event")}, indent=2, default=int))
    print(f"saved {out_dir / 'outcomes_summary.json'} and reports/figures/km_by_grade.png")


if __name__ == "__main__":
    main()
