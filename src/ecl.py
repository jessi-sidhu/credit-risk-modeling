"""Lifetime expected credit loss, IFRS 9-style ILLUSTRATION.

    python -m src.ecl          (after python -m src.train_survival)

Not a regulatory model: no macroeconomic scenarios, no stage allocation
(12-month vs lifetime by significant increase in credit risk), and LGD is a
simple average by grade.

Per loan and quarter t of its term:

    EAD(t)   balance outstanding at the start of quarter t, from the
             amortization schedule (features._monthly_installment)
    PD(t)    marginal default probability S(t-1) * h_default(t) from the
             Phase 2 hazard model
    LGD      by grade, from TRAINING-period charge-offs only:
             (loan_amnt - total_rec_prncp - recoveries) / (loan_amnt - total_rec_prncp)
    D(t)     discount factor at the loan's own interest rate

    ECL      = sum_t PD(t) * EAD(t) * LGD * D(t)
    12-month = first 4 quarters; lifetime = all quarters of the term

`total_rec_prncp` and `recoveries` are post-origination columns. They are
read only here, only for training-period charge-offs (LGD) and for the
realized-loss comparison on matured test loans, never as model features.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .data import RAW_DIR, find_accepted_file
from .features import _monthly_installment
from .survival import PERIOD_MONTHS, cumulative_incidence
from .survival_data import EVENT_DEFAULT, load_survival_data
from .train import TRAIN_UNTIL

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "reports" / "artifacts" / "survival"
FIG = ROOT / "reports" / "figures"
RECOVERY_COLS = ["total_rec_prncp", "recoveries"]
MODELS = ("lr", "lgbm")


def read_recovery_columns(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    path = find_accepted_file(raw_dir)
    raw = pd.read_csv(path, usecols=["id", "loan_amnt"] + RECOVERY_COLS, low_memory=False)
    raw = raw.dropna(subset=["loan_amnt"]).set_index("id")
    raw.index = raw.index.astype(str)
    return raw


def realized_loss(rec: pd.DataFrame) -> pd.Series:
    """Unrecovered principal of a charged-off loan (>= 0)."""
    return (rec["loan_amnt"] - rec["total_rec_prncp"] - rec["recoveries"]).clip(lower=0)


def lgd_by_grade(train_defaults: pd.DataFrame) -> pd.DataFrame:
    """`train_defaults`: loan_amnt, total_rec_prncp, recoveries, grade for
    training-period charge-offs. LGD per loan clipped to [0, 1]."""
    exposure = train_defaults["loan_amnt"] - train_defaults["total_rec_prncp"]
    ok = exposure > 0
    lgd = (realized_loss(train_defaults[ok]) / exposure[ok]).clip(0, 1)
    return (pd.DataFrame({"grade": train_defaults.loc[ok, "grade"], "lgd": lgd})
              .groupby("grade")["lgd"].agg(lgd="mean", n="size"))


def ead_schedule(loan_amount, int_rate, term, n_quarters: int) -> np.ndarray:
    """(n, n_quarters) balance at the start of each quarter; 0 after the term."""
    P = np.asarray(loan_amount, float)[:, None]
    r = (np.asarray(int_rate, float) / 1200.0)[:, None]
    n = np.asarray(term, float)[:, None]
    pmt = _monthly_installment(P[:, 0], np.asarray(int_rate, float), n[:, 0])[:, None]
    k = (np.arange(n_quarters) * PERIOD_MONTHS)[None, :].astype(float)      # payments made before quarter t
    growth = (1 + r) ** k
    bal = np.where(r > 0, P * growth - pmt * (growth - 1) / np.where(r > 0, r, 1), P - pmt * k)
    return np.where(k < n, np.clip(bal, 0, None), 0.0)


def discount_factors(int_rate, n_quarters: int) -> np.ndarray:
    """(n, n_quarters) discount at the loan's monthly rate to the end of quarter t."""
    r = (np.asarray(int_rate, float) / 1200.0)[:, None]
    months = ((np.arange(n_quarters) + 1) * PERIOD_MONTHS)[None, :]
    return (1 + r) ** (-months)


def expected_loss(marginal_pd, ead, lgd, disc) -> dict[str, np.ndarray]:
    loss = marginal_pd * ead * np.asarray(lgd, float)[:, None]
    return {
        "ecl_12m": (loss * disc)[:, :4].sum(axis=1),
        "ecl_lifetime": (loss * disc).sum(axis=1),
        "el_lifetime_undiscounted": loss.sum(axis=1),
    }


def plot_predicted_vs_realized(tbl: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    x = np.arange(len(tbl)); w = 0.26
    bars = [("realized_loss_rate", "realized", "#3d3d3a"),
            ("predicted_loss_rate_lr", "predicted, multinomial LR", "#2a78d6"),
            ("predicted_loss_rate_lgbm", "predicted, LightGBM", "#eb6834")]
    for i, (col, label, color) in enumerate(bars):
        ax.bar(x + (i - 1) * w, tbl[col] * 100, width=w - 0.02, color=color, label=label)
        for xi, v in zip(x, tbl[col] * 100):
            ax.text(xi + (i - 1) * w, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9, color="#3d3d3a")
    ax.set_xticks(x); ax.set_xticklabels([f"{g}\n({n:,} loans)" for g, n in zip(tbl.index, tbl["loans"])])
    ax.set_ylabel("lifetime loss, % of funded amount (undiscounted)")
    ax.set_title("Predicted vs realized loss, matured 36-month test loans", loc="left")
    ax.grid(axis="y", color="#e6e6e1", lw=0.8); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=120, facecolor=fig.get_facecolor()); plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t = pd.read_csv(ART / "test_loans.csv.gz", index_col="id")
    t.index = t.index.astype(str)
    rec = read_recovery_columns()

    # LGD on training-period charge-offs only
    features, outcomes = load_survival_data()
    features.index = features.index.astype(str); outcomes.index = outcomes.index.astype(str)
    end_train = pd.Period(TRAIN_UNTIL, freq="M").to_timestamp("M")
    train_def = (features["issue_d"] <= end_train) & (outcomes["event"] == EVENT_DEFAULT)
    td = rec.loc[train_def[train_def].index].assign(grade=features.loc[train_def, "grade"])
    lgd = lgd_by_grade(td)
    logger.info("LGD by grade (training charge-offs):\n%s", lgd.round(4).to_string())
    del features, outcomes

    n_q = 60 // PERIOD_MONTHS
    ead = ead_schedule(t["loan_amount"], t["int_rate"], t["term"], n_q)
    disc = discount_factors(t["int_rate"], n_q)
    lgd_loan = t["grade"].map(lgd["lgd"]).fillna(lgd["lgd"].mean()).to_numpy()
    for m in MODELS:
        h = np.load(ART / f"hazards_{m}.npy")
        full = np.concatenate([np.zeros(h.shape[:2] + (1,), np.float32), h], axis=2)
        mpd = cumulative_incidence(full)["marginal_pd_default"]
        for k, v in expected_loss(mpd, ead, lgd_loan, disc).items():
            t[f"{k}_{m}"] = v
        del h, full, mpd

    portfolio = {m: {"loans": int(len(t)), "funded": float(t["loan_amount"].sum()),
                     "ecl_12m": float(t[f"ecl_12m_{m}"].sum()),
                     "ecl_lifetime": float(t[f"ecl_lifetime_{m}"].sum())} for m in MODELS}
    for m in MODELS:
        p = portfolio[m]
        p["ecl_12m_pct_funded"] = p["ecl_12m"] / p["funded"]
        p["ecl_lifetime_pct_funded"] = p["ecl_lifetime"] / p["funded"]

    # Realized vs predicted on matured test loans: 36-month, issued 2015-01..2016-03
    issue = pd.to_datetime(t["issue_d"])
    mat = (t["term"] == 36) & (issue <= "2016-03-31")
    mt = t[mat].copy()
    mt["realized"] = np.where(mt["event"] == EVENT_DEFAULT, realized_loss(rec.loc[mt.index]), 0.0)
    grp = mt.groupby(pd.to_datetime(mt["issue_d"]).dt.year)
    cmp = pd.DataFrame({
        "loans": grp.size(),
        "funded": grp["loan_amount"].sum(),
        "realized_loss": grp["realized"].sum(),
        **{f"predicted_loss_{m}": grp[f"el_lifetime_undiscounted_{m}"].sum() for m in MODELS},
    })
    cmp.loc["all"] = cmp.sum()
    cmp["loans"] = cmp["loans"].astype(int)
    cmp["realized_loss_rate"] = cmp["realized_loss"] / cmp["funded"]
    for m in MODELS:
        cmp[f"predicted_loss_rate_{m}"] = cmp[f"predicted_loss_{m}"] / cmp["funded"]
        cmp[f"predicted_over_realized_{m}"] = cmp[f"predicted_loss_{m}"] / cmp["realized_loss"]
    cmp.index = cmp.index.astype(str)
    plot_predicted_vs_realized(cmp, FIG / "ecl_predicted_vs_realized.png")

    out = {
        "note": "Illustration only: no macro scenarios, no staging rules, LGD is a grade average.",
        "lgd_by_grade": {g: {"lgd": float(r["lgd"]), "n_train_chargeoffs": int(r["n"])} for g, r in lgd.iterrows()},
        "test_portfolio": portfolio,
        "matured_test_predicted_vs_realized": json.loads(cmp.to_json(orient="index")),
    }
    json.dump(out, open(ART / "ecl.json", "w"), indent=2)
    print(lgd.round(4).to_string())
    print(json.dumps(portfolio, indent=1))
    print(cmp.round(4).to_string())
    print(f"saved {ART / 'ecl.json'} and reports/figures/ecl_predicted_vs_realized.png")


if __name__ == "__main__":
    main()
