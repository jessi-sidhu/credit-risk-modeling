"""Discrete-time competing-risks hazard model (Phase 2).

Each loan is expanded into one row per quarter it was at risk
(`expand_person_period`). A classifier then predicts, for that quarter,
{0: no event, 1: default, 2: prepay} given the application features and
the quarter number. From the per-quarter hazards:

    S(t)            = prod_{k<=t} (1 - h_default(k) - h_prepay(k))
    CIF_default(t)  = sum_{k<=t} S(k-1) * h_default(k)

Reference: Botha & Verster (2025), arXiv 2507.15441.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.linear_model import LogisticRegression

from . import RANDOM_STATE
from .features import feature_lists, make_preprocessor
from .survival_data import EVENT_CENSORED

logger = logging.getLogger(__name__)

PERIOD_MONTHS = 3
TERMS = (36, 60)
N_PERIODS = {t: t // PERIOD_MONTHS for t in TERMS}           # 12 and 20 quarters
MAX_PERIODS = max(N_PERIODS.values())

# Numeric features are capped at these training-set quantiles before the
# Phase 1 preprocessor. Recent LC loans include stated incomes near zero
# (and one of exactly 0, which makes loan_to_income infinite), DTI up to
# 999 and incomes up to $110M. Logistic regression extrapolates linearly
# on such values and its hazards saturate at 0 or 1; trees are unaffected.
CLIP_QUANTILES = (0.001, 0.999)

# Cap on the number of LOANS (not rows) used for fitting, so the expanded
# table fits in memory. Loans are sampled, never individual rows, so every
# sampled loan keeps its full history.
MAX_TRAIN_LOANS = 300_000


# --- Person-period expansion ---------------------------------------------------

def periods_at_risk(outcomes: pd.DataFrame, period_months: int = PERIOD_MONTHS) -> np.ndarray:
    """Number of quarters each loan contributes.

    A loan with an event contributes through the quarter the event falls in
    (ceil(duration / 3)). A censored loan contributes only the quarters it
    was observed for in full (floor(duration / 3)): counting a partly
    observed quarter as survived would bias the hazard down.
    """
    d = outcomes["duration"].to_numpy()
    has_event = outcomes["event"].to_numpy() != EVENT_CENSORED
    return np.where(has_event, -(-d // period_months), d // period_months).astype(np.int64)


def expand_person_period(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    period_months: int = PERIOD_MONTHS,
) -> pd.DataFrame:
    """One row per loan per quarter at risk.

    Columns: every column of `features` (numerics as float32, strings as
    categoricals), `period` (1, 2, ...), `loan_id`, and the target `y`:
    0 if no event that quarter, else the event code (1 default, 2 prepay)
    in the quarter the event happened. Censored loans only have y = 0 rows.
    """
    if not features.index.equals(outcomes.index):
        raise ValueError("features and outcomes must share the same index")
    k = periods_at_risk(outcomes, period_months)
    n_rows = int(k.sum())
    loan_pos = np.repeat(np.arange(len(k)), k)
    # period number within each loan: 1..k_i
    starts = np.repeat(np.cumsum(k) - k, k)
    period = (np.arange(n_rows) - starts + 1).astype(np.int16)
    last = period == k[loan_pos]
    event = outcomes["event"].to_numpy()[loan_pos]
    y = np.where(last, event, 0).astype(np.int8)

    out = {}
    for col in features.columns:
        s = features[col]
        if s.dtype.kind == "f" or col in ("emp_length", "open_acc"):
            out[col] = s.to_numpy(dtype=np.float32)[loan_pos]
        elif s.dtype.kind in "iu":
            out[col] = s.to_numpy()[loan_pos].astype(np.int16 if col == "term" else s.dtype)
        elif s.dtype.kind == "M":
            out[col] = s.to_numpy()[loan_pos]
        else:
            cat = s.astype("category")
            out[col] = pd.Categorical.from_codes(cat.cat.codes.to_numpy()[loan_pos], cat.cat.categories)
    pp = pd.DataFrame(out)
    pp["period"] = period
    pp["loan_id"] = pd.Categorical.from_codes(loan_pos, features.index.astype(str)) \
        if len(features) else pd.Categorical([])
    pp["y"] = y
    logger.info("person-period: %d loans -> %d rows (%.1f MB)",
                len(features), n_rows, pp.memory_usage(deep=True).sum() / 1e6)
    return pp


def sample_loans(features: pd.DataFrame, outcomes: pd.DataFrame, max_loans: int,
                 seed: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(features) <= max_loans:
        return features, outcomes
    idx = features.sample(n=max_loans, random_state=seed).index
    logger.info("sampled %d of %d loans", max_loans, len(features))
    return features.loc[idx], outcomes.loc[idx]


# --- Time-varying features (level experiments) --------------------------------------

UNRATE_PATH = Path(__file__).resolve().parent.parent / "data" / "external" / "UNRATE.csv"
VINTAGE_ORIGIN = 2007.0
TIME_FEATURES = ("vintage", "unrate")


@lru_cache(maxsize=1)
def _unrate_by_month() -> tuple[int, np.ndarray]:
    """(first month index, monthly US unemployment rate) from FRED UNRATE,
    month index = year * 12 + month - 1."""
    u = pd.read_csv(UNRATE_PATH, parse_dates=["observation_date"])
    first = int(u["observation_date"].dt.year.iloc[0] * 12 + u["observation_date"].dt.month.iloc[0] - 1)
    return first, u["UNRATE"].to_numpy(dtype=float)


def time_feature_block(issue_d, period, names: Sequence[str]) -> np.ndarray:
    """(n, len(names)) values of time features for loan-quarters.

    vintage  years from 2007-01 to the issue month (constant per loan);
             a linear model can extrapolate it, trees cannot.
    unrate   US unemployment rate in the first calendar month of the
             loan-quarter (issue month + 3 * (period - 1)). For test loans
             this uses the REALIZED path, i.e. perfect foresight: an upper
             bound on what a macro scenario could add.
    """
    issue = pd.DatetimeIndex(issue_d)
    month = issue.year.to_numpy() * 12 + issue.month.to_numpy() - 1
    cols = []
    for name in names:
        if name == "vintage":
            cols.append(month / 12.0 - VINTAGE_ORIGIN)
        elif name == "unrate":
            first, series = _unrate_by_month()
            idx = month + PERIOD_MONTHS * (np.asarray(period) - 1) - first
            if (idx < 0).any() or (idx >= len(series)).any():
                raise ValueError("unemployment series does not cover all loan-quarters")
            cols.append(series[idx])
        else:
            raise ValueError(f"unknown time feature {name!r}")
    return np.column_stack(cols).astype(np.float32) if cols else np.zeros((len(month), 0), np.float32)


# --- Hazard models ---------------------------------------------------------------

_PERIOD_CATS = [(t, p) for t in TERMS for p in range(1, N_PERIODS[t] + 1)]
_PERIOD_INDEX = {tp: i for i, tp in enumerate(_PERIOD_CATS)}


class HazardModel:
    """Multiclass per-quarter hazard model.

    kind='lr'   multinomial logistic regression; baseline hazard as one-hot
                (term, quarter) dummies, i.e. a separate baseline per term.
    kind='lgbm' LightGBM multiclass; `period` and `term` enter as numeric
                features so trees can learn their interactions.

    Numeric features are capped at training quantiles (CLIP_QUANTILES,
    infinities become missing), then the Phase 1 preprocessor (impute,
    scale for LR, one-hot) is fit on the training rows; the period block
    is appended after it, then any `time_features` (standardized on the
    training rows; see time_feature_block).
    """

    def __init__(self, kind: str, time_features: Sequence[str] = (), **params: Any):
        if kind not in ("lr", "lgbm"):
            raise ValueError(kind)
        self.kind, self.params = kind, params
        self.time_features = tuple(time_features)
        self.pre = make_preprocessor(scale_numeric=(kind == "lr"))
        self.best_iteration_: int | None = None

    def _period_block(self, period: np.ndarray, term: np.ndarray) -> np.ndarray:
        if self.kind == "lgbm":
            return np.asarray(period, dtype=np.float32).reshape(-1, 1)
        block = np.zeros((len(period), len(_PERIOD_CATS)), dtype=np.float32)
        cols = np.array([_PERIOD_INDEX[(int(t), int(p))] for t, p in zip(term, period)]) \
            if len(period) else np.zeros(0, int)
        block[np.arange(len(period)), cols] = 1.0
        return block

    def _fit_clip(self, pp: pd.DataFrame) -> None:
        numeric, _ = feature_lists()
        vals = pp[numeric].replace([np.inf, -np.inf], np.nan)
        lo, hi = CLIP_QUANTILES
        self.clip_ = {c: (float(vals[c].quantile(lo)), float(vals[c].quantile(hi))) for c in numeric}

    def _clip(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c, (lo, hi) in self.clip_.items():
            X[c] = X[c].replace([np.inf, -np.inf], np.nan).clip(lo, hi)
        return X

    def _time_block(self, issue_d, period) -> np.ndarray:
        if not self.time_features:
            return np.zeros((len(period), 0), np.float32)
        T = time_feature_block(issue_d, period, self.time_features)
        return ((T - self.time_mean_) / self.time_std_).astype(np.float32)

    def _design(self, pp: pd.DataFrame) -> np.ndarray:
        base = self.pre.transform(self._clip(pp)).astype(np.float32)
        period = pp["period"].to_numpy()
        return np.hstack([base, self._period_block(period, pp["term"].to_numpy()),
                          self._time_block(pp["issue_d"].to_numpy(), period)])

    def fit(self, pp: pd.DataFrame, eval_pp: pd.DataFrame | None = None) -> "HazardModel":
        self._fit_clip(pp)
        self.pre.fit(self._clip(pp))
        if self.time_features:
            T = time_feature_block(pp["issue_d"].to_numpy(), pp["period"].to_numpy(), self.time_features)
            self.time_mean_, self.time_std_ = T.mean(axis=0), T.std(axis=0) + 1e-9
        Z, y = self._design(pp), pp["y"].to_numpy()
        if self.kind == "lr":
            self.clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, **self.params)
            self.clf.fit(Z, y)
        else:
            params = dict(objective="multiclass", n_estimators=2000, learning_rate=0.05,
                          num_leaves=31, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
            params.update(self.params)
            self.clf = LGBMClassifier(**params)
            fit_kw = {}
            if eval_pp is not None:
                fit_kw = dict(eval_set=[(self._design(eval_pp), eval_pp["y"].to_numpy())],
                              eval_metric="multi_logloss",
                              callbacks=[early_stopping(50, verbose=False)])
            self.clf.fit(Z, y, **fit_kw)
            self.best_iteration_ = getattr(self.clf, "best_iteration_", None)
        return self

    def predict_rows(self, pp: pd.DataFrame) -> np.ndarray:
        """Per-row class probabilities [no event, default, prepay]."""
        return self.clf.predict_proba(self._design(pp))

    def hazards(self, loans: pd.DataFrame, chunk: int = 200_000) -> np.ndarray:
        """(n_loans, MAX_PERIODS, 3) per-quarter hazards for every quarter of
        each loan's term; NaN after the term. The base features are
        transformed once per chunk and reused for every quarter."""
        n = len(loans)
        out = np.full((n, MAX_PERIODS, 3), np.nan, dtype=np.float32)
        terms = loans["term"].to_numpy()
        issue = loans["issue_d"].to_numpy()
        for lo in range(0, n, chunk):
            sl = slice(lo, min(lo + chunk, n))
            base = self.pre.transform(self._clip(loans.iloc[sl])).astype(np.float32)
            t_chunk = terms[sl]
            for p in range(1, MAX_PERIODS + 1):
                m = t_chunk // PERIOD_MONTHS >= p
                if not m.any():
                    continue
                per = np.full(m.sum(), p)
                Z = np.hstack([base[m], self._period_block(per, t_chunk[m]),
                               self._time_block(issue[sl][m], per)])
                rows = np.flatnonzero(m) + lo
                out[rows, p - 1] = self.clf.predict_proba(Z)
        return out


# --- Level recalibration ---------------------------------------------------------

def fit_level_offsets(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Log-odds shifts (c_default, c_prepay) that maximize the likelihood of
    `y` (0/1/2) when added to the logits of `probs` (rows of
    [no event, default, prepay]). Fit on validation rows: it moves the level
    of each hazard without changing which loans rank riskier."""
    from scipy.optimize import minimize

    logp = np.log(np.clip(probs, 1e-12, 1.0)).astype(np.float64)
    y = np.asarray(y, dtype=int)

    def nll(c):
        z = logp + np.r_[0.0, c]
        z -= z.max(axis=1, keepdims=True)
        lse = np.log(np.exp(z).sum(axis=1))
        return -(z[np.arange(len(y)), y] - lse).mean()

    return minimize(nll, x0=np.zeros(2), method="Nelder-Mead",
                    options={"xatol": 1e-5, "fatol": 1e-9}).x


def apply_level_offsets(h: np.ndarray, offsets) -> np.ndarray:
    """Apply `fit_level_offsets` shifts to (..., 3) hazards; NaN stays NaN."""
    import warnings

    shift = np.r_[0.0, np.asarray(offsets, float)]
    z = np.log(np.clip(h, 1e-12, 1.0)) + shift
    with warnings.catch_warnings():                 # quarters after the term are all-NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        z = np.exp(z - np.nanmax(z, axis=-1, keepdims=True))
    return (z / z.sum(axis=-1, keepdims=True)).astype(h.dtype)


# --- Hazards -> curves -------------------------------------------------------------

def cumulative_incidence(h: np.ndarray) -> dict[str, np.ndarray]:
    """From (n, T, 3) hazards to survival and cumulative incidence curves,
    each (n, T) and indexed by quarter (column t = end of quarter t+1).
    After a loan's term (NaN hazards) the curves stay flat."""
    hd = np.nan_to_num(h[..., 1])
    hp = np.nan_to_num(h[..., 2])
    S = np.cumprod(1.0 - hd - hp, axis=1)
    S_prev = np.hstack([np.ones((len(h), 1)), S[:, :-1]])
    return {
        "survival": S,
        "cif_default": np.cumsum(S_prev * hd, axis=1),
        "cif_prepay": np.cumsum(S_prev * hp, axis=1),
        "marginal_pd_default": S_prev * hd,
    }


def cif_at(cif: np.ndarray, months: int) -> np.ndarray:
    """CIF at a horizon in months (quarter boundary)."""
    q = months // PERIOD_MONTHS
    return cif[:, min(q, cif.shape[1]) - 1]
