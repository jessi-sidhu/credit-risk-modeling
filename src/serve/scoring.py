"""Score loan applications with the Phase 2 hazard model bundle.

request -> features -> per-quarter hazards -> default curve -> expected loss

1. `LoanApplication` validates the application-time inputs against
   models/pd_survival_v1/feature_schema.json (types, ranges, categories).
2. The Phase 1 feature engineering adds the four ratio features.
3. The hazard model predicts, for each quarter of the term, the chance of
   default and of prepayment given the loan is still active.
4. `calibrated=True` (the default) shifts the hazard levels by the offsets
   fit on the 2014 validation cohort. Phase 2 showed the raw model predicts
   only ~71-80% of observed defaults on 2015-2018 loans; the offsets close
   about 30% of that gap, so calibrated is the less wrong default.
5. Hazards become the default curve (CIF), and with EAD from the
   amortization schedule, LGD by grade and discounting at the loan's rate,
   12-month and lifetime expected loss (same code as src/ecl.py).
6. Reason codes: the 4 features pushing first-quarter default risk up most,
   from SHAP on the default-class output.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..ecl import discount_factors, ead_schedule, expected_loss
from ..features import engineer_features
from ..survival import (
    MAX_PERIODS,
    PERIOD_MONTHS,
    apply_level_offsets,
    cif_at,
    cumulative_incidence,
)
from .bundle import DEFAULT_BUNDLE, ModelBundle, load_bundle

N_REASONS = 4
_SCHEMA = json.loads((DEFAULT_BUNDLE / "feature_schema.json").read_text())
_FIELDS = _SCHEMA["fields"]
INPUT_FIELDS = list(_FIELDS)


def _range_check(name: str, value):
    if value is None:
        return value
    spec = _FIELDS[name]
    if "valid" in spec:
        lo, hi = spec["valid"]
        if not lo <= value <= hi:
            raise ValueError(f"{name} must be between {lo:g} and {hi:g}")
    if "allowed" in spec and value not in spec["allowed"]:
        raise ValueError(f"{name} must be one of {spec['allowed']}")
    return value


class LoanApplication(BaseModel):
    """Application-time inputs only. Unknown fields are rejected."""
    model_config = ConfigDict(extra="forbid")

    loan_amount: float = Field(description="requested amount, USD")
    term: Literal[36, 60]
    int_rate: float = Field(description="interest rate, percent")
    grade: str = Field(description="LC grade A-G")
    emp_length: int = Field(description="years employed, 0-10 (10 = 10+)")
    home_ownership: str
    annual_income: float = Field(description="stated annual income, USD")
    purpose: str
    dti: float = Field(description="debt-to-income ratio, percent")
    fico: float = Field(description="FICO score at application")
    open_acc: Optional[float] = Field(default=None, description="open credit lines")
    revol_util: Optional[float] = Field(default=None, description="revolving utilization, percent")
    loan_id: Optional[str] = Field(default=None, max_length=64,
                                   description="client id, used to match outcomes later")
    issue_d: Optional[str] = Field(default=None, description="issue month YYYY-MM, for monitoring only")

    @field_validator(*[f for f in INPUT_FIELDS if f != "term"])
    @classmethod
    def _check(cls, v, info):
        return _range_check(info.field_name, v)

    @field_validator("issue_d")
    @classmethod
    def _check_month(cls, v):
        if v is not None and not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", v):
            raise ValueError("issue_d must be YYYY-MM")
        return v


# --- reason codes --------------------------------------------------------------

@lru_cache(maxsize=4)
def _explainer(bundle_path: str):
    import shap
    return shap.TreeExplainer(load_bundle(bundle_path).model.clf)


# (above training median, below training median) wording per numeric feature
PHRASES = {
    "loan_amount": ("Large loan amount", "Small loan amount"),
    "int_rate": ("High interest rate", "Low interest rate"),
    "emp_length": ("Long employment history", "Short employment history"),
    "annual_income": ("High annual income", "Low annual income"),
    "dti": ("High debt-to-income ratio", "Low debt-to-income ratio"),
    "fico": ("High FICO score", "Low FICO score"),
    "open_acc": ("Many open credit lines", "Few open credit lines"),
    "revol_util": ("High revolving utilization", "Low revolving utilization"),
    "loan_to_income": ("Loan is large relative to income", "Loan is small relative to income"),
    "installment_estimate": ("High monthly installment", "Low monthly installment"),
    "installment_to_income": ("Installment is high relative to income", "Installment is low relative to income"),
    "emp_x_homeowner": ("Long-tenured homeowner", "Not a homeowner, or short employment history"),
}


def _feature_label(name: str, value, schema: dict) -> str:
    """Plain-English label for a design-matrix column with a positive
    contribution to default risk."""
    fields, eng = schema["fields"], schema["engineered"]
    for cat in ("grade", "home_ownership", "purpose"):
        if name.startswith(cat + "_"):
            level = name[len(cat) + 1:].replace("_", " ")
            return f"{fields[cat]['label']}: {level}"
    if name == "term":
        return f"{int(value)}-month term"
    spec = fields.get(name) or eng.get(name)
    if spec is None:
        return name
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return f"{spec['label']} not provided"
    med = spec.get("training_median")
    high, low = PHRASES.get(name, (f"High {spec['label']}", f"Low {spec['label']}"))
    return high if med is None or value > med else low


def _reason_codes(bundle: ModelBundle, X: pd.DataFrame) -> list[list[dict]]:
    model = bundle.model
    Z = np.hstack([model.pre.transform(model._clip(X)).astype(np.float32),
                   model._period_block(np.ones(len(X), int), X["term"].to_numpy())])
    sv = _explainer(str(bundle.path)).shap_values(Z)
    sv = sv[1] if isinstance(sv, list) else sv[..., 1]          # default class
    names = list(model.pre.get_feature_names_out()) + ["period"]
    out = []
    for i in range(len(X)):
        row = []
        for j in np.argsort(-sv[i]):
            if sv[i, j] <= 0 or len(row) == N_REASONS:
                break
            name = names[j]
            if name == "period":
                continue
            if any(name.startswith(c + "_") for c in ("grade", "home_ownership", "purpose")) and Z[i, j] == 0:
                continue      # an absent category "raising" risk isn't a useful reason
            value = X[name].iloc[i] if name in X.columns else None
            row.append({"feature": name, "label": _feature_label(name, value, bundle.schema),
                        "shap_log_odds": float(sv[i, j])})
        out.append(row)
    return out


# --- scoring ---------------------------------------------------------------------

def _frame(apps: list[LoanApplication]) -> pd.DataFrame:
    rows = [{f: getattr(a, f) for f in INPUT_FIELDS} for a in apps]
    df = pd.DataFrame(rows).astype({"open_acc": float, "revol_util": float})
    # issue_d is not a model input (the bundled model has no time features);
    # HazardModel reads the column, so fill it when absent.
    df["issue_d"] = pd.to_datetime([a.issue_d or "2015-01" for a in apps], format="%Y-%m")
    return engineer_features(df)


def _warnings(X: pd.DataFrame, schema: dict) -> list[list[str]]:
    caps = {**{k: v for k, v in schema["fields"].items() if "training_cap" in v},
            **schema["engineered"]}
    out = []
    for i in range(len(X)):
        w = []
        for name, spec in caps.items():
            v = X[name].iloc[i]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            lo, hi = spec["training_cap"]
            if np.isinf(v) or v > hi:
                w.append(f"{name}={v:g} is above the training range; scored as {hi:g}")
            elif v < lo:
                w.append(f"{name}={v:g} is below the training range; scored as {lo:g}")
        out.append(w)
    return out


def score_many(apps: list[LoanApplication], calibrated: bool = True,
               bundle: ModelBundle | None = None, reasons: bool = True) -> list[dict]:
    bundle = bundle or get_bundle()
    X = _frame(apps)
    h = bundle.model.hazards(X)
    if calibrated:
        off = bundle.level_offsets
        h = apply_level_offsets(h, [off["default"], off["prepay"]])
    curves = cumulative_incidence(h)
    cif, cif_p, mpd = curves["cif_default"], curves["cif_prepay"], curves["marginal_pd_default"]

    lgd_table = bundle.lgd_by_grade["lgd"]
    lgd = X["grade"].map(lgd_table).fillna(float(np.mean(list(lgd_table.values())))).to_numpy()
    el = expected_loss(mpd, ead_schedule(X["loan_amount"], X["int_rate"], X["term"], MAX_PERIODS),
                       lgd, discount_factors(X["int_rate"], MAX_PERIODS))
    codes = _reason_codes(bundle, X) if reasons else [[] for _ in apps]
    warns = _warnings(X, bundle.schema)

    out = []
    for i, app in enumerate(apps):
        nq = app.term // PERIOD_MONTHS
        out.append({
            "loan_id": app.loan_id,
            "model_version": bundle.version,
            "calibrated": calibrated,
            "pd_12m": float(cif_at(cif[i:i + 1], 12)[0]),
            "pd_lifetime": float(cif[i, nq - 1]),
            "prepay_lifetime": float(cif_p[i, nq - 1]),
            "cif_default": [float(x) for x in cif[i, :nq]],
            "cif_default_months": [PERIOD_MONTHS * (q + 1) for q in range(nq)],
            "ecl_12m": float(el["ecl_12m"][i]),
            "ecl_lifetime": float(el["ecl_lifetime"][i]),
            "lgd": float(lgd[i]),
            "reason_codes": codes[i],
            "warnings": warns[i],
        })
    return out


def score(app: LoanApplication, calibrated: bool = True, bundle: ModelBundle | None = None) -> dict:
    return score_many([app], calibrated=calibrated, bundle=bundle)[0]


@lru_cache(maxsize=1)
def get_bundle() -> ModelBundle:
    return load_bundle()
