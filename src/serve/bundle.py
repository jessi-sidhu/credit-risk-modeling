"""Load the self-contained model bundle written by `src/export_model.py`.

The bundle holds everything scoring needs, so the API never touches the
Lending Club data or the Phase 2 artifacts:

    hazard_lgbm.joblib     fitted Phase 2 HazardModel (preprocessor + LightGBM)
    level_offsets.json     recalibration log-odds shifts (default, prepay)
    lgd_by_grade.json      LGD per grade from Phase 2's expected-loss step
    feature_schema.json    input fields, types, valid ranges, categories, training caps
    reference_stats.json   training-set binned distributions, for PSI
    model_card.json        version, training window, validation and test metrics
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BUNDLE = ROOT / "models" / "pd_survival_v1"
BUNDLE_FILES = ("hazard_lgbm.joblib", "level_offsets.json", "lgd_by_grade.json",
                "feature_schema.json", "reference_stats.json", "model_card.json")


@dataclass
class ModelBundle:
    path: Path
    model: Any                      # src.survival.HazardModel
    level_offsets: dict
    lgd_by_grade: dict
    schema: dict
    reference: dict
    card: dict

    @property
    def version(self) -> str:
        return self.card["model_version"]


def load_bundle(path: Path | str = DEFAULT_BUNDLE) -> ModelBundle:
    path = Path(path)
    missing = [f for f in BUNDLE_FILES if not (path / f).exists()]
    if missing:
        raise FileNotFoundError(f"model bundle {path} is missing {missing}; run `make export-model`")
    read = lambda name: json.loads((path / name).read_text())
    model = joblib.load(path / "hazard_lgbm.joblib")
    # The Phase 2 model was pickled before HazardModel gained the optional
    # `time_features` attribute (level experiments); it was trained without
    # any, which is what the default means.
    if not hasattr(model, "time_features"):
        model.time_features = ()
    return ModelBundle(
        path=path,
        model=model,
        level_offsets=read("level_offsets.json"),
        lgd_by_grade=read("lgd_by_grade.json"),
        schema=read("feature_schema.json"),
        reference=read("reference_stats.json"),
        card=read("model_card.json"),
    )
