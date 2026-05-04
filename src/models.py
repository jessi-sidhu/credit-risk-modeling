"""Model factories.

Each returns a `Pipeline` of (preprocessor, classifier) so the
preprocessor is fit on the training fold only — protecting against
leakage end-to-end.

- Logistic regression baseline: scaled numerics, balanced class weight.
- Random forest: defaults, no scaling needed.
- LightGBM: defaults; the tuned version is built in `train.py`.
"""
from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import RANDOM_STATE
from .features import make_preprocessor


def logistic_regression() -> Pipeline:
    return Pipeline([
        ("pre", make_preprocessor(scale_numeric=True)),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])


def random_forest() -> Pipeline:
    return Pipeline([
        ("pre", make_preprocessor(scale_numeric=False)),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])


def lightgbm_classifier(**kwargs) -> Pipeline:
    params = dict(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    params.update(kwargs)
    return Pipeline([
        ("pre", make_preprocessor(scale_numeric=False)),
        ("clf", LGBMClassifier(**params)),
    ])
