"""SHAP interpretability for the tuned LightGBM pipeline.

Operates on a `Pipeline(pre=ColumnTransformer, clf=LGBMClassifier)`:
the preprocessor is applied first to get the transformed feature matrix
(with one-hot expanded names), then `shap.TreeExplainer` runs against
the booster.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def explain(pipe: Pipeline, X: pd.DataFrame, sample_n: int | None = 1000, seed: int = 42):
    """Compute SHAP values for a sample of X.

    Returns `(shap_explanation, X_transformed_df)`. SHAP values are in
    log-odds space (LightGBM default).
    """
    if sample_n is not None and sample_n < len(X):
        X = X.sample(n=sample_n, random_state=seed)

    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    Xt = pre.transform(X)
    feature_names = list(pre.get_feature_names_out())
    Xt_df = pd.DataFrame(Xt, columns=feature_names, index=X.index)

    explainer = shap.TreeExplainer(clf)
    explanation = explainer(Xt_df)
    return explanation, Xt_df


def global_importance(explanation) -> pd.Series:
    """Mean absolute SHAP value per feature, descending."""
    vals = np.abs(explanation.values)
    if vals.ndim == 3:
        # (n_samples, n_features, n_classes); take positive class.
        vals = vals[..., -1]
    mean_abs = vals.mean(axis=0)
    return pd.Series(mean_abs, index=explanation.feature_names).sort_values(ascending=False)


def pick_representative_indices(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, seed: int = 42) -> dict:
    """Pick three test rows for local SHAP waterfalls:
    - high-confidence default (y_true=1, very high prob),
    - high-confidence paid (y_true=0, very low prob),
    - borderline (prob within ±0.03 of threshold).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    high_default_pool = np.where((y_true == 1) & (y_prob > 0.85))[0]
    high_paid_pool = np.where((y_true == 0) & (y_prob < 0.05))[0]
    borderline_pool = np.where(np.abs(y_prob - threshold) < 0.03)[0]

    return {
        "clear_default": int(rng.choice(high_default_pool)) if len(high_default_pool) else None,
        "clear_paid": int(rng.choice(high_paid_pool)) if len(high_paid_pool) else None,
        "borderline": int(rng.choice(borderline_pool)) if len(borderline_pool) else None,
    }
