import numpy as np
import pandas as pd
import torch

from src.data import load_data
from src.features import engineer_features
from src.tabular_dl import (
    FTTransformer,
    MLP,
    TabularDataset,
    fit_spec,
    predict_proba,
)


def _small_df(n: int = 200) -> pd.DataFrame:
    return engineer_features(load_data().head(n))


def test_fit_spec_includes_unseen_slot():
    df = _small_df()
    spec = fit_spec(df)
    # one extra slot for unseen category at inference (index 0)
    for col, mapping, card in zip(spec.categorical_cols, spec.cat_to_idx, spec.cat_cardinalities):
        assert card == len(mapping) + 1
        assert 0 not in mapping.values()


def test_dataset_shapes():
    df = _small_df()
    spec = fit_spec(df)
    y = df["default"].to_numpy()
    ds = TabularDataset(df.drop(columns=["default"]), spec, y=y)
    item = ds[0]
    assert item["num"].shape == (len(spec.numeric_cols),)
    assert item["cat"].shape == (len(spec.categorical_cols),)
    assert item["y"].dtype == torch.float32


def test_mlp_forward_pass_and_grad():
    df = _small_df()
    spec = fit_spec(df)
    y = df["default"].to_numpy()
    ds = TabularDataset(df.drop(columns=["default"]), spec, y=y)
    model = MLP(spec)
    num = torch.stack([ds[i]["num"] for i in range(8)])
    cat = torch.stack([ds[i]["cat"] for i in range(8)])
    logits = model(num, cat)
    assert logits.shape == (8,)
    loss = logits.sum()
    loss.backward()
    # at least one weight has a gradient
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_ftt_forward_pass_and_grad():
    df = _small_df()
    spec = fit_spec(df)
    ds = TabularDataset(df.drop(columns=["default"]), spec, y=df["default"].to_numpy())
    model = FTTransformer(spec, d_token=32, n_blocks=2, n_heads=4)
    num = torch.stack([ds[i]["num"] for i in range(8)])
    cat = torch.stack([ds[i]["cat"] for i in range(8)])
    logits = model(num, cat)
    assert logits.shape == (8,)
    logits.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_overfit_tiny_batch():
    """A bigger sanity test: the model should fit a 32-row batch to near zero loss
    in a few hundred steps. If it can't overfit, something is wrong with the
    architecture or the optimizer hookup."""
    torch.manual_seed(0)
    df = _small_df(n=32)
    spec = fit_spec(df)
    y = df["default"].to_numpy()
    ds = TabularDataset(df.drop(columns=["default"]), spec, y=y)
    num = torch.stack([ds[i]["num"] for i in range(32)])
    cat = torch.stack([ds[i]["cat"] for i in range(32)])
    yt = torch.tensor(y, dtype=torch.float32)

    model = MLP(spec, hidden_dims=(64, 32))
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    losses = []
    for _ in range(400):
        opt.zero_grad()
        logits = model(num, cat)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] / 2, f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    assert losses[-1] < 0.3, f"final loss too high: {losses[-1]:.4f}"


def test_predict_proba_in_unit_interval():
    df = _small_df()
    spec = fit_spec(df)
    ds = TabularDataset(df.drop(columns=["default"]), spec, y=df["default"].to_numpy())
    model = FTTransformer(spec, d_token=16, n_blocks=1, n_heads=4)
    probs = predict_proba(model, ds, batch_size=64, device=torch.device("cpu"))
    assert probs.shape == (len(ds),)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_unseen_category_does_not_crash():
    df = _small_df()
    spec = fit_spec(df)
    df_test = df.head(5).copy()
    df_test.loc[df_test.index[0], "purpose"] = "totally_made_up"
    ds = TabularDataset(df_test.drop(columns=["default"]), spec, y=df_test["default"].to_numpy())
    # unseen -> mapped to index 0 (the reserved unknown slot)
    assert ds.cat[0, spec.categorical_cols.index("purpose")].item() == 0
