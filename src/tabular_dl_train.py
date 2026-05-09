"""Training loop for the tabular deep-learning models.

Run as a CLI:
    python -m src.tabular_dl_train [--quick] [--save] [--model {ftt,mlp}]

`--quick` uses 5,000 rows and 30 epochs for a fast smoke test.
`--save` persists the trained model + test predictions to
`reports/artifacts/dl/`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from . import RANDOM_STATE
from .data import load_data
from .features import engineer_features, split_xy
from .tabular_dl import (
    FTTransformer,
    MLP,
    TabularDataset,
    fit_spec,
    get_device,
    predict_proba,
)


def _build_model(name: str, spec, **overrides):
    if name == "mlp":
        return MLP(spec, **overrides)
    if name == "ftt":
        return FTTransformer(spec, **overrides)
    raise ValueError(f"unknown model: {name!r}")


def _train_one(
    model_name: str,
    X_tr,
    y_tr,
    X_va,
    y_va,
    X_te,
    y_te,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    pos_weight: float,
    device: torch.device,
    verbose: bool = True,
) -> dict:
    spec = fit_spec(X_tr)
    train_ds = TabularDataset(X_tr, spec, y=y_tr.to_numpy())
    val_ds = TabularDataset(X_va, spec, y=y_va.to_numpy())
    test_ds = TabularDataset(X_te, spec, y=y_te.to_numpy())

    g = torch.Generator()
    g.manual_seed(RANDOM_STATE)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)

    model = _build_model(model_name, spec).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pos_weight_t = torch.tensor([pos_weight], device=device)

    best_pr_auc = -1.0
    best_state = None
    epochs_since_improve = 0
    history: list[dict] = []

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n = 0
        for batch in train_loader:
            num = batch["num"].to(device)
            cat = batch["cat"].to(device)
            y = batch["y"].to(device)
            opt.zero_grad()
            logits = model(num, cat)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight_t)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * y.shape[0]
            n += y.shape[0]
        sched.step()

        # validation
        val_prob = predict_proba(model, val_ds, batch_size=2048, device=device)
        val_pr = float(average_precision_score(y_va.to_numpy(), val_prob))
        val_roc = float(roc_auc_score(y_va.to_numpy(), val_prob))
        history.append({
            "epoch": epoch, "train_loss": epoch_loss / max(n, 1),
            "val_pr_auc": val_pr, "val_roc_auc": val_roc,
        })
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d}  train_loss={epoch_loss/n:.4f}  val_pr_auc={val_pr:.4f}  val_roc_auc={val_roc:.4f}")

        if val_pr > best_pr_auc + 1e-4:
            best_pr_auc = val_pr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                if verbose:
                    print(f"  early stop at epoch {epoch} (best val PR-AUC {best_pr_auc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_prob = predict_proba(model, test_ds, batch_size=2048, device=device)
    train_seconds = time.time() - t0

    return {
        "model_name": model_name,
        "model": model,
        "spec": spec,
        "y_prob": test_prob,
        "best_val_pr_auc": best_pr_auc,
        "test_pr_auc": float(average_precision_score(y_te.to_numpy(), test_prob)),
        "test_roc_auc": float(roc_auc_score(y_te.to_numpy(), test_prob)),
        "history": history,
        "train_seconds": train_seconds,
    }


def fit_and_evaluate_dl(
    quick: bool = False,
    models: tuple[str, ...] = ("mlp", "ftt"),
    epochs: int | None = None,
    seed: int = RANDOM_STATE,
    verbose: bool = True,
) -> dict:
    """Train MLP and FT-Transformer on the same split as the random-split
    LightGBM baseline. Returns a dict keyed by model name."""
    n = 5_000 if quick else None
    df = engineer_features(load_data())
    if n is not None:
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)

    X, y = split_xy(df)
    # 80/20 split; carve a 10% validation slice from train for early stopping
    X_trva, X_te, y_trva, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(X_trva, y_trva, test_size=0.125, stratify=y_trva, random_state=seed)

    device = get_device()
    pos_weight = float((1 - y_tr).sum() / max(y_tr.sum(), 1))

    if epochs is None:
        epochs = 30 if quick else 80
    common = dict(
        epochs=epochs, batch_size=256, lr=1e-3, weight_decay=1e-4,
        patience=10, pos_weight=pos_weight, device=device, verbose=verbose,
    )

    results: dict[str, dict] = {}
    for name in models:
        if verbose:
            print(f"\n=== training {name} on {device} ({len(X_tr):,} train / {len(X_va):,} val / {len(X_te):,} test) ===")
        r = _train_one(name, X_tr, y_tr, X_va, y_va, X_te, y_te, **common)
        results[name] = r

    return {
        "models": results,
        "splits": (X_tr, X_va, X_te, y_tr, y_va, y_te),
        "device": str(device),
        "pos_weight": pos_weight,
        "epochs_planned": epochs,
    }


def save_dl_artifacts(out: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _, _, X_te, _, _, y_te = out["splits"]
    X_te.to_csv(out_dir / "X_test.csv", index=False)
    y_te.to_frame("default").to_csv(out_dir / "y_test.csv", index=False)
    rows = []
    for name, r in out["models"].items():
        np.save(out_dir / f"y_prob_{name}.npy", r["y_prob"])
        # save weights with joblib (small models, simpler than torch.save for portability)
        torch.save({"state_dict": r["model"].state_dict(), "spec": r["spec"]},
                   out_dir / f"{name}.pt")
        joblib.dump(r["history"], out_dir / f"{name}_history.joblib")
        rows.append({
            "model": name,
            "test_pr_auc": round(r["test_pr_auc"], 4),
            "test_roc_auc": round(r["test_roc_auc"], 4),
            "best_val_pr_auc": round(r["best_val_pr_auc"], 4),
            "train_seconds": round(r["train_seconds"], 1),
        })
    table = pd.DataFrame(rows).set_index("model")
    table.to_csv(out_dir / "results.csv")
    json.dump({"device": out["device"], "pos_weight": out["pos_weight"],
               "epochs_planned": out["epochs_planned"]},
              open(out_dir / "meta.json", "w"), indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="5,000 rows + 30 epochs")
    parser.add_argument("--save", action="store_true", help="persist artifacts to reports/artifacts/dl/")
    parser.add_argument("--model", choices=("mlp", "ftt", "both"), default="both")
    args = parser.parse_args()

    models = ("mlp", "ftt") if args.model == "both" else (args.model,)
    out = fit_and_evaluate_dl(quick=args.quick, models=models)

    rows = []
    for name, r in out["models"].items():
        rows.append({
            "model": name,
            "test_pr_auc": round(r["test_pr_auc"], 4),
            "test_roc_auc": round(r["test_roc_auc"], 4),
            "best_val_pr_auc": round(r["best_val_pr_auc"], 4),
            "train_seconds": round(r["train_seconds"], 1),
        })
    print()
    print(pd.DataFrame(rows).set_index("model"))

    if args.save:
        out_dir = Path(__file__).resolve().parent.parent / "reports" / "artifacts" / "dl"
        save_dl_artifacts(out, out_dir)
        print(f"\nArtifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
