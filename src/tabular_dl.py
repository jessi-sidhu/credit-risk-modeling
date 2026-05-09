"""Deep-learning models for tabular credit data, in pure PyTorch.

Two architectures:

1. `MLP` — a vanilla feedforward baseline. Categorical features go
   through learned embeddings, numerical features through a linear
   projection, the two are concatenated and fed into a 3-layer MLP.

2. `FTTransformer` — Feature Tokenizer + Transformer (Gorishniy et al.
   2021). Each feature, numerical or categorical, becomes a single
   d_token-dimensional token. A learned [CLS] token is prepended and
   the stack of L Transformer blocks attends across all tokens. The
   final [CLS] embedding feeds a small classification head.

Both consume a `TabularDataset` that the training loop builds from a
`pandas.DataFrame` plus the categorical/numerical column lists from
`features.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .features import CATEGORICAL, NUMERIC_ALL


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class TabularSpec:
    """Schema info needed to build a model from a fitted preprocessor."""
    numeric_cols: list[str]
    categorical_cols: list[str]
    cat_cardinalities: list[int]   # number of unique values per categorical
    cat_to_idx: list[dict]         # per-column value -> index lookup
    num_mean: np.ndarray           # per-numeric-feature mean (from train)
    num_std: np.ndarray            # per-numeric-feature std (from train)


def fit_spec(
    train_df: pd.DataFrame,
    numeric_cols: Sequence[str] = tuple(NUMERIC_ALL),
    categorical_cols: Sequence[str] = tuple(CATEGORICAL),
) -> TabularSpec:
    """Compute the schema (categorical vocabularies + numeric scaling)
    from the *training* fold only. The same spec is then used to encode
    the test fold so there's no leakage."""
    cat_to_idx = []
    cat_cardinalities = []
    for c in categorical_cols:
        vals = sorted(train_df[c].astype(str).unique().tolist())
        # reserve index 0 for unseen categories at inference time
        mapping = {v: i + 1 for i, v in enumerate(vals)}
        cat_to_idx.append(mapping)
        cat_cardinalities.append(len(vals) + 1)

    num = train_df[list(numeric_cols)].to_numpy(dtype=np.float32)
    mean = num.mean(axis=0)
    std = num.std(axis=0)
    std = np.where(std == 0, 1.0, std)  # avoid div-by-zero

    return TabularSpec(
        numeric_cols=list(numeric_cols),
        categorical_cols=list(categorical_cols),
        cat_cardinalities=cat_cardinalities,
        cat_to_idx=cat_to_idx,
        num_mean=mean.astype(np.float32),
        num_std=std.astype(np.float32),
    )


class TabularDataset(Dataset):
    def __init__(self, df: pd.DataFrame, spec: TabularSpec, y: np.ndarray | None = None):
        # numerics: standardized
        num = df[spec.numeric_cols].to_numpy(dtype=np.float32)
        num = (num - spec.num_mean) / spec.num_std
        # impute remaining NaNs (real LC data) with zero (mean post-standardize)
        num = np.nan_to_num(num, nan=0.0)
        self.num = torch.from_numpy(num)

        # categoricals: integer-encoded, 0 for unseen
        cat_arrays = []
        for col, mapping in zip(spec.categorical_cols, spec.cat_to_idx):
            arr = df[col].astype(str).map(mapping).fillna(0).to_numpy(dtype=np.int64)
            cat_arrays.append(arr)
        self.cat = torch.from_numpy(np.stack(cat_arrays, axis=1)) if cat_arrays else torch.zeros((len(df), 0), dtype=torch.long)

        self.y = torch.from_numpy(y.astype(np.float32)) if y is not None else None

    def __len__(self) -> int:
        return self.num.shape[0]

    def __getitem__(self, idx: int):
        item = {"num": self.num[idx], "cat": self.cat[idx]}
        if self.y is not None:
            item["y"] = self.y[idx]
        return item


# ---------------------------------------------------------------------------
# MLP baseline
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Embedding-MLP baseline."""

    def __init__(
        self,
        spec: TabularSpec,
        embed_dim: int = 16,
        hidden_dims: Sequence[int] = (128, 64, 32),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.spec = spec
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, embed_dim) for card in spec.cat_cardinalities
        ])
        n_num = len(spec.numeric_cols)
        n_cat_total = embed_dim * len(spec.cat_cardinalities)
        in_dim = n_num + n_cat_total

        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, num: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        embs = [emb(cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([num] + embs, dim=-1)
        return self.mlp(x).squeeze(-1)  # logits


# ---------------------------------------------------------------------------
# FT-Transformer
# ---------------------------------------------------------------------------


class _NumericalTokenizer(nn.Module):
    """Each numerical feature becomes a d-dim token via a per-feature
    weight + bias (Gorishniy et al. 2021, eq. 2)."""

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.bias, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features) -> (B, n_features, d_token)
        return x.unsqueeze(-1) * self.weight + self.bias


class _CategoricalTokenizer(nn.Module):
    """Per-column embedding table; outputs a (B, n_cat, d_token) tensor."""

    def __init__(self, cat_cardinalities: Sequence[int], d_token: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, d_token) for card in cat_cardinalities]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [emb(x[:, i]) for i, emb in enumerate(self.embeddings)], dim=1
        )


class _TransformerBlock(nn.Module):
    """Pre-norm Transformer block: MHA + FFN, both residual."""

    def __init__(self, d_token: int, n_heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_token * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token * ffn_mult, d_token),
        )
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn_norm(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.attn_drop(attn_out)

        h = self.ffn_norm(x)
        x = x + self.ffn_drop(self.ffn(h))
        return x


class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer (Gorishniy et al. 2021).

    Each feature -> d_token-dim token. Prepend [CLS]. Apply L Transformer
    blocks. Use the final [CLS] embedding for classification.
    """

    def __init__(
        self,
        spec: TabularSpec,
        d_token: int = 64,
        n_blocks: int = 3,
        n_heads: int = 8,
        ffn_mult: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.spec = spec
        self.num_tokenizer = _NumericalTokenizer(len(spec.numeric_cols), d_token)
        self.cat_tokenizer = _CategoricalTokenizer(spec.cat_cardinalities, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList(
            [_TransformerBlock(d_token, n_heads, ffn_mult, dropout) for _ in range(n_blocks)]
        )
        self.head_norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )

    def forward(self, num: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        # tokens: (B, n_features, d_token)
        num_tok = self.num_tokenizer(num)
        if cat.shape[1] > 0:
            cat_tok = self.cat_tokenizer(cat)
            tok = torch.cat([num_tok, cat_tok], dim=1)
        else:
            tok = num_tok

        cls = self.cls_token.expand(tok.shape[0], -1, -1)
        x = torch.cat([cls, tok], dim=1)

        for block in self.blocks:
            x = block(x)

        cls_out = self.head_norm(x[:, 0])
        return self.head(cls_out).squeeze(-1)  # logits


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    """Prefer Apple Silicon MPS, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def predict_proba(model: nn.Module, dataset: TabularDataset, batch_size: int = 1024,
                  device: torch.device | None = None) -> np.ndarray:
    """Return P(default=1) on a dataset; same shape as sklearn's predict_proba()[:, 1]."""
    device = device or get_device()
    model = model.to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch_idx = list(range(i, min(i + batch_size, len(dataset))))
            num = torch.stack([dataset[j]["num"] for j in batch_idx]).to(device)
            cat = torch.stack([dataset[j]["cat"] for j in batch_idx]).to(device)
            logits = model(num, cat)
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)
