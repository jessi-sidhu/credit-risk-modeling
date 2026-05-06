import pandas as pd

from src.data import load_data, temporal_split
from src.synth import generate


def test_temporal_split_partitions_disjointly():
    df = generate(n=5_000, seed=42)
    tr, va, te = temporal_split(df, train_until="2017-06", test_from="2018-01")

    assert len(tr) + len(va) + len(te) == len(df)
    assert tr["issue_d"].max() < va["issue_d"].min() if len(va) else True
    assert va["issue_d"].max() < te["issue_d"].min() if len(va) else True
    assert tr["issue_d"].max() < te["issue_d"].min()


def test_temporal_split_train_only_through_2017_06():
    df = generate(n=5_000, seed=42)
    tr, _, _ = temporal_split(df, train_until="2017-06", test_from="2018-01")
    # train cohort end is end-of-month 2017-06
    assert tr["issue_d"].dt.to_period("M").max() == pd.Period("2017-06", freq="M")


def test_temporal_split_no_row_overlap():
    df = generate(n=5_000, seed=42).reset_index(drop=True)
    df["row_id"] = range(len(df))
    tr, va, te = temporal_split(df)
    ids = pd.concat([tr["row_id"], va["row_id"], te["row_id"]])
    assert ids.is_unique


def test_temporal_split_raises_when_column_missing():
    df = generate(n=100, seed=42).drop(columns=["issue_d"])
    try:
        temporal_split(df)
    except ValueError as e:
        assert "issue_d" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_load_data_then_temporal_split_consistent():
    """Round-trip via load_data (which round-trips through CSV) so we
    catch any dtype regression on issue_d after CSV read."""
    df = load_data(regenerate=True, n=2_000)
    tr, _, te = temporal_split(df, train_until="2017-06", test_from="2018-01")
    assert len(tr) > 0 and len(te) > 0
