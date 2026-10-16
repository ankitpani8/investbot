"""
Contract tests for investbot.scoring.technical.indicators.

These run on synthetic OHLCV data, so they need no DB and no network.
They are the guardrail autonomous agents must keep green.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

talib = pytest.importorskip("talib")

from investbot.scoring.technical.indicators import (  # noqa: E402
    add_all_indicators,
    list_indicator_columns,
)

MIN_EXPECTED_INDICATORS = 90


def _synthetic_ohlcv(n: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.015, n)
    close = 1000 * np.exp(np.cumsum(rets))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    volume = rng.integers(200_000, 5_000_000, n).astype(float)
    idx = pd.bdate_range("2021-01-01", periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    return _synthetic_ohlcv()


@pytest.fixture(scope="module")
def full(ohlcv) -> pd.DataFrame:
    return add_all_indicators(ohlcv)


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=["number", "bool"]).astype(float)


def test_indicator_count(full):
    cols = list_indicator_columns(full)
    assert len(cols) >= MIN_EXPECTED_INDICATORS, f"only {len(cols)} indicator columns"


def test_no_rows_dropped(ohlcv, full):
    # Rule: never dropna() inside compute functions.
    assert len(full) == len(ohlcv)
    assert full.index.equals(ohlcv.index)


def test_input_not_mutated(ohlcv):
    before = ohlcv.copy()
    add_all_indicators(ohlcv)
    pd.testing.assert_frame_equal(ohlcv, before)


def test_no_all_nan_columns(full):
    cols = list_indicator_columns(full)
    all_nan = [c for c in cols if full[c].isna().all()]
    assert not all_nan, f"all-NaN columns: {all_nan}"


@pytest.mark.parametrize(
    "col", ["RSI_14", "ADX_14", "STOCHk_14_3_3", "STOCHd_14_3_3", "MFI_14"]
)
def test_bounded_oscillators(full, col):
    if col not in full.columns:
        pytest.skip(f"{col} not produced")
    vals = full[col].dropna()
    assert vals.between(-0.01, 100.01).all(), f"{col} out of [0, 100]"


@pytest.mark.parametrize("cut", [250, 400, 599])
def test_no_lookahead_truncation_invariance(ohlcv, full, cut):
    """
    The value at bar t must not change when future bars are removed.
    Any column that fails is using future data, which is a lookahead bug.
    """
    trunc = _numeric(add_all_indicators(ohlcv.iloc[:cut]))
    ref = _numeric(full).iloc[cut - 1]
    last = trunc.iloc[-1]

    offenders = []
    for col in trunc.columns:
        a, b = last[col], ref[col]
        if np.isnan(a) and np.isnan(b):
            continue
        if np.isnan(a) != np.isnan(b) or not np.isclose(a, b, rtol=1e-6, atol=1e-8):
            offenders.append(f"{col}: truncated={a!r} full={b!r}")

    assert not offenders, "lookahead suspected in:\n  " + "\n  ".join(offenders)
