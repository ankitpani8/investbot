"""
Contract tests for investbot.scoring.technical.candle_patterns.

Synthetic OHLCV only: no DB and no network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

talib = pytest.importorskip("talib")

from investbot.scoring.technical.candle_patterns import (  # noqa: E402
    NET_SCORE_SCALE,
    PATTERN_FUNCTIONS,
    add_candle_patterns,
    compute_candle_patterns,
    list_candle_columns,
    max_lookback,
    pattern_lookbacks,
)

AGGREGATES = ("CDL_BULL_COUNT", "CDL_BEAR_COUNT", "CDL_NET_SCORE")
RAW_VALUES = {-200.0, -100.0, -80.0, 0.0, 80.0, 100.0, 200.0}


def _synthetic_ohlcv(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.018, n)
    close = 1000 * np.exp(np.cumsum(rets))
    # Open near the previous close, with gaps sometimes, so gap patterns can fire.
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.006, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.007, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.007, n)))
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
    return add_candle_patterns(ohlcv)


def _raw_cols() -> list[str]:
    return [c for c in list_candle_columns() if c.startswith("CDL_") and c not in AGGREGATES]


def _norm_cols() -> list[str]:
    return [c for c in list_candle_columns() if c.startswith("CDLN_")]


def test_runs_every_talib_pattern():
    assert set(PATTERN_FUNCTIONS) == set(talib.get_function_groups()["Pattern Recognition"])
    assert all(name.startswith("CDL") for name in PATTERN_FUNCTIONS)


def test_column_count(ohlcv, full):
    out = compute_candle_patterns(ohlcv)
    assert len(out) == 2 * len(PATTERN_FUNCTIONS) + 3
    assert list(out) == list_candle_columns()
    assert list(full.columns) == list(ohlcv.columns) + list_candle_columns()


def test_dict_of_float_arrays(ohlcv):
    for name, arr in compute_candle_patterns(ohlcv).items():
        assert isinstance(arr, np.ndarray), name
        assert arr.dtype == np.float64, name
        assert arr.shape == (len(ohlcv),), name


def test_row_count_and_index_preserved(ohlcv, full):
    assert len(full) == len(ohlcv)
    assert full.index.equals(ohlcv.index)


def test_input_not_mutated(ohlcv):
    before = ohlcv.copy()
    add_candle_patterns(ohlcv)
    pd.testing.assert_frame_equal(ohlcv, before)


def test_raw_values_in_allowed_set(full):
    for col in _raw_cols():
        vals = set(full[col].dropna().unique())
        assert vals <= RAW_VALUES, f"{col}: unexpected {vals - RAW_VALUES}"


def test_normalised_is_sign_of_raw(full):
    for col in _norm_cols():
        raw = full["CDL_" + col[len("CDLN_"):]]
        assert set(full[col].dropna().unique()) <= {-1.0, 0.0, 1.0}, col
        np.testing.assert_array_equal(full[col].to_numpy(), np.sign(raw.to_numpy()), err_msg=col)


def test_aggregates_consistent(full):
    valid = full.iloc[max_lookback():]
    norm = valid[_norm_cols()]
    np.testing.assert_array_equal(valid["CDL_BULL_COUNT"], (norm > 0).sum(axis=1))
    np.testing.assert_array_equal(valid["CDL_BEAR_COUNT"], (norm < 0).sum(axis=1))

    raw = valid[_raw_cols()].to_numpy()
    weights = np.where(np.abs(raw) >= 200, 1.0, 0.5)
    expected = np.clip((np.sign(raw) * weights).sum(axis=1) / NET_SCORE_SCALE, -1, 1)
    np.testing.assert_allclose(valid["CDL_NET_SCORE"], expected)
    assert valid["CDL_NET_SCORE"].between(-1, 1).all()
    assert valid[list(AGGREGATES)].notna().all().all()


def test_synthetic_data_exercises_patterns(full):
    # Guard against a vacuous suite: plenty of patterns must actually fire.
    fired = [c for c in _norm_cols() if (full[c].abs() > 0).any()]
    assert len(fired) >= 20, f"only {len(fired)} patterns fired"
    assert (full["CDL_NET_SCORE"].dropna() != 0).any()


def test_warmup_is_nan_not_zero(full):
    lookbacks = pattern_lookbacks()
    for name in PATTERN_FUNCTIONS:
        col = full["CDL_" + name[len("CDL"):]]
        lb = lookbacks[name]
        assert col.iloc[:lb].isna().all(), name
        assert col.iloc[lb:].notna().all(), name
    for col in AGGREGATES:
        assert full[col].iloc[: max_lookback()].isna().all(), col


@pytest.mark.parametrize("cut", [30, 150, 333, 599])
def test_no_lookahead_truncation_invariance(ohlcv, full, cut):
    """Values on bars < cut must not change when the bars after them are removed."""
    trunc = pd.DataFrame(compute_candle_patterns(ohlcv.iloc[:cut]), index=ohlcv.index[:cut])
    ref = full[trunc.columns].iloc[:cut]
    pd.testing.assert_frame_equal(trunc, ref)


@pytest.mark.parametrize("start", [1, 57, 200])
def test_start_invariance_after_warmup(ohlcv, full, start):
    """Dropping old history must not change values once warmup is over again."""
    late = add_candle_patterns(ohlcv.iloc[start:])
    cols = list_candle_columns()
    pd.testing.assert_frame_equal(
        late[cols].iloc[max_lookback():], full[cols].iloc[start + max_lookback():]
    )


def _engulfing_fixture() -> pd.DataFrame:
    """A downtrend of small red candles, then a large green candle engulfing the last one."""
    rows = []
    price = 120.0
    for _ in range(20):
        o, c = price, price - 1.0
        rows.append((o, o + 0.2, c - 0.2, c))
        price = c
    # Last red candle: open 100.5, close 99.5 (body 1.0).
    rows.append((100.5, 100.7, 99.3, 99.5))
    # Bullish engulfing: opens below the prior close, closes above the prior open.
    rows.append((99.0, 103.2, 98.8, 103.0))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["volume"] = 1_000_000.0
    df.index = pd.bdate_range("2024-01-01", periods=len(df))
    return df


def test_bullish_engulfing_fixture():
    out = add_candle_patterns(_engulfing_fixture())
    assert out["CDLN_ENGULFING"].iloc[-1] == 1.0
    assert out["CDL_ENGULFING"].iloc[-1] == 100.0
    assert out["CDL_BULL_COUNT"].iloc[-1] >= 1
    # No assertion on CDL_NET_SCORE: TA-Lib also flags this bar as a -100
    # 3LINESTRIKE (it signs that pattern by the three prior candles), so net is 0.


def test_bearish_engulfing_mirror():
    bull = _engulfing_fixture()
    # Reflect prices: a downtrend of red candles becomes an uptrend of green ones.
    bear = bull.copy()
    bear[["open", "close"]] = 300.0 - bull[["open", "close"]].to_numpy()
    bear["high"] = 300.0 - bull["low"]
    bear["low"] = 300.0 - bull["high"]
    out = add_candle_patterns(bear)
    assert out["CDLN_ENGULFING"].iloc[-1] == -1.0


def test_nan_bar_is_nan_and_does_not_leak_backwards(ohlcv, full):
    bad = ohlcv.copy()
    pos = 300
    bad.iloc[pos, bad.columns.get_loc("close")] = np.nan
    out = add_candle_patterns(bad)
    cols = list_candle_columns()
    assert out[cols].iloc[pos].isna().all()
    pd.testing.assert_frame_equal(out[cols].iloc[:pos], full[cols].iloc[:pos])


def test_missing_ohlc_column_raises(ohlcv):
    with pytest.raises(ValueError, match="close"):
        compute_candle_patterns(ohlcv.drop(columns=["close"]))
