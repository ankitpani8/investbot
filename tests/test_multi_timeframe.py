"""
Contract and no-lookahead tests for investbot.scoring.technical.multi_timeframe.

Synthetic series only: no DB and no network.
"""
from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("talib")

from investbot.scoring.technical import multi_timeframe as M  # noqa: E402
from investbot.scoring.technical.multi_timeframe import (  # noqa: E402
    align_higher_timeframe,
    label_daily_bars,
    list_mtf_columns,
    momentum_regime,
    mtf_alignment,
    resample_ohlcv,
    trend_direction,
)

BARS_PER_DAY = 25
#: Small regime windows so the synthetic daily history can be short.
REGIME_KW = dict(hurst_windows=(60,), lyap_windows=(60,), regime_window=60)


def _intraday(days: pd.DatetimeIndex, seed: int = 3) -> pd.DataFrame:
    """15m bars labelled by open time (09:15 .. 15:15), like the DB."""
    opens = (days.repeat(BARS_PER_DAY)
             + pd.Timedelta("9h15min")
             + pd.to_timedelta(np.tile(np.arange(BARS_PER_DAY) * 15, len(days)), unit="min"))
    n = len(opens)
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    op = np.r_[close[0], close[:-1]]
    high = np.maximum(op, close) * (1 + rng.uniform(0, 0.002, n))
    low = np.minimum(op, close) * (1 - rng.uniform(0, 0.002, n))
    vol = rng.integers(1_000, 10_000, n).astype(np.float64)
    return pd.DataFrame({"open": op, "high": high, "low": low, "close": close,
                         "volume": vol}, index=pd.DatetimeIndex(opens, name="timestamp"))


def _daily(days: pd.DatetimeIndex, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, len(days))))
    op = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"open": op, "high": np.maximum(op, close) * 1.005,
                         "low": np.minimum(op, close) * 0.995, "close": close,
                         "volume": 1e6}, index=days)


DAILY_DAYS = pd.bdate_range("2025-01-01", periods=200)
INTRA_DAYS = DAILY_DAYS[-40:]


@pytest.fixture(scope="module")
def m15() -> pd.DataFrame:
    return _intraday(INTRA_DAYS)


@pytest.fixture(scope="module")
def daily() -> pd.DataFrame:
    return _daily(DAILY_DAYS)


@pytest.fixture(scope="module")
def full(m15, daily) -> pd.DataFrame:
    return mtf_alignment(m15, daily, regime_kwargs=REGIME_KW)


# ──────────────────────────────────────────────────────────
# Resampling
# ──────────────────────────────────────────────────────────

def test_resample_1h_counts_labels_and_ohlc(m15):
    two = m15.iloc[: 2 * BARS_PER_DAY]
    h1 = resample_ohlcv(two, "1h")
    assert len(h1) == 14                      # 6 full hours + 1 partial, two days
    labels = [t.strftime("%H:%M") for t in h1.index[:7]]
    assert labels == ["10:15", "11:15", "12:15", "13:15", "14:15", "15:15", "15:30"]
    assert h1.index[7] == INTRA_DAYS[1] + pd.Timedelta("10h15min")
    assert list(h1["BAR_COUNT"][:7]) == [4, 4, 4, 4, 4, 4, 1]

    for i, (start, stop) in enumerate([(0, 4), (4, 8), (24, 25), (25, 29)]):
        j = [0, 1, 6, 7][i]
        chunk = two.iloc[start:stop]
        row = h1.iloc[j]
        assert row["open"] == chunk["open"].iloc[0]
        assert row["high"] == chunk["high"].max()
        assert row["low"] == chunk["low"].min()
        assert row["close"] == chunk["close"].iloc[-1]
        assert row["volume"] == chunk["volume"].sum()


def test_resample_15min_relabels_by_close(m15):
    base = resample_ohlcv(m15, "15min")
    assert len(base) == len(m15)
    assert (base.index == m15.index + pd.Timedelta("15min")).all()
    np.testing.assert_array_equal(base["close"], m15["close"])
    assert base.index[BARS_PER_DAY - 1].time() == time(15, 30)
    # input already labelled by close gives the same bars
    closed = m15.set_axis(m15.index + pd.Timedelta("15min"))
    pd.testing.assert_frame_equal(resample_ohlcv(closed, "15min", input_label="close"), base)


def test_resample_missing_bars_keep_scheduled_labels(m15):
    day = m15.iloc[:BARS_PER_DAY]
    gappy = day.drop(day.index[[3, 24]])       # 10:00 bar and the 15:15 bar missing
    h1 = resample_ohlcv(gappy, "1h")
    assert len(h1) == 6                         # the partial 15:30 bin had only 15:15
    assert h1.index[0].time() == time(10, 15) and h1["BAR_COUNT"].iloc[0] == 3
    assert h1.index[-1].time() == time(15, 15)


def test_resample_forming_bin_is_labelled_in_the_future(m15):
    """A 1h bin with only 2 of its 15m bars is still labelled by its scheduled close."""
    h1 = resample_ohlcv(m15.iloc[:6], "1h")    # 09:15..10:30 opens → last close 10:45
    assert h1.index[-1].time() == time(11, 15) and h1["BAR_COUNT"].iloc[-1] == 2


def test_resample_validation(m15):
    with pytest.raises(ValueError, match="multiple"):
        resample_ohlcv(m15, "20min")
    with pytest.raises(ValueError, match="outside the session"):
        resample_ohlcv(m15.set_axis(m15.index - pd.Timedelta("1h")), "1h")
    with pytest.raises(ValueError, match="missing"):
        resample_ohlcv(m15.drop(columns="volume"), "1h")


# ──────────────────────────────────────────────────────────
# Alignment: the daily-close lookahead bug
# ──────────────────────────────────────────────────────────

def test_intraday_bars_never_see_same_day_daily_close(m15):
    """Daily close of day i is 1000 + i, so every value names its source day."""
    marker = pd.DataFrame({"close": 1000.0 + np.arange(len(DAILY_DAYS))}, index=DAILY_DAYS)
    base = resample_ohlcv(m15, "15min")
    aligned = align_higher_timeframe(base, label_daily_bars(marker), M.D1_PREFIX)

    seen = aligned["D1_close"].to_numpy()
    day_pos = DAILY_DAYS.get_indexer(base.index.normalize())
    assert (day_pos >= 1).all()
    np.testing.assert_array_equal(seen, 1000.0 + day_pos - 1)     # previous session only
    assert not (seen == 1000.0 + day_pos).any()
    # including the last bar of each day, which closes at 15:30 like the session
    last = base.index.time == time(15, 30)
    assert last.sum() == len(INTRA_DAYS)
    np.testing.assert_array_equal(seen[last], 1000.0 + day_pos[last] - 1)


def test_daily_alignment_skips_holidays_and_starts_nan():
    days = pd.DatetimeIndex(["2025-03-03", "2025-03-04", "2025-03-06"])   # 03-05 holiday
    m15 = _intraday(days)
    marker = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=days)
    out = align_higher_timeframe(resample_ohlcv(m15, "15min"), label_daily_bars(marker), "D1_")
    per_day = out["D1_close"].groupby(out.index.normalize()).agg(["min", "max"])
    assert per_day.iloc[0].isna().all()
    assert (per_day.iloc[1] == 1.0).all() and (per_day.iloc[2] == 2.0).all()


def test_hourly_bar_visible_only_once_closed(m15):
    base = resample_ohlcv(m15.iloc[:BARS_PER_DAY], "15min")
    h1 = resample_ohlcv(m15.iloc[:BARS_PER_DAY], "1h")
    out = align_higher_timeframe(base, h1[["close"]], M.H1_PREFIX)
    day = INTRA_DAYS[0]
    at = lambda hm: out.loc[day + pd.Timedelta(hm)]  # noqa: E731
    assert np.isnan(at("10h00min")["H1_close"])
    assert at("10h15min")["H1_close"] == h1["close"].iloc[0] == base.loc[day + pd.Timedelta("10h15min"), "close"]
    assert at("11h00min")["H1_close"] == h1["close"].iloc[0]
    assert at("11h15min")["H1_close"] == h1["close"].iloc[1]
    assert at("15h15min")["H1_close"] == h1["close"].iloc[5]
    assert at("15h30min")["H1_close"] == h1["close"].iloc[6]


def test_align_keeps_base_and_prefixes(m15):
    base = resample_ohlcv(m15, "15min")
    out = align_higher_timeframe(base, resample_ohlcv(m15, "1h"), "H1_")
    assert out.index.equals(base.index)
    assert list(out.columns[: base.shape[1]]) == list(base.columns)
    assert {"H1_open", "H1_close", "H1_BAR_COUNT"} <= set(out.columns)


# ──────────────────────────────────────────────────────────
# mtf_alignment
# ──────────────────────────────────────────────────────────

def test_mtf_columns_shape_and_ranges(m15, full):
    assert list(full.columns) == list_mtf_columns()
    assert len(full) == len(m15)
    assert all(full[c].dtype == np.float64 for c in full.columns)
    for col in ("MTF_TREND_ALIGN", "MTF_MOMENTUM_ALIGN"):
        v = full[col].dropna()
        assert len(v) > 0.5 * len(full) and v.between(-1, 1).all(), col
    assert set(full["MTF_REGIME_1D"].dropna().unique()) <= {0.0, 1.0, 2.0}
    assert full["MTF_REGIME_CONFIDENCE_1D"].dropna().between(0, 1).all()


def test_mtf_regime_is_previous_session(m15, daily, full):
    from investbot.scoring.technical.regime import compute_regime
    reg = pd.Series(compute_regime(daily, **REGIME_KW)["REGIME"], index=daily.index)
    day_pos = DAILY_DAYS.get_indexer(full.index.normalize())
    expected = reg.to_numpy()[day_pos - 1]
    np.testing.assert_array_equal(full["MTF_REGIME_1D"].to_numpy(), expected)


def test_mtf_regime_confidence_is_smoothed_previous_session():
    """MTF_REGIME_CONFIDENCE_1D is the EMA-smoothed daily confidence, not the raw gap."""
    from investbot.scoring.technical.regime import compute_regime
    days = pd.bdate_range("2024-01-01", periods=300)   # past the 250-bar VOL_LONG warmup
    daily, m15 = _daily(days), _intraday(days[-20:])
    out = mtf_alignment(m15, daily, regime_kwargs=REGIME_KW)
    reg = compute_regime(daily, **REGIME_KW)
    day_pos = days.get_indexer(out.index.normalize())
    got = out["MTF_REGIME_CONFIDENCE_1D"].to_numpy()
    np.testing.assert_array_equal(got, reg["REGIME_CONFIDENCE"][day_pos - 1])
    raw = reg["REGIME_CONFIDENCE_RAW"][day_pos - 1]
    assert np.isfinite(got).all() and np.isfinite(raw).all()
    assert not np.allclose(got, raw)


def test_mtf_align_is_weighted_mean_of_components(full):
    for kind in ("TREND", "MOMENTUM"):
        parts = full[[f"MTF_{kind}_{tf}" for tf in ("15M", "1H", "1D")]]
        expected = (parts.to_numpy() @ np.array(M.TIMEFRAME_WEIGHTS)) / sum(M.TIMEFRAME_WEIGHTS)
        np.testing.assert_allclose(full[f"MTF_{kind}_ALIGN"], expected)


def test_timeframe_weights_are_1_2_3():
    assert M.TIMEFRAME_WEIGHTS == (1.0, 2.0, 3.0)


def test_weighted_alignment_follows_daily_when_only_daily_disagrees():
    # 15m and 1h agree up (one of them only half-strength), daily trend is down.
    trends = [np.array([1.0, 0.5]), np.array([0.5, 1.0]), np.array([-1.0, -1.0])]
    got = M._weighted_mean(trends, M.TIMEFRAME_WEIGHTS)
    assert (got < 0).all()                                   # sign of the daily trend
    assert (M._weighted_mean(trends, (1.0, 1.0, 1.0)) > 0).all()   # equal weights said up
    # Full-strength intraday against full-strength daily is an exact tie.
    np.testing.assert_allclose(
        M._weighted_mean([np.array([1.0]), np.array([1.0]), np.array([-1.0])],
                         M.TIMEFRAME_WEIGHTS), 0.0, atol=1e-12)


def test_mtf_alignment_daily_downtrend_vs_intraday_uptrend():
    days = pd.bdate_range("2025-01-01", periods=120)
    m15 = _intraday(days[-15:])
    m15[["open", "high", "low", "close"]] = (
        np.linspace(100, 160, len(m15))[:, None] * np.array([0.999, 1.002, 0.998, 1.0]))
    d = _daily(days)
    d[["open", "high", "low", "close"]] = (
        np.linspace(100, 50, len(d))[:, None] * np.array([0.995, 1.01, 0.99, 1.0]))
    tail = mtf_alignment(m15, d, regime_kwargs=REGIME_KW).iloc[-BARS_PER_DAY:]
    assert (tail["MTF_TREND_1D"] == -1.0).all() and (tail["MTF_TREND_1H"] == 1.0).all()
    assert (tail["MTF_TREND_ALIGN"].abs() < 1e-12).all()


def test_component_rules():
    np.testing.assert_array_equal(
        trend_direction(np.array([2.0, 1.0, 2.0, np.nan]), np.array([1.0, 2.0, 1.0, 1.0]),
                        np.array([1.0, -1.0, -1.0, 1.0])),
        [1.0, -1.0, 0.0, np.nan])
    np.testing.assert_array_equal(
        momentum_regime(np.array([70.0, 50.0, 30.0, np.nan])), [1.0, 0.0, -1.0, np.nan])


def test_all_up_trend_aligns_to_plus_one():
    days = pd.bdate_range("2025-01-01", periods=120)
    m15 = _intraday(days[-15:])
    m15[["open", "high", "low", "close"]] = (
        np.linspace(100, 160, len(m15))[:, None] * np.array([0.999, 1.002, 0.998, 1.0]))
    d = _daily(days)
    d[["open", "high", "low", "close"]] = (
        np.linspace(50, 100, len(d))[:, None] * np.array([0.995, 1.01, 0.99, 1.0]))
    out = mtf_alignment(m15, d, regime_kwargs=REGIME_KW)
    tail = out.iloc[-BARS_PER_DAY:]
    assert (tail["MTF_TREND_ALIGN"] == 1.0).all()
    assert (tail["MTF_MOMENTUM_ALIGN"] == 1.0).all()


def test_inputs_not_mutated(m15, daily):
    a, b = m15.copy(), daily.copy()
    mtf_alignment(m15, daily, regime_kwargs=REGIME_KW)
    pd.testing.assert_frame_equal(m15, a)
    pd.testing.assert_frame_equal(daily, b)


# ──────────────────────────────────────────────────────────
# No lookahead: truncation invariance on the base timeframe
# ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("cut", [30, 200, 3 + 7 * BARS_PER_DAY, 24 + 20 * BARS_PER_DAY,
                                 BARS_PER_DAY * 40 - 1])
def test_no_lookahead_truncation_invariance(m15, daily, full, cut):
    """
    Cut the 15m data after bar ``cut`` and give daily data up to and
    including that bar's day (as a live run would hold once D's daily bar
    exists). No surviving row may change.
    """
    day = m15.index[cut - 1].normalize()
    trunc = mtf_alignment(m15.iloc[:cut], daily.loc[:day], regime_kwargs=REGIME_KW)
    ref = full.iloc[:cut]
    assert trunc.index.equals(ref.index)
    offenders = []
    for col in list_mtf_columns():
        a, b = trunc[col].to_numpy(), ref[col].to_numpy()
        ok = np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True)
        if not ok.all():
            bad = int(np.flatnonzero(~ok)[0])
            offenders.append(f"{col}@{bad}: truncated={a[bad]!r} full={b[bad]!r}")
    assert not offenders, "lookahead suspected in:\n  " + "\n  ".join(offenders)


def test_future_daily_rows_change_nothing(m15, daily, full):
    extra = _daily(pd.bdate_range(DAILY_DAYS[-1] + pd.Timedelta(days=1), periods=5), seed=9)
    more = pd.concat([daily, extra])
    pd.testing.assert_frame_equal(mtf_alignment(m15, more, regime_kwargs=REGIME_KW), full)
