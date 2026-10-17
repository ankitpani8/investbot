"""
Tests for investbot.scoring.technical.support_resistance.

Synthetic data only: no DB, no network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from investbot.scoring.technical.support_resistance import (
    FIB_RATIOS,
    PIVOT_RIGHT_BARS,
    SR_COLUMNS,
    SR_RECENCY_HALFLIFE,
    VP_WINDOW_DAILY,
    add_support_resistance,
    compute_support_resistance_columns,
)


def _random_walk(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 1000 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    volume = rng.integers(200_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.bdate_range("2021-01-01", periods=n),
    )


# Zig-zag through known vertices, 10 bars per leg. Each bar has a ±0.5 range
# around the close, so pivot highs sit at vertex + 0.5 and lows at vertex - 0.5.
ZIGZAG_VERTICES = [100.0, 110.0, 100.0, 110.3, 100.2, 120.0, 105.0, 115.0, 108.0]
LEG = 10


def _zigzag(n: int = 78) -> pd.DataFrame:
    xs = np.arange(len(ZIGZAG_VERTICES)) * LEG
    close = np.interp(np.arange(n), xs, ZIGZAG_VERTICES)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1000.0),
        },
        index=pd.bdate_range("2024-01-01", periods=n),
    )


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    return _random_walk()


@pytest.fixture(scope="module")
def full(ohlcv) -> pd.DataFrame:
    return pd.DataFrame(compute_support_resistance_columns(ohlcv), index=ohlcv.index)


# ──────────────────────────────────────────────────────────
# Contract
# ──────────────────────────────────────────────────────────

def test_all_columns_present_and_populated(full):
    assert list(full.columns) == list(SR_COLUMNS)
    all_nan = [c for c in SR_COLUMNS if full[c].isna().all()]
    assert not all_nan, f"all-NaN columns: {all_nan}"


def test_add_keeps_rows_and_does_not_mutate(ohlcv):
    before = ohlcv.copy()
    out = add_support_resistance(ohlcv)
    pd.testing.assert_frame_equal(ohlcv, before)
    assert out.index.equals(ohlcv.index)
    pd.testing.assert_frame_equal(out[list(ohlcv.columns)], before)


def test_rejects_non_datetime_index(ohlcv):
    with pytest.raises(TypeError):
        compute_support_resistance_columns(ohlcv.reset_index(drop=True))


def test_vp_warmup_is_nan_not_dropped(full):
    assert full["VP_POC"].iloc[: VP_WINDOW_DAILY - 1].isna().all()
    assert full["VP_POC"].iloc[VP_WINDOW_DAILY - 1:].notna().all()


def test_bounds(ohlcv, full):
    pos = full["VP_POS"].dropna()
    assert pos.between(-1, 1).all()
    vp = full[["VP_VAL", "VP_POC", "VP_VAH"]].dropna()
    assert (vp["VP_VAL"] <= vp["VP_POC"]).all() and (vp["VP_POC"] <= vp["VP_VAH"]).all()
    assert (full["SR_DIST_SUPPORT_PCT"].dropna() >= 0).all()
    assert (full["SR_DIST_RESIST_PCT"].dropna() > 0).all()
    s = full[["SR_SUPPORT_1", "SR_SUPPORT_2", "SR_SUPPORT_3"]].dropna()
    assert (s["SR_SUPPORT_1"] > s["SR_SUPPORT_2"]).all()
    assert (s["SR_SUPPORT_2"] > s["SR_SUPPORT_3"]).all()
    r = full[["SR_RESIST_1", "SR_RESIST_2", "SR_RESIST_3"]].dropna()
    assert (r["SR_RESIST_1"] < r["SR_RESIST_2"]).all()
    assert (r["SR_RESIST_2"] < r["SR_RESIST_3"]).all()


# ──────────────────────────────────────────────────────────
# No lookahead: the key test for this module
# ──────────────────────────────────────────────────────────

def _assert_prefix_equal(trunc: pd.DataFrame, full: pd.DataFrame, cut: int) -> None:
    offenders = []
    for col in SR_COLUMNS:
        a = trunc[col].to_numpy()
        b = full[col].to_numpy()[:cut]
        same_nan = np.isnan(a) == np.isnan(b)
        close = np.isclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)
        if not (same_nan.all() and close.all()):
            bad = int(np.flatnonzero(~(same_nan & close))[0])
            offenders.append(f"{col}@{bad}: truncated={a[bad]!r} full={b[bad]!r}")
    assert not offenders, "lookahead suspected in:\n  " + "\n  ".join(offenders)


@pytest.mark.parametrize("cut", [10, 125, 200, 333, 450, 599])
def test_no_lookahead_truncation_invariance(ohlcv, full, cut):
    """Every row of every column must be unchanged when future bars are removed."""
    trunc = pd.DataFrame(
        compute_support_resistance_columns(ohlcv.iloc[:cut]), index=ohlcv.index[:cut]
    )
    _assert_prefix_equal(trunc, full, cut)


@pytest.mark.parametrize("cut", [150, 260, 399])
def test_no_lookahead_intraday_window(cut):
    df = _random_walk(400, seed=11)
    full = pd.DataFrame(compute_support_resistance_columns(df, vp_window=100), index=df.index)
    trunc = pd.DataFrame(
        compute_support_resistance_columns(df.iloc[:cut], vp_window=100), index=df.index[:cut]
    )
    _assert_prefix_equal(trunc, full, cut)


@pytest.mark.parametrize("cut", [40, 77])
def test_no_lookahead_zigzag(cut):
    df = _zigzag()
    full = pd.DataFrame(compute_support_resistance_columns(df, vp_window=20), index=df.index)
    trunc = pd.DataFrame(
        compute_support_resistance_columns(df.iloc[:cut], vp_window=20), index=df.index[:cut]
    )
    _assert_prefix_equal(trunc, full, cut)


# ──────────────────────────────────────────────────────────
# Zig-zag fixture with known pivots
# ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def zz() -> pd.DataFrame:
    return pd.DataFrame(compute_support_resistance_columns(_zigzag(), vp_window=20),
                        index=_zigzag().index)


def test_pivots_emitted_at_confirmation_bar_only(zz):
    k = PIVOT_RIGHT_BARS
    expected_highs = {10 + k: 110.5, 30 + k: 110.8, 50 + k: 120.5, 70 + k: 115.5}
    expected_lows = {20 + k: 99.5, 40 + k: 99.7, 60 + k: 104.5}
    ph = zz["PIVOT_HIGH"].to_numpy()
    pl = zz["PIVOT_LOW"].to_numpy()
    assert {int(i): round(ph[i], 6) for i in np.flatnonzero(~np.isnan(ph))} == expected_highs
    assert {int(i): round(pl[i], 6) for i in np.flatnonzero(~np.isnan(pl))} == expected_lows
    # The swing bar itself never carries the pivot.
    assert np.isnan(ph[10]) and np.isnan(pl[20])


def test_sr_levels_cluster_and_rank(zz):
    t = 77                     # close = 110.1, all seven pivots confirmed
    row = zz.iloc[t]
    h = SR_RECENCY_HALFLIFE
    # 99.5 (bar 20) and 99.7 (bar 40) are within 0.5% → one level.
    w20, w40 = 0.5 ** ((40 - 20) / h), 1.0
    low_cluster = (99.5 * w20 + 99.7 * w40) / (w20 + w40)
    w10, w30 = 0.5 ** ((30 - 10) / h), 1.0
    high_cluster = (110.5 * w10 + 110.8 * w30) / (w10 + w30)

    assert row["SR_SUPPORT_1"] == pytest.approx(104.5)
    assert row["SR_SUPPORT_2"] == pytest.approx(low_cluster)
    assert np.isnan(row["SR_SUPPORT_3"])
    assert row["SR_RESIST_1"] == pytest.approx(high_cluster)
    assert row["SR_RESIST_2"] == pytest.approx(115.5)
    assert row["SR_RESIST_3"] == pytest.approx(120.5)

    close = 110.1
    assert row["SR_DIST_SUPPORT_PCT"] == pytest.approx((close - 104.5) / close * 100)
    assert row["SR_DIST_RESIST_PCT"] == pytest.approx((high_cluster - close) / close * 100)
    assert row["SR_STRENGTH_SUPPORT"] == pytest.approx(0.5 ** ((t - 60) / h))
    assert row["SR_STRENGTH_RESIST"] == pytest.approx(
        0.5 ** ((t - 10) / h) + 0.5 ** ((t - 30) / h)
    )


def test_fibonacci_most_recent_swing(zz):
    # At bar 77 the latest swing is low 104.5 (bar 60) → high 115.5 (bar 70): up.
    row = zz.iloc[77]
    assert row["FIB_SWING_DIR"] == 1
    assert row["FIB_SWING_HIGH"] == pytest.approx(115.5)
    assert row["FIB_SWING_LOW"] == pytest.approx(104.5)
    for name, r in FIB_RATIOS.items():
        assert row[name] == pytest.approx(115.5 - r * 11.0)
    levels = np.array([row[n] for n in FIB_RATIOS])
    nearest = levels[np.argmin(np.abs(110.1 - levels))]
    assert row["FIB_NEAREST_DIST_PCT"] == pytest.approx((110.1 - nearest) / 110.1 * 100)

    # At bar 65 (low at 60 confirmed at 63, high at 70 not yet) the swing is
    # high 120.5 → low 104.5: down, retracements measured up from the low.
    row = zz.iloc[65]
    assert row["FIB_SWING_DIR"] == -1
    assert row["FIB_382"] == pytest.approx(104.5 + 0.382 * 16.0)


def test_fibonacci_merges_same_type_pivots():
    # Peaks at bars 10, 25 and 40 with a flat low series in between, so no
    # pivot low separates the last two highs. The up swing keeps the higher one.
    n = 60
    close = np.interp(np.arange(n), [0, 10, 20, 25, 30, 40, 50],
                      [100, 110, 100, 112, 111, 115, 105])
    low = close - 0.5
    low[21:46] = 99.6
    df = pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": low, "close": close,
         "volume": np.full(n, 1000.0)},
        index=pd.bdate_range("2024-01-01", periods=n),
    )
    out = compute_support_resistance_columns(df, vp_window=20)
    k = PIVOT_RIGHT_BARS
    assert np.flatnonzero(~np.isnan(out["PIVOT_LOW"])).tolist() == [20 + k, 50 + k]
    assert out["FIB_SWING_HIGH"][25 + k] == pytest.approx(112.5)
    t = 40 + k
    assert out["FIB_SWING_DIR"][t] == 1
    assert out["FIB_SWING_HIGH"][t] == pytest.approx(115.5)
    assert out["FIB_SWING_LOW"][t] == pytest.approx(99.5)


# ──────────────────────────────────────────────────────────
# Volume profile
# ──────────────────────────────────────────────────────────

def test_volume_profile_known_concentration():
    n = 40
    high = np.full(n, 101.0)
    low = np.full(n, 100.0)
    volume = np.full(n, 10.0)
    # A few wide, thin bars stretch the range to [100, 110].
    high[::10] = 110.0
    volume[::10] = 1.0
    close = np.full(n, 100.5)
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.bdate_range("2024-01-01", periods=n),
    )
    out = compute_support_resistance_columns(df, vp_window=20, vp_bins=10)
    t = n - 1
    assert out["VP_POC"][t] == pytest.approx(100.5)
    assert out["VP_VAL"][t] == pytest.approx(100.0)
    assert out["VP_VAH"][t] == pytest.approx(101.0)
    assert out["VP_POS"][t] == pytest.approx(0.0)
    assert np.isnan(out["VP_POC"][18]) and not np.isnan(out["VP_POC"][19])


def test_volume_profile_nan_window_is_nan():
    df = _random_walk(200)
    df.iloc[150, df.columns.get_loc("volume")] = np.nan
    out = compute_support_resistance_columns(df, vp_window=50)
    assert np.isnan(out["VP_POC"][150:200]).all()
    assert not np.isnan(out["VP_POC"][149])
