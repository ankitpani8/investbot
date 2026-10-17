"""
Contract and sanity tests for investbot.scoring.technical.regime.

Synthetic series only: no DB and no network.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("talib")

from investbot.scoring.technical import regime as R  # noqa: E402
from investbot.scoring.technical.regime import (  # noqa: E402
    add_regime,
    compute_regime,
    list_regime_columns,
    regime_scores,
    rolling_hurst,
    rolling_lyapunov,
)

N = 600


def _frame_from_returns(r: np.ndarray, spread: float = 0.004) -> pd.DataFrame:
    """OHLC whose close-to-close log returns are exactly ``r`` (r[0] is ignored)."""
    close = 100.0 * np.exp(np.cumsum(r))
    prev = np.r_[close[0], close[:-1]]
    high = np.maximum(close, prev) * (1 + spread)
    low = np.minimum(close, prev) * (1 - spread)
    idx = pd.bdate_range("2021-01-01", periods=len(r))
    return pd.DataFrame({"open": prev, "high": high, "low": low, "close": close,
                         "volume": 1e6}, index=idx)


def _ar1(phi: float, n: int, seed: int, sigma: float = 0.01, drift: float = 0.0) -> np.ndarray:
    e = np.random.default_rng(seed).normal(0.0, sigma, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x + drift


def _logistic(n: int, r: float = 4.0, x0: float = 0.3) -> np.ndarray:
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    return _frame_from_returns(np.random.default_rng(11).normal(0.0003, 0.015, N))


@pytest.fixture(scope="module")
def full(ohlcv) -> pd.DataFrame:
    return add_regime(ohlcv)


# ──────────────────────────────────────────────────────────
# Contract
# ──────────────────────────────────────────────────────────

def test_columns_and_shape(ohlcv, full):
    out = compute_regime(ohlcv)
    assert list(out) == list_regime_columns()
    assert list(full.columns) == list(ohlcv.columns) + list_regime_columns()
    assert len(full) == len(ohlcv) and full.index.equals(ohlcv.index)
    for name, arr in out.items():
        assert isinstance(arr, np.ndarray) and arr.dtype == np.float64, name
        assert arr.shape == (len(ohlcv),), name


def test_custom_windows_name_columns(ohlcv):
    out = compute_regime(ohlcv, hurst_windows=(60,), lyap_windows=(60, 120), regime_window=60)
    assert "HURST_60" in out and "LYAP_60" in out and "LYAP_120" in out
    assert "HURST_100" not in out


def test_input_not_mutated(ohlcv):
    before = ohlcv.copy()
    add_regime(ohlcv)
    pd.testing.assert_frame_equal(ohlcv, before)


def test_validation(ohlcv):
    with pytest.raises(ValueError, match="missing"):
        compute_regime(ohlcv.drop(columns="high"))
    with pytest.raises(ValueError, match="regime_window"):
        compute_regime(ohlcv, regime_window=50)
    with pytest.raises(ValueError, match="too short"):
        compute_regime(ohlcv, hurst_windows=(12, 100), lyap_windows=(100,), regime_window=100)
    with pytest.raises(ValueError, match="too short"):
        compute_regime(ohlcv, hurst_windows=(100,), lyap_windows=(15, 100), regime_window=100)


def test_warmup_is_nan_then_filled(full):
    for w in R.HURST_WINDOWS:
        col = full[f"HURST_{w}"]
        assert col.iloc[:w].isna().all() and col.iloc[w:].notna().all(), w
    for w in R.LYAP_WINDOWS:
        col = full[f"LYAP_{w}"]
        assert col.iloc[:w].isna().all() and col.iloc[w:].notna().all(), w
    assert full["VOL_RATIO"].iloc[: R.VOL_LONG].isna().all()
    assert full["VOL_RATIO"].iloc[R.VOL_LONG:].notna().all()
    first = R.VOL_LONG  # the slowest regime input
    assert full["REGIME"].iloc[:first].isna().all()
    assert full["REGIME"].iloc[first:].notna().all()


def test_regime_value_ranges(full):
    valid = full.iloc[R.VOL_LONG:]
    assert set(valid["REGIME"].unique()) <= {0.0, 1.0, 2.0}
    for col in ("REGIME_SCORE_RANGE", "REGIME_SCORE_TREND", "REGIME_SCORE_CHAOS",
                "REGIME_CONFIDENCE"):
        assert valid[col].between(0, 1).all(), col
    np.testing.assert_allclose(valid["REGIME_SCORE_RANGE"] + valid["REGIME_SCORE_TREND"], 1.0)
    scores = valid[["REGIME_SCORE_RANGE", "REGIME_SCORE_TREND", "REGIME_SCORE_CHAOS"]].to_numpy()
    np.testing.assert_array_equal(valid["REGIME"], scores.argmax(axis=1))
    top2 = np.sort(scores, axis=1)[:, -2:]
    np.testing.assert_allclose(valid["REGIME_CONFIDENCE"], top2[:, 1] - top2[:, 0])


# ──────────────────────────────────────────────────────────
# No lookahead
# ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("cut", [120, 251, 400, 599])
def test_no_lookahead_truncation_invariance(ohlcv, full, cut):
    """Values on bars < cut must not change when the bars after them are removed."""
    trunc = add_regime(ohlcv.iloc[:cut])[list_regime_columns()]
    ref = full[list_regime_columns()].iloc[:cut]
    pd.testing.assert_frame_equal(trunc, ref, check_exact=False, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("start", [1, 37, 140])
def test_start_invariance_of_fixed_windows(ohlcv, full, start):
    """Fixed-window metrics only depend on their own window, not on older history."""
    late = add_regime(ohlcv.iloc[start:])
    cols = [c for c in list_regime_columns()
            if c.startswith(("HURST_", "LYAP_", "REALVOL_", "VOL_RATIO"))]
    pd.testing.assert_frame_equal(
        late[cols].iloc[R.VOL_LONG:], full[cols].iloc[start + R.VOL_LONG:],
        check_exact=False, rtol=1e-9, atol=1e-12,
    )


def test_nan_close_only_poisons_windows_that_contain_it(ohlcv, full):
    bad = ohlcv.copy()
    pos = 320
    bad.iloc[pos, bad.columns.get_loc("close")] = np.nan
    out = add_regime(bad)
    cols = list_regime_columns()
    pd.testing.assert_frame_equal(out[cols].iloc[:pos], full[cols].iloc[:pos])
    assert out[cols].iloc[pos].isna().all()
    # HURST_100 uses closes t-100..t; LYAP_100 uses returns t-99..t, and the
    # NaN close kills returns pos and pos+1.
    assert out["HURST_100"].iloc[pos: pos + 101].isna().all()
    assert out["HURST_100"].iloc[pos + 101:].notna().all()
    assert out["LYAP_100"].iloc[pos: pos + 101].isna().all()
    assert out["LYAP_100"].iloc[pos + 101:].notna().all()


# ──────────────────────────────────────────────────────────
# Sanity fixtures
# ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_walk_hurst_near_half(seed):
    df = _frame_from_returns(np.random.default_rng(seed).normal(0, 0.01, 1250))
    log_c = np.log(df["close"].to_numpy())
    assert abs(np.nanmedian(rolling_hurst(log_c, 100)) - 0.5) < 0.08
    assert abs(np.nanmedian(rolling_hurst(log_c, 250)) - 0.5) < 0.05


def test_trending_series_hurst_above_0_6():
    df = _frame_from_returns(_ar1(0.5, 1250, seed=4, drift=0.002))
    out = compute_regime(df)
    assert np.nanmedian(out["HURST_100"]) > 0.6
    assert np.nanmedian(out["HURST_250"]) > 0.6


def test_mean_reverting_series_hurst_below_0_45():
    df = _frame_from_returns(_ar1(-0.5, 1250, seed=5))
    out = compute_regime(df)
    assert np.nanmedian(out["HURST_100"]) < 0.45
    assert np.nanmedian(out["HURST_250"]) < 0.45


def test_logistic_map_positive_lyapunov():
    # Fed in as log returns. The true exponent at r = 4 is ln 2 ≈ 0.693 per step.
    x = _logistic(800)
    for w in (100, 250):
        lyap = rolling_lyapunov(np.r_[np.nan, x[1:]], w)
        med = np.nanmedian(lyap)
        assert med > 0.4, (w, med)
        assert abs(med - np.log(2)) < 0.2, (w, med)


def test_sine_wave_lyapunov_near_zero():
    r = 0.01 * np.sin(0.3 * np.arange(800))
    for w in (100, 250):
        lyap = rolling_lyapunov(np.r_[np.nan, r[1:]], w)
        assert np.nanmax(np.abs(lyap)) < 0.05, w


def test_trending_series_labelled_trend():
    df = _frame_from_returns(_ar1(0.5, 800, seed=6, drift=0.004), spread=0.002)
    out = add_regime(df).iloc[R.VOL_LONG:]
    assert (out["REGIME"] == R.REGIME_TREND).mean() > 0.6


def test_mean_reverting_series_labelled_range():
    df = _frame_from_returns(_ar1(-0.6, 800, seed=7))
    out = add_regime(df).iloc[R.VOL_LONG:]
    assert (out["REGIME"] == R.REGIME_RANGE).mean() > 0.6


def test_volatility_spike_labelled_chaotic():
    rng = np.random.default_rng(8)
    r = np.r_[rng.normal(0, 0.008, 500), rng.normal(0, 0.035, 40)]
    out = add_regime(_frame_from_returns(r))
    assert (out["REGIME"].iloc[-20:] == R.REGIME_CHAOS).all()
    assert (out["REGIME_SCORE_CHAOS"].iloc[-20:] > 0.8).all()
    assert (out["REGIME_CONFIDENCE"].iloc[-20:] > 0).all()


# ──────────────────────────────────────────────────────────
# Smooth confidence
# ──────────────────────────────────────────────────────────

def test_confidence_is_continuous_across_label_flip():
    """Sweeping one input through the trend/range boundary moves confidence smoothly."""
    hurst = np.linspace(0.3, 0.7, 4001)
    n = len(hurst)
    out = regime_scores(hurst, np.full(n, R.ADX_NEUTRAL), np.full(n, 1.0), np.full(n, 0.26))
    conf, label = out["REGIME_CONFIDENCE"], out["REGIME"]
    assert set(np.unique(label)) == {R.REGIME_RANGE, R.REGIME_TREND}
    # No jumps: step size bounded by the sigmoid slope (1/4 per z unit).
    step_z = (hurst[1] - hurst[0]) * R.HURST_WEIGHT / R.HURST_SCALE
    assert np.abs(np.diff(conf)).max() <= 2 * 0.25 * step_z + 1e-12
    flip = np.flatnonzero(np.diff(label))[0]
    assert conf[flip] < 0.01 and conf[flip + 1] < 0.01
    assert conf[0] > 0.6 and conf[-1] > 0.6


def test_regime_scores_propagate_nan():
    out = regime_scores(np.array([np.nan, 0.5]), np.array([20.0, 20.0]),
                        np.array([1.0, 1.0]), np.array([0.26, 0.26]))
    assert np.isnan(out["REGIME"][0]) and np.isnan(out["REGIME_CONFIDENCE"][0])
    assert np.isfinite(out["REGIME"][1])


# ──────────────────────────────────────────────────────────
# Performance
# ──────────────────────────────────────────────────────────

def test_runtime_budget_per_asset():
    # Budget is ~1 minute for 50 assets of 1250 daily bars, about 1.2 s each.
    # Measured ~0.55 s on the dev laptop; the bound here is loose for slow CI.
    df = _frame_from_returns(np.random.default_rng(9).normal(0, 0.015, 1250))
    t0 = time.perf_counter()
    compute_regime(df)
    assert time.perf_counter() - t0 < 5.0
