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


# ──────────────────────────────────────────────────────────
# NaN close propagation
#
# Compute functions must NOT dropna() or patch NaN (warmup NaNs are a contract),
# so a NaN close flows straight through. These tests document how far it
# spreads. That is why incomplete OHLC rows are rejected at ingestion
# (investbot/data/quality.py) and never reach this layer.
# ──────────────────────────────────────────────────────────

# Indicators whose TA-Lib state is carried bar to bar: one NaN input poisons
# every later bar, not just the next `window` bars.
STATEFUL_COLUMNS = ["SMA_20", "EMA_20", "MACD_12_26_9", "BBM_20_2.0"]


def _with_nan_close(ohlcv: pd.DataFrame, pos: int) -> pd.DataFrame:
    bad = ohlcv.copy()
    bad.iloc[pos, bad.columns.get_loc("close")] = np.nan
    return bad


def test_mid_series_nan_close_propagation(ohlcv, full):
    pos = 400
    out = add_all_indicators(_with_nan_close(ohlcv, pos))

    # The compute layer neither drops nor repairs rows.
    assert len(out) == len(ohlcv)
    assert np.isnan(out["close"].iloc[pos])

    # Nothing before the bad bar changes (no backward contamination).
    before_out, before_ref = _numeric(out).iloc[:pos], _numeric(full).iloc[:pos]
    np.testing.assert_allclose(
        before_out.to_numpy(), before_ref.to_numpy(), rtol=1e-6, atol=1e-8, equal_nan=True
    )

    # Stateful indicators never recover: NaN from the bad bar to the end.
    for col in STATEFUL_COLUMNS:
        assert full[col].iloc[pos:].notna().all(), f"{col}: clean run should be populated"
        assert out[col].iloc[pos:].isna().all(), f"{col}: expected permanent NaN after bad bar"

    # Worse than NaN: some columns stay finite but wrong. RSI is 0.0 here, which
    # a scorer would read as "maximally oversold".
    rsi_out, rsi_ref = out["RSI_14"].iloc[pos:], full["RSI_14"].iloc[pos:]
    assert rsi_out.notna().all()
    assert (rsi_out - rsi_ref).abs().max() > 1.0


def test_nan_close_on_latest_bar_breaks_latest_coverage(ohlcv, full):
    """The production symptom: a NaN close on the newest bar guts the latest-bar score inputs."""
    last = len(ohlcv) - 1
    out = add_all_indicators(_with_nan_close(ohlcv, last))

    cols = list_indicator_columns(out)
    clean_filled = int(full[cols].iloc[-1].notna().sum())
    bad_filled = int(out[cols].iloc[-1].notna().sum())

    assert clean_filled >= MIN_EXPECTED_INDICATORS
    assert bad_filled < MIN_EXPECTED_INDICATORS
    for col in STATEFUL_COLUMNS:
        assert np.isnan(out[col].iloc[-1])


def test_returns_fill_method_none_unchanged_on_gap_free_data(ohlcv):
    """fill_method=None must not change any return column when close has no gaps."""
    from investbot.scoring.technical._extensions import returns

    out = returns(ohlcv)
    close = ohlcv["close"]
    legacy_pct = close / close.shift(1) - 1.0  # what the old ffill default gave with no NaNs
    np.testing.assert_allclose(out["PCTRET_1"], legacy_pct, rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(
        out["CUMPCTRET_1"], (1.0 + legacy_pct).cumprod() - 1.0, rtol=1e-12, equal_nan=True
    )


def test_returns_nan_close_not_forward_filled(ohlcv):
    """A NaN close yields NaN returns at that bar and the next, not a masked 0."""
    from investbot.scoring.technical._extensions import returns

    df = _with_nan_close(ohlcv, 300)
    pct = pd.Series(returns(df)["PCTRET_1"])
    assert np.isnan(pct.iloc[300]) and np.isnan(pct.iloc[301])
    assert np.isfinite(pct.iloc[302])


def test_cumulative_returns_restart_after_nan_close(ohlcv):
    """CUM* are NaN where the return is NaN and restart from the next valid bar."""
    from investbot.scoring.technical._extensions import returns

    k = 300
    out = returns(_with_nan_close(ohlcv, k))
    clean = returns(ohlcv)
    for name, ret in (("CUMLOGRET_1", "LOGRET_1"), ("CUMPCTRET_1", "PCTRET_1")):
        cum, r = pd.Series(out[name]), pd.Series(out[ret])
        assert np.isnan(cum.iloc[k]) and np.isnan(cum.iloc[k + 1])
        assert cum.isna().equals(r.isna())
        # before the gap: identical to the gap-free run
        np.testing.assert_allclose(cum.iloc[:k], pd.Series(clean[name]).iloc[:k], equal_nan=True)
        # first bar after the gap holds its own return, then accumulates from there
        assert cum.iloc[k + 2] == pytest.approx(r.iloc[k + 2])
        seg = r.iloc[k + 2:]
        want = seg.cumsum() if name == "CUMLOGRET_1" else (1.0 + seg).cumprod() - 1.0
        np.testing.assert_allclose(cum.iloc[k + 2:], want, rtol=1e-12)


def test_cumulative_returns_unchanged_on_gap_free_data(ohlcv):
    from investbot.scoring.technical._extensions import returns

    out = returns(ohlcv)
    close = ohlcv["close"]
    np.testing.assert_allclose(
        out["CUMLOGRET_1"], np.log(close / close.shift(1)).cumsum(), rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(
        out["CUMPCTRET_1"], (close / close.shift(1)).cumprod() - 1.0, rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize("cut", [100, 350])
def test_returns_truncation_invariance(ohlcv, cut):
    from investbot.scoring.technical._extensions import returns

    full_out = pd.DataFrame(returns(ohlcv))
    trunc_out = pd.DataFrame(returns(ohlcv.iloc[:cut]))
    pd.testing.assert_frame_equal(trunc_out, full_out.iloc[:cut])


@pytest.mark.parametrize("cut", [250, 301, 302, 350])
def test_cumulative_returns_truncation_invariance_with_gap(ohlcv, cut):
    from investbot.scoring.technical._extensions import returns

    gappy = _with_nan_close(ohlcv, 300)
    full_out = pd.DataFrame(returns(gappy))
    trunc_out = pd.DataFrame(returns(gappy.iloc[:cut]))
    pd.testing.assert_frame_equal(trunc_out, full_out.iloc[:cut])
