"""
Tests for investbot.scoring.technical.confluence.

Synthetic OHLCV only: no DB, no network. The upstream modules run for real,
so the column-contract test fails as soon as one of them adds a column that
confluence.COLUMN_FAMILIES does not place.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("talib")

from investbot.config import scoring_weights as sw  # noqa: E402
from investbot.scoring.technical import confluence as cf  # noqa: E402
from investbot.scoring.technical.candle_patterns import (  # noqa: E402
    compute_candle_patterns,
    list_candle_columns,
)
from investbot.scoring.technical.indicators import (  # noqa: E402
    add_all_indicators,
    list_indicator_columns,
)
from investbot.scoring.technical.multi_timeframe import list_mtf_columns  # noqa: E402
from investbot.scoring.technical.regime import (  # noqa: E402
    REGIME_CHAOS,
    REGIME_RANGE,
    REGIME_TREND,
    compute_regime,
    list_regime_columns,
)
from investbot.scoring.technical.support_resistance import (  # noqa: E402
    SR_COLUMNS,
    compute_support_resistance_columns,
)

N = 500
WARMUP = 260          # past EMA_200 and the regime warmup


def _ohlcv(close: np.ndarray, seed: int, vol: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1000)
    n = len(close)
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, vol / 4, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.bdate_range("2022-01-03", periods=n),
    )


def monotonic(drift: float, n: int = N, seed: int = 1) -> pd.DataFrame:
    """Every close moves the same way as `drift`, by a random 10%..190% of it."""
    rng = np.random.default_rng(seed)
    steps = drift * (1 + rng.uniform(-0.9, 0.9, n))
    return _ohlcv(100 * np.exp(np.cumsum(steps)), seed)


def flat_noise(n: int = N, seed: int = 2) -> pd.DataFrame:
    """iid noise around a constant level: no trend at all."""
    rng = np.random.default_rng(seed)
    return _ohlcv(100 * np.exp(rng.normal(0, 0.01, n)), seed)


def features(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV + indicators + candles + S/R + regime, the daily frame scorer.py will build."""
    base = add_all_indicators(df)
    extra = {
        **compute_candle_patterns(df),
        **compute_support_resistance_columns(df),
        **compute_regime(df),
    }
    return pd.concat([base, pd.DataFrame(extra, index=df.index)], axis=1)


@pytest.fixture(scope="module")
def up() -> pd.DataFrame:
    return features(monotonic(0.004))


@pytest.fixture(scope="module")
def down() -> pd.DataFrame:
    return features(monotonic(-0.004))


@pytest.fixture(scope="module")
def flat() -> pd.DataFrame:
    return features(flat_noise())


@pytest.fixture(scope="module")
def flat_ohlcv() -> pd.DataFrame:
    return flat_noise()


def _score(frame: pd.DataFrame) -> np.ndarray:
    return cf.compute_confluence(frame)["TECH_SCORE"]


# ──────────────────────────────────────────────────────────
# Acceptance: direction
# ──────────────────────────────────────────────────────────

def test_uptrend_clearly_positive(up):
    s = _score(up)[WARMUP:]
    assert np.nanmin(s) > 0.3
    assert np.nanmean(s) > 0.5


def test_downtrend_clearly_negative(down):
    s = _score(down)[WARMUP:]
    assert np.nanmax(s) < -0.3
    assert np.nanmean(s) < -0.5


def test_flat_noise_near_zero(flat):
    s = _score(flat)[WARMUP:]
    assert abs(np.nanmean(s)) < 0.05
    assert np.nanmedian(np.abs(s)) < 0.3


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_flat_noise_near_zero_other_seeds(seed):
    s = _score(features(flat_noise(seed=seed)))[WARMUP:]
    assert abs(np.nanmean(s)) < 0.08


def test_family_scores_follow_the_trend(up, down):
    res_up, res_down = cf.compute_confluence(up), cf.compute_confluence(down)
    for fam in ("TREND", "MOMENTUM", "VOLUME", "SR"):
        assert np.nanmean(res_up[f"TECH_SCORE_{fam}"][WARMUP:]) > 0.2, fam
        assert np.nanmean(res_down[f"TECH_SCORE_{fam}"][WARMUP:]) < -0.2, fam


# ──────────────────────────────────────────────────────────
# Bounds, shape, NaN handling
# ──────────────────────────────────────────────────────────

def test_outputs_and_bounds(up, down, flat):
    for frame in (up, down, flat):
        res = cf.compute_confluence(frame)
        assert list(res) == cf.list_confluence_columns()
        for key, arr in res.items():
            assert len(arr) == len(frame)
            if key in ("TECH_REASONS", "TECH_SCORE_MTF"):    # MTF: no 15m data here
                continue
            vals = arr[~np.isnan(arr)]
            assert vals.size, key
            assert np.all((vals >= -1.0) & (vals <= 1.0)), key


def test_signals_bounded(up, down, flat):
    for frame in (up, down, flat):
        for name, arr in cf.compute_signals(frame).items():
            vals = arr[~np.isnan(arr)]
            assert np.all((vals >= -1.0) & (vals <= 1.0)), name


def test_bounded_under_extreme_inputs(flat):
    """Wild indicator values (and a chaotic regime) still give a score in [-1, 1]."""
    rng = np.random.default_rng(0)
    wild = flat.copy()
    for col in wild.columns:
        if col in ("REGIME", "REGIME_CONFIDENCE"):
            continue
        wild[col] = wild[col] * rng.choice([-1e6, -1.0, 1.0, 1e6], len(wild))
    wild["REGIME"] = float(REGIME_CHAOS)
    wild["REGIME_CONFIDENCE"] = 1.0
    s = _score(wild)
    vals = s[~np.isnan(s)]
    assert vals.size and np.all(np.abs(vals) <= 1.0)


def test_warmup_is_nan_not_dropped(flat_ohlcv, flat):
    res = cf.compute_confluence(flat)
    s = res["TECH_SCORE"]
    assert len(s) == len(flat_ohlcv)
    assert np.isnan(s[:150]).all()          # EMA_200 is still warming up
    assert not np.isnan(s[WARMUP:]).any()
    assert all(r == [] for r in res["TECH_REASONS"][:150])


def test_input_not_mutated(flat):
    before = flat.copy()
    cf.compute_confluence(flat)
    pd.testing.assert_frame_equal(flat, before)


def test_add_confluence_appends_columns(flat):
    out = cf.add_confluence(flat)
    assert list(out.columns[: flat.shape[1]]) == list(flat.columns)
    assert set(cf.list_confluence_columns()) <= set(out.columns)


# ──────────────────────────────────────────────────────────
# No lookahead
# ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("cut", [262, 380, N - 1])
def test_truncation_invariance(flat_ohlcv, flat, cut):
    """The score at bar t must not change when bars after t are removed."""
    full = cf.compute_confluence(flat)
    trunc = cf.compute_confluence(features(flat_ohlcv.iloc[:cut]))
    for key in cf.list_confluence_columns():
        a, b = trunc[key][-1], full[key][cut - 1]
        if key == "TECH_REASONS":
            assert a == b
        elif np.isnan(a) or np.isnan(b):
            assert np.isnan(a) and np.isnan(b), key
        else:
            assert np.isclose(a, b, rtol=1e-9, atol=1e-12), f"{key}: {a} vs {b}"


@pytest.mark.parametrize("cut", [300, 450])
def test_truncation_invariance_every_signal(flat_ohlcv, flat, cut):
    full = cf.compute_signals(flat)
    trunc = cf.compute_signals(features(flat_ohlcv.iloc[:cut]))
    assert set(full) == set(trunc)
    for name in full:
        a, b = trunc[name][-1], full[name][cut - 1]
        assert (np.isnan(a) and np.isnan(b)) or np.isclose(a, b, rtol=1e-9, atol=1e-12), name


# ──────────────────────────────────────────────────────────
# Column → family contract
# ──────────────────────────────────────────────────────────

def _upstream_columns(frame: pd.DataFrame) -> set[str]:
    return (
        set(list_indicator_columns(add_all_indicators(frame[["open", "high", "low", "close", "volume"]])))
        | set(list_candle_columns())
        | set(SR_COLUMNS)
        | set(list_regime_columns())
        | set(list_mtf_columns())
    )


def test_every_upstream_column_is_mapped(flat_ohlcv):
    missing = sorted(_upstream_columns(flat_ohlcv) - set(cf.COLUMN_FAMILIES))
    assert not missing, f"add these to confluence.COLUMN_FAMILIES (family or None): {missing}"


def test_no_stale_mappings(flat_ohlcv):
    stale = sorted(set(cf.COLUMN_FAMILIES) - _upstream_columns(flat_ohlcv))
    assert not stale, f"mapped but no longer produced upstream: {stale}"


def test_mapped_families_are_known():
    assert set(cf.COLUMN_FAMILIES.values()) - {None} <= set(sw.FAMILIES)


# Inputs a signal may read without belonging to its family: raw bars and normalisers.
CONTEXT_INPUTS = {"open", "high", "low", "close", "volume", "ATR_14", "NATR_14"}


def test_signal_inputs_belong_to_their_family():
    for sig in cf.SIGNALS:
        assert sig.family in sw.FAMILIES, sig.name
        for col in sig.inputs:
            if col in CONTEXT_INPUTS:
                continue
            assert cf.COLUMN_FAMILIES.get(col) == sig.family, f"{sig.name}: {col}"


def test_every_scored_column_feeds_a_signal():
    used = {c for sig in cf.SIGNALS for c in sig.inputs}
    scored = {c for c, fam in cf.COLUMN_FAMILIES.items() if fam is not None}
    indecision = {f"CDL_{n[3:]}" for n in cf.INDECISION_PATTERNS}
    assert scored - used == indecision


def test_every_family_has_a_signal_and_unique_names():
    assert {s.family for s in cf.SIGNALS} == set(sw.FAMILIES)
    names = [s.name for s in cf.SIGNALS]
    assert len(names) == len(set(names))


# ──────────────────────────────────────────────────────────
# Weights config
# ──────────────────────────────────────────────────────────

def test_weight_config_is_consistent():
    assert set(sw.FAMILY_WEIGHTS) == {"neutral", *sw.REGIME_NAMES.values()}
    assert set(sw.REGIME_SCORE_MULTIPLIER) == set(sw.FAMILY_WEIGHTS)
    for regime, row in sw.FAMILY_WEIGHTS.items():
        assert set(row) == set(sw.FAMILIES), regime
        assert all(v >= 0 for v in row.values()), regime
        assert sum(row.values()) == pytest.approx(1.0), regime
    assert all(0 < m <= 1 for m in sw.REGIME_SCORE_MULTIPLIER.values())
    assert sw.REGIME_NAMES == {REGIME_RANGE: "range", REGIME_TREND: "trend", REGIME_CHAOS: "chaotic"}


def test_trend_regime_upweights_trend_and_mtf_range_upweights_momentum_and_sr():
    t, r = sw.FAMILY_WEIGHTS["trend"], sw.FAMILY_WEIGHTS["range"]
    assert t["trend"] > r["trend"] and t["mtf"] > r["mtf"]
    assert r["momentum"] > t["momentum"] and r["sr"] > t["sr"]


def _regime_frame(regime: float, conf: float, n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({"REGIME": [regime] * n, "REGIME_CONFIDENCE": [conf] * n})


def _row(name: str) -> np.ndarray:
    return np.array([sw.FAMILY_WEIGHTS[name][f] for f in sw.FAMILIES]
                    + [sw.REGIME_SCORE_MULTIPLIER[name]])


@pytest.mark.parametrize("code,name", [(REGIME_RANGE, "range"), (REGIME_TREND, "trend"),
                                       (REGIME_CHAOS, "chaotic")])
def test_weights_blend_by_confidence(code, name):
    np.testing.assert_allclose(cf.family_weights(_regime_frame(code, 1.0))[0], _row(name))
    np.testing.assert_allclose(cf.family_weights(_regime_frame(code, 0.0))[0], _row("neutral"))
    np.testing.assert_allclose(cf.family_weights(_regime_frame(code, 0.4))[0],
                               0.4 * _row(name) + 0.6 * _row("neutral"))


def test_weights_neutral_without_regime():
    np.testing.assert_allclose(cf.family_weights(_regime_frame(np.nan, np.nan))[0], _row("neutral"))
    np.testing.assert_allclose(cf.family_weights(pd.DataFrame(index=range(2)))[0], _row("neutral"))


def test_regime_changes_the_score(flat):
    """Same signals, different regime → different weights → different score."""
    as_trend = flat.assign(REGIME=float(REGIME_TREND), REGIME_CONFIDENCE=1.0)
    as_range = flat.assign(REGIME=float(REGIME_RANGE), REGIME_CONFIDENCE=1.0)
    as_chaos = flat.assign(REGIME=float(REGIME_CHAOS), REGIME_CONFIDENCE=1.0)
    s_t, s_r, s_c = _score(as_trend), _score(as_range), _score(as_chaos)
    assert not np.allclose(s_t[WARMUP:], s_r[WARMUP:])
    # chaotic multiplier shrinks conviction
    assert np.nanmean(np.abs(s_c[WARMUP:])) < np.nanmean(np.abs(s_t[WARMUP:]))


# ──────────────────────────────────────────────────────────
# Group normalisation, missing families, reasons
# ──────────────────────────────────────────────────────────

def test_family_score_is_mean_of_its_signals(up):
    res, sig = cf.compute_confluence(up), cf.compute_signals(up)
    trend = [sig[s.name] for s in cf.SIGNALS if s.family == "trend"]
    np.testing.assert_allclose(res["TECH_SCORE_TREND"][WARMUP:],
                               np.mean(trend, axis=0)[WARMUP:])


def test_contributions_sum_to_score(up, flat):
    for frame in (up, flat):
        contrib = cf.signal_contributions(frame).sum(axis=1).to_numpy()
        s = _score(frame)
        ok = ~np.isnan(s)
        np.testing.assert_allclose(contrib[ok], s[ok], atol=1e-12)
        assert np.all(contrib[~ok] == 0)


def test_mtf_family_absent_on_daily_frame_and_used_when_present(flat):
    res = cf.compute_confluence(flat)
    assert np.isnan(res["TECH_SCORE_MTF"]).all()
    with_mtf = flat.assign(MTF_TREND_ALIGN=1.0, MTF_MOMENTUM_ALIGN=1.0)
    res2 = cf.compute_confluence(with_mtf)
    assert np.all(res2["TECH_SCORE_MTF"] == 1.0)
    assert np.all(res2["TECH_SCORE"][WARMUP:] > res["TECH_SCORE"][WARMUP:])


def test_reasons_are_top_contributors(up):
    res = cf.compute_confluence(up)
    contrib = cf.signal_contributions(up)
    labels = {s.name: s.label for s in cf.SIGNALS}
    i = len(up) - 1
    reasons = res["TECH_REASONS"][i]
    assert 1 <= len(reasons) <= cf.REASONS_TOP_N
    assert all(isinstance(r, str) and "bullish" in r for r in reasons)
    top = contrib.iloc[i].abs().sort_values(ascending=False, kind="stable").index[0]
    assert reasons[0].startswith(labels[top])


def test_partial_frame_only_uses_available_signals(up):
    only_trend = up[["open", "high", "low", "close", "volume", "EMA_20", "EMA_50", "ATR_14"]]
    res = cf.compute_confluence(only_trend)
    assert set(cf.compute_signals(only_trend)) == {"EMA_20_POS", "EMA_50_POS", "EMA_20_50_CROSS"}
    np.testing.assert_allclose(res["TECH_SCORE"][WARMUP:], res["TECH_SCORE_TREND"][WARMUP:])
    assert np.isnan(res["TECH_SCORE_MOMENTUM"]).all()


def test_no_signals_gives_nan():
    frame = pd.DataFrame({"close": [1.0, 2.0]})
    res = cf.compute_confluence(frame)
    assert np.isnan(res["TECH_SCORE"]).all()
    assert list(res["TECH_REASONS"]) == [[], []]


# ──────────────────────────────────────────────────────────
# Individual rules
# ──────────────────────────────────────────────────────────

def test_oscillator_rule():
    x = np.array([0, 15, 30, 40, 50, 60, 70, 85, 100, 120, np.nan], dtype=float)
    got = cf.oscillator(x, 0, 30, 50, 70, 100)
    want = [0, -0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0, 0, np.nan]
    np.testing.assert_allclose(got, want)
    np.testing.assert_allclose(cf.oscillator(np.array([100.0]), 0, 30, 50, 70, 100, extreme=-1.0), [-1])


def test_sr_position_rule():
    d_sup = np.array([0.0, 1.0, 3.0, 2.0, np.nan, np.nan])
    d_res = np.array([2.0, 1.0, 1.0, np.nan, 2.0, np.nan])
    got = cf.compute_signals(pd.DataFrame({"SR_DIST_SUPPORT_PCT": d_sup,
                                           "SR_DIST_RESIST_PCT": d_res}))["SR_POSITION"]
    np.testing.assert_allclose(got, [1.0, 0.0, -0.5, 1.0, -1.0, np.nan])


def test_supertrend_warmup_zero_is_nan():
    frame = pd.DataFrame({"SUPERTd_10_3.0": [0, 1, -1], "SUPERT_10_3.0": [np.nan, 5.0, 6.0]})
    np.testing.assert_allclose(cf.compute_signals(frame)["SUPERTREND"], [np.nan, 1, -1])


def test_indecision_patterns_do_not_count():
    names = [f"CDL_{n[3:]}" for n in cf.DIRECTIONAL_PATTERNS]
    frame = pd.DataFrame({c: [0.0, 0.0, 0.0] for c in names})
    frame.loc[1, "CDL_ENGULFING"] = 100.0
    frame.loc[2, "CDL_HIKKAKE"] = -200.0
    for n in cf.INDECISION_PATTERNS:          # DOJI & co. firing must not move it
        frame[f"CDL_{n[3:]}"] = 100.0
    got = cf.compute_signals(frame)["CANDLES"]
    from investbot.scoring.technical import candle_patterns as cp
    np.testing.assert_allclose(got, [0.0, cp.PLAIN_WEIGHT / cp.NET_SCORE_SCALE,
                                     -cp.CONFIRMED_WEIGHT / cp.NET_SCORE_SCALE])
    assert "CDL_DOJI" not in next(s for s in cf.SIGNALS if s.name == "CANDLES").inputs


def test_flow_rule_is_volume_fraction():
    n = cf.FLOW_WINDOW + 2
    volume = np.full(n, 10.0)
    obv = np.cumsum(volume)                   # every bar up → fraction 1 → +1
    got = cf.compute_signals(pd.DataFrame({"OBV": obv, "volume": volume}))["OBV_FLOW"]
    assert np.isnan(got[: cf.FLOW_WINDOW]).all()
    np.testing.assert_allclose(got[cf.FLOW_WINDOW:], 1.0)


def test_macd_hist_sign_and_slope():
    frame = pd.DataFrame({"MACDh_12_26_9": [np.nan, 1.0, 2.0, 1.5, -0.5, -0.2]})
    np.testing.assert_allclose(cf.compute_signals(frame)["MACD_HIST"],
                               [np.nan, np.nan, 1.0, 0.0, -1.0, 0.0])
