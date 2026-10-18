"""
Tests for investbot.evaluation.variants (task 180: pre-registered polarity variants)
and their wiring into the panel and the report. Synthetic data only.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("talib")

from investbot.evaluation import variants as vr  # noqa: E402
from investbot.evaluation.panel import (  # noqa: E402
    VARIANT_TIMING_KEY,
    asset_features,
    signal_columns,
)
from investbot.scoring.technical import confluence as cf  # noqa: E402
from investbot.scoring.technical.regime import REGIME_CHAOS, REGIME_RANGE, REGIME_TREND  # noqa: E402
from investbot.scoring.technical.scorer import compute_daily_frame  # noqa: E402
from tests.test_evaluation import _bars  # noqa: E402

N = 520
NON_OSC_FAMILIES = ("TREND", "PATTERN", "SR")      # no oscillator-rule signal in them


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return _bars(N, seed=11)


@pytest.fixture(scope="module")
def frame(bars) -> pd.DataFrame:
    return compute_daily_frame(bars)


@pytest.fixture(scope="module")
def variants(frame) -> dict[str, np.ndarray]:
    return vr.compute_variants(frame)


def _eq(a, b) -> bool:
    return np.array_equal(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64),
                          equal_nan=True)


def test_keys_and_shapes(frame, variants):
    assert list(variants) == vr.list_variant_columns()
    assert set(variants) == set(vr.VARIANT_COLUMNS)
    assert len(variants) == 6
    for k, v in variants.items():
        assert v.shape == (len(frame),) and v.dtype == np.float64, k
        assert np.nanmax(np.abs(v)) <= 1.0, k
        assert np.isfinite(v).sum() > 100, k


def test_warmup_nan_pattern_matches_production(frame, variants):
    for k, prod in vr.VARIANT_COLUMNS.items():
        assert _eq(np.isnan(variants[k]), np.isnan(frame[prod].to_numpy())), k


def test_osc_extreme_variants_are_aggregated_alternatives(frame, variants):
    for name, extreme in vr.OSC_EXTREME_VARIANTS.items():
        agg = cf.aggregate_signals(frame, cf.compute_signals(frame, osc_extreme=extreme))
        assert _eq(variants[f"VAR_{name}"], agg["TECH_SCORE"])
        assert _eq(variants[f"VAR_{name}_MOMENTUM"], agg["TECH_SCORE_MOMENTUM"])
        # they differ from production somewhere (oscillators do reach their extremes)
        assert not _eq(variants[f"VAR_{name}"], frame["TECH_SCORE"].to_numpy())
    assert vr.OSC_EXTREME_VARIANTS == {"OSC_REVERT": -1.0, "OSC_MOMENTUM": 1.0}


def test_passing_production_signals_changes_nothing(frame, variants):
    again = vr.compute_variants(frame, signals=cf.compute_signals(frame))
    for k in variants:
        assert _eq(again[k], variants[k]), k


# ──────────────────────────────────────────────────────────
# Regime-dependent polarity
# ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mixed_regime(frame) -> pd.DataFrame:
    """The real frame with REGIME forced to a known range/trend/chaotic/NaN pattern."""
    pattern = np.array([REGIME_RANGE, REGIME_TREND, REGIME_CHAOS, np.nan], dtype=np.float64)
    out = frame.copy()
    out["REGIME"] = np.resize(pattern, len(frame))
    return out


def test_regime_polarity_signals_flip_only_oscillators_on_range(mixed_regime):
    base = cf.compute_signals(mixed_regime)
    flipped = vr.regime_polarity_signals(mixed_regime, base)
    assert list(flipped) == list(base)
    rng = mixed_regime["REGIME"].to_numpy() == REGIME_RANGE
    for k in base:
        if k in cf.OSCILLATOR_SIGNALS:
            assert _eq(flipped[k][rng], -base[k][rng]), k
            assert _eq(flipped[k][~rng], base[k][~rng]), k
        else:
            assert flipped[k] is base[k] or _eq(flipped[k], base[k]), k
    # input not mutated
    assert _eq(base["RSI"], cf.compute_signals(mixed_regime)["RSI"])


def test_regime_polarity_equals_production_off_range(mixed_regime):
    prod = cf.aggregate_signals(mixed_regime, cf.compute_signals(mixed_regime))
    var = vr.compute_variants(mixed_regime)
    regime = mixed_regime["REGIME"].to_numpy()
    off = regime != REGIME_RANGE                       # trend, chaotic and NaN
    assert np.isnan(regime[off]).any() and (regime[off] == REGIME_TREND).any()
    for suffix, col in (("", "TECH_SCORE"), ("_MOMENTUM", "TECH_SCORE_MOMENTUM")):
        v = var[f"VAR_REGIME_POLARITY{suffix}"]
        assert _eq(v[off], prod[col][off]), col
        on = ~off & np.isfinite(v)
        assert (v[on] != prod[col][on]).any(), col     # the flip does something on range bars


def test_regime_polarity_differs_only_through_oscillators(mixed_regime):
    """On range bars the families without oscillator signals are unchanged, and
    zeroing the oscillator signals makes the variant and production identical."""
    base = cf.compute_signals(mixed_regime)
    prod = cf.aggregate_signals(mixed_regime, base)
    var = cf.aggregate_signals(mixed_regime, vr.regime_polarity_signals(mixed_regime, base))
    for fam in NON_OSC_FAMILIES:
        assert _eq(var[f"TECH_SCORE_{fam}"], prod[f"TECH_SCORE_{fam}"]), fam
    zeroed = {k: (np.where(np.isnan(v), np.nan, 0.0) if k in cf.OSCILLATOR_SIGNALS else v)
              for k, v in base.items()}
    a = cf.aggregate_signals(mixed_regime, zeroed)
    b = cf.aggregate_signals(mixed_regime, vr.regime_polarity_signals(mixed_regime, zeroed))
    for k in a:
        assert _eq(a[k], b[k]), k                    # -0.0 == 0.0


def test_regime_polarity_without_regime_column_is_production(frame):
    no_regime = frame.drop(columns=["REGIME"])
    base = cf.compute_signals(no_regime)
    flipped = vr.regime_polarity_signals(no_regime, base)
    for k in base:
        assert _eq(flipped[k], base[k]), k


def test_real_frame_has_range_bars(frame, variants):
    rng = frame["REGIME"].to_numpy() == REGIME_RANGE
    off = ~rng
    assert rng.any()
    assert _eq(variants["VAR_REGIME_POLARITY"][off], frame["TECH_SCORE"].to_numpy()[off])


# ──────────────────────────────────────────────────────────
# No lookahead
# ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("cut", [300, 410, N - 2])
def test_truncation_invariance_every_variant(bars, variants, cut):
    """The value at bar t is unchanged when the bars after t are removed."""
    short = vr.compute_variants(compute_daily_frame(bars.iloc[: cut + 1]))
    assert list(short) == list(variants)
    for k in variants:
        np.testing.assert_allclose(short[k], variants[k][: cut + 1], rtol=1e-9, atol=1e-12,
                                   equal_nan=True, err_msg=k)


def test_no_dropna_in_variants():
    assert "dropna(" not in inspect.getsource(vr)


# ──────────────────────────────────────────────────────────
# Panel wiring
# ──────────────────────────────────────────────────────────

def test_asset_features_adds_variants_only_when_asked(bars, variants):
    plain = asset_features(bars)
    assert not any(c.startswith(vr.VARIANT_PREFIX) for c in plain.columns)
    timings: dict[str, float] = {}
    f = asset_features(bars, variants=True, timings=timings)
    assert timings[VARIANT_TIMING_KEY] > 0
    for k in vr.list_variant_columns():
        assert _eq(f[k].to_numpy(), variants[k]), k
    pd.testing.assert_frame_equal(f[plain.columns], plain)
    cols = signal_columns(f)
    assert set(vr.list_variant_columns()) <= set(cols)
    assert len(cols) == len(signal_columns(plain)) + 6


# ──────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────

def _head(rows: list[tuple]) -> pd.DataFrame:
    cols = ["signal", "horizon", "ic_mean", "icir", "ic_t_nw", "ic_q", "block_sign_share",
            "promising", "held_cagr_net", "held_info_ratio_net"]
    df = pd.DataFrame(rows, columns=cols)
    return df.assign(label="open", subset="oos", n_days=100)


def test_variant_verdicts_and_section():
    from investbot.evaluation.report import RunInfo, variant_verdicts, variants_section

    rows = []
    for h in (1, 5):
        rows += [("TECH_SCORE", h, 0.01, 0.05, 1.0, 0.5, 0.6, False, 0.10, 0.1),
                 ("TECH_SCORE_MOMENTUM", h, 0.0, 0.0, 0.0, 0.9, 0.5, False, 0.08, 0.0),
                 ("BASE_MOM_20", h, 0.02, 0.10, 2.0, 0.2, 0.7, False, 0.12, 0.2)]
        for v in vr.list_variant_columns():
            rows.append((v, h, -0.01, -0.05, -1.0, 0.6, 0.5, False, 0.05, -0.1))
    head = _head(rows)
    # VAR_OSC_REVERT is promising at h = 5 with ICIR above production and the baseline
    i = head.index[(head["signal"] == "VAR_OSC_REVERT") & (head["horizon"] == 5)][0]
    head.loc[i, ["ic_mean", "icir", "ic_q", "block_sign_share", "promising"]] = \
        [0.03, 0.20, 0.01, 0.9, True]
    # VAR_OSC_MOMENTUM promising at h = 1 but its ICIR is below the baseline: no win
    j = head.index[(head["signal"] == "VAR_OSC_MOMENTUM") & (head["horizon"] == 1)][0]
    head.loc[j, ["ic_mean", "icir", "ic_q", "block_sign_share", "promising"]] = \
        [0.02, 0.08, 0.05, 0.8, True]

    lines = variant_verdicts(head, (1, 5))
    assert len(lines) == 6
    by = {ln.split("**")[1]: ln for ln in lines}
    assert "best baseline at h = 5 " in by["VAR_OSC_REVERT"]
    assert "at no horizon" in by["VAR_OSC_MOMENTUM"]
    assert "vs TECH_SCORE_MOMENTUM" in by["VAR_REGIME_POLARITY_MOMENTUM"]
    assert "ΔICIR +0.150" in by["VAR_OSC_REVERT"]

    info = RunInfo(n_assets=3, first_date=None, last_date=None, horizons=(1, 5), start=None,
                   universe="index")
    text = variants_section(head, (1, 5), info)
    assert "## Polarity variants (pre-registered)" in text
    assert "### h = 1" in text and "### h = 5" in text
    assert "Ankit's decision" in text
    assert "Best baseline at h = 5: BASE_MOM_20" in text
    for v in vr.list_variant_columns():
        assert f"| {v} |" in text
