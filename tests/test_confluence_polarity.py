"""
Task 180: the confluence refactor (``osc_extreme``, ``aggregate_signals``,
``OSCILLATOR_SIGNALS``) leaves production scoring bit-identical.

``_legacy_confluence`` is a frozen copy of the pre-refactor ``_evaluate`` +
``compute_confluence`` score assembly (commit 011e3c4), so "before and after"
is compared with ``np.array_equal`` (NaN-aware), not a tolerance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("talib")

from investbot.config.scoring_weights import FAMILIES  # noqa: E402
from investbot.scoring.technical import confluence as cf  # noqa: E402
from tests.test_confluence import features, flat_noise, monotonic  # noqa: E402

SCORE_KEYS = [k for k in cf.list_confluence_columns() if k != "TECH_REASONS"]


@pytest.fixture(scope="module", params=["up", "down", "flat"])
def frame(request) -> pd.DataFrame:
    return {"up": lambda: features(monotonic(0.004)),
            "down": lambda: features(monotonic(-0.004)),
            "flat": lambda: features(flat_noise())}[request.param]()


def _legacy_confluence(df: pd.DataFrame):
    """Pre-refactor aggregation, verbatim (signals from the unchanged default path)."""
    n = len(df)
    signals = cf.compute_signals(df)
    names = list(signals)
    family_of = {s.name: s.family for s in cf.SIGNALS}
    fam_idx = np.array([FAMILIES.index(family_of[s]) for s in names], dtype=np.int64)
    stack = np.column_stack([signals[s] for s in names]) if names else np.empty((n, 0))
    in_frame = np.array([(fam_idx == j).any() for j in range(len(FAMILIES))])
    fam_scores = np.full((n, len(FAMILIES)), np.nan)
    counts = np.zeros(len(FAMILIES))
    for j in np.flatnonzero(in_frame):
        cols = fam_idx == j
        counts[j] = cols.sum()
        fam_scores[:, j] = stack[:, cols].mean(axis=1)
    complete = ~np.isnan(fam_scores[:, in_frame]).any(axis=1) if in_frame.any() else np.zeros(n, bool)
    wm = cf.family_weights(df)
    w = np.where(in_frame, wm[:, :-1], 0.0)
    w = w / w.sum(axis=1, keepdims=True) if in_frame.any() else w
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(in_frame, wm[:, -1:] * w / np.maximum(counts, 1), 0.0)
    contrib = np.where(complete[:, None], np.nan_to_num(stack) * share[:, fam_idx], 0.0)
    score = np.where(complete, np.clip(contrib.sum(axis=1), -1.0, 1.0), np.nan)
    out = {"TECH_SCORE": score}
    for j, fam in enumerate(FAMILIES):
        out[f"TECH_SCORE_{fam.upper()}"] = fam_scores[:, j]
    out["TECH_REASONS"] = cf._reasons(names, contrib, complete)
    return out, contrib


def test_compute_confluence_bit_identical_to_pre_refactor(frame):
    got = cf.compute_confluence(frame)
    want, contrib = _legacy_confluence(frame)
    assert list(got) == list(want)
    for k in SCORE_KEYS:
        assert np.array_equal(got[k], want[k], equal_nan=True), k
    assert list(got["TECH_REASONS"]) == list(want["TECH_REASONS"])
    assert np.array_equal(cf.signal_contributions(frame).to_numpy(), contrib, equal_nan=True)


def test_aggregate_signals_equals_compute_confluence(frame):
    agg = cf.aggregate_signals(frame, cf.compute_signals(frame))
    full = cf.compute_confluence(frame)
    assert list(agg) == SCORE_KEYS
    for k in SCORE_KEYS:
        assert np.array_equal(agg[k], full[k], equal_nan=True), k


def test_osc_extreme_none_is_production(frame):
    a = cf.compute_signals(frame)
    b = cf.compute_signals(frame, osc_extreme=cf.OSC_EXTREME_VALUE)
    for k in a:
        assert np.array_equal(a[k], b[k], equal_nan=True), k


def test_oscillator_signals_list():
    assert cf.OSCILLATOR_SIGNALS == frozenset(
        {"RSI", "STOCH", "WILLR", "CCI", "ULTOSC", "BB_POS", "KC_POS", "MFI"})
    assert cf.OSCILLATOR_SIGNALS <= {s.name for s in cf.SIGNALS}


@pytest.mark.parametrize("extreme, want", [
    # RSI 15 is halfway between (0, -e) and (30, -1); RSI 85 halfway between (70, 1) and (100, e)
    (-1.0, [0.0, 0.0, -1.0]),
    (0.0, [-0.5, 0.5, 0.0]),
    (1.0, [-1.0, 1.0, 1.0]),
])
def test_osc_extreme_hand_computed_rsi(extreme, want):
    df = pd.DataFrame({"RSI_14": [15.0, 85.0, 100.0, np.nan]})
    got = cf.compute_signals(df, osc_extreme=extreme)
    assert list(got) == ["RSI"]
    np.testing.assert_allclose(got["RSI"], [*want, np.nan])


@pytest.mark.parametrize("extreme", [-1.0, 0.0, 1.0])
def test_osc_extreme_reaches_every_oscillator_signal(extreme):
    """Every oscillator-rule input at its ceiling reads ``extreme``; nothing else moves."""
    df = pd.DataFrame({
        "RSI_14": [100.0], "STOCHk_14_3_3": [100.0], "WILLR_14": [0.0], "CCI_20": [250.0],
        "ULTOSC_7_14_28": [100.0], "BBP_20_2.0": [1.5], "MFI_14": [100.0],
        "close": [12.0], "KCM_20_2.0": [10.0], "KCU_20_2.0": [11.0],      # rel = 2
        "AROONOSC_14": [40.0],                                            # not an oscillator
    })
    got = cf.compute_signals(df, osc_extreme=extreme)
    assert set(got) == cf.OSCILLATOR_SIGNALS | {"AROON"}
    for name in cf.OSCILLATOR_SIGNALS:
        np.testing.assert_allclose(got[name], [extreme], err_msg=name)
    np.testing.assert_allclose(got["AROON"], [0.4])


def test_osc_extreme_changes_only_oscillator_signals(frame):
    base = cf.compute_signals(frame)
    for extreme in (-1.0, 1.0):
        alt = cf.compute_signals(frame, osc_extreme=extreme)
        assert list(alt) == list(base)
        for k in base:
            if k not in cf.OSCILLATOR_SIGNALS:
                assert np.array_equal(alt[k], base[k], equal_nan=True), k
        # NaN pattern (warmup) is unchanged for the oscillators too
        for k in cf.OSCILLATOR_SIGNALS & set(base):
            assert np.array_equal(np.isnan(alt[k]), np.isnan(base[k])), k
