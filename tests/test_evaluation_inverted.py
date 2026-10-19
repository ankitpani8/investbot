"""
Descriptive ``oos:inverted`` rows (task 320): promising-inverted signals re-run on the negated
signal. Synthetic panel, no DB.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from investbot.evaluation.evaluate import (
    INVERTED_SUBSET,
    NEGATED_DIRECTION,
    PROMISING_Q,
    _add_multiple_testing,
    _wide,
    apply_universe,
    evaluate,
    evaluation_calendar,
    headline,
    promising_inverted,
    signal_matrix,
    universe_frame,
)
from investbot.evaluation.metrics import MIN_NAMES, ROUND_TRIP_COST, signal_metrics
from investbot.evaluation.report import inverted_section
from tests.test_evaluation_universe import synthetic_panel

HORIZONS = (1, 5)
MTCOLS = ["ic_q", "block_sign_share", "promising", "direction"]


def _same(a, b):
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    for x, y in zip(a, b):
        if isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y):
            continue
        assert x == y


@pytest.fixture(scope="module")
def panel():
    f, y = synthetic_panel(n_dates=700, n_names=15, seed=3)
    rng = np.random.default_rng(11)
    r1 = y["FWD_RET_1"].to_numpy()
    f = f.assign(NEG=-r1 + rng.normal(0, 0.002, len(f)))      # predicts negatively
    return f, y


@pytest.fixture(scope="module")
def result(panel):
    f, y = panel
    return evaluate(f, y, HORIZONS, signals=["NEG", "TECH_SCORE", "BASE_MOM_20"])


def test_inverted_signal_is_promising_and_gets_one_row_per_horizon(result):
    table, _ = result
    head = headline(table).set_index(["signal", "horizon"])
    for h in HORIZONS:
        r = head.loc[("NEG", h)]
        assert r["promising"] and r["direction"] == "inverted" and r["ic_q"] < PROMISING_Q
    assert sorted(promising_inverted(table)) == [("NEG", 1), ("NEG", 5)]
    inv = table[table["subset"] == INVERTED_SUBSET]
    assert INVERTED_SUBSET == "oos:inverted"
    assert sorted(zip(inv["signal"], inv["horizon"])) == [("NEG", 1), ("NEG", 5)]
    assert (inv["label"] == "open").all()
    assert inv["ic_q"].isna().all() and inv["block_sign_share"].isna().all()
    assert (inv["promising"] == False).all()                            # noqa: E712
    assert (inv["direction"] == NEGATED_DIRECTION).all()
    assert (inv["ic_mean"] > 0).all()


def test_inverted_metrics_equal_signal_metrics_on_negated_signal(panel, result):
    f, y = panel
    table, splits = result
    cal = evaluation_calendar(f)
    tickers = sorted(f.index.get_level_values("ticker").unique())
    from investbot.evaluation.walkforward import oos_dates
    oos = oos_dates(splits)
    S = _wide(f["NEG"], cal, tickers)
    for h in HORIZONS:
        Y = _wide(y[f"FWD_TR_RET_{h}"], cal, tickers)
        Yo = _wide(y[f"FWD_TR_OO_RET_{h}"], cal, tickers)
        want = signal_metrics(-S.loc[oos], Y.loc[oos], h, MIN_NAMES, ROUND_TRIP_COST,
                              Yo.loc[oos], bootstrap=True)
        row = table[(table["signal"] == "NEG") & (table["horizon"] == h)
                    & (table["subset"] == INVERTED_SUBSET)].iloc[0]
        assert set(want) <= set(row.index)
        for k, v in want.items():
            _same([row[k]], [v])
        assert np.isfinite(row["held_ir_net_lo"]) and np.isfinite(row["held_cagr_net_hi"])


def test_inverted_held_beats_designed_polarity(result):
    table, _ = result
    for h in HORIZONS:
        sel = table[(table["signal"] == "NEG") & (table["horizon"] == h) & (table["label"] == "open")]
        designed = sel[sel["subset"] == "oos"].iloc[0]["held_cagr_net"]
        inverted = sel[sel["subset"] == INVERTED_SUBSET].iloc[0]["held_cagr_net"]
        assert inverted > designed


def test_no_inverted_promising_means_no_extra_rows(panel):
    f, y = panel
    table, _ = evaluate(f, y, HORIZONS, signals=["BASE_MOM_20", "TECH_SCORE"])
    assert promising_inverted(table) == []
    assert not (table["subset"] == INVERTED_SUBSET).any()
    assert "No promising signal has an inverted IC" in inverted_section(table)


def test_headline_stats_identical_with_and_without_the_step(result):
    table, _ = result
    inv = table["subset"] == INVERTED_SUBSET
    assert inv.any()
    base = table[~inv].drop(columns=MTCOLS)
    without = _add_multiple_testing(base.copy())
    # BH over the full table including the inverted rows must pick only subset "oos"
    with_rows = _add_multiple_testing(table.drop(columns=MTCOLS).copy())
    for t in (without, with_rows):
        h_new, h_ref = headline(t), headline(table[~inv])
        for c in ("ic_q", "block_sign_share", "promising", "direction"):
            _same(h_new[c].to_numpy(), h_ref[c].to_numpy())
    # the inverted rows themselves never got a q-value
    assert with_rows.loc[with_rows["subset"] == INVERTED_SUBSET, "ic_q"].isna().all()
    # and the pre-existing rows are untouched by the extra rows
    pd.testing.assert_frame_equal(table[~inv].reset_index(drop=True),
                                  without.reset_index(drop=True)[table.columns],
                                  check_exact=True)


def test_universe_masking_applies_to_negated_signal(panel):
    f, y = panel
    rng = np.random.default_rng(5)
    inside = pd.Series(rng.uniform(size=len(f)) < 0.75, index=f.index)
    f = f.assign(IN_UNIVERSE=inside)
    table, splits = evaluate(f, y, HORIZONS, signals=["NEG", "TECH_SCORE"])
    cal = evaluation_calendar(f)
    tickers = sorted(f.index.get_level_values("ticker").unique())
    U = universe_frame(f, cal, tickers)
    S = signal_matrix(f, "NEG", cal, tickers, U)
    assert S.isna().to_numpy()[~U.to_numpy()].all()
    assert (-S).isna().to_numpy()[~U.to_numpy()].all()          # cells outside stay NaN
    from investbot.evaluation.walkforward import oos_dates
    oos = oos_dates(splits)
    h = 1
    Y = apply_universe(_wide(y[f"FWD_TR_RET_{h}"], cal, tickers), U)
    Yo = apply_universe(_wide(y[f"FWD_TR_OO_RET_{h}"], cal, tickers), U)
    want = signal_metrics(-S.loc[oos], Y.loc[oos], h, MIN_NAMES, ROUND_TRIP_COST, Yo.loc[oos], True)
    row = table[(table["signal"] == "NEG") & (table["horizon"] == h)
                & (table["subset"] == INVERTED_SUBSET)]
    assert len(row) == 1
    for k, v in want.items():
        _same([row.iloc[0][k]], [v])
    # without any masking the numbers differ (more cells enter the cross-section)
    raw = signal_metrics(-_wide(f["NEG"], cal, tickers).loc[oos],
                         _wide(y[f"FWD_TR_RET_{h}"], cal, tickers).loc[oos], h, MIN_NAMES,
                         ROUND_TRIP_COST, None, False)
    assert raw["ic_mean"] != want["ic_mean"]


def test_negation_is_truncation_invariant(panel):
    """The negated signal at t uses only what S uses at t: dropping the tail leaves it unchanged."""
    f, _ = panel
    cal = evaluation_calendar(f)
    tickers = sorted(f.index.get_level_values("ticker").unique())
    full = -signal_matrix(f, "NEG", cal, tickers, None)
    cut = cal[400]
    keep = f.index.get_level_values("date") <= cut
    part = -signal_matrix(f[keep], "NEG", cal[cal <= cut], tickers, None)
    pd.testing.assert_frame_equal(full.loc[:cut], part, check_exact=True)


def test_negated_ties_use_original_lowest_bucket():
    """Discrete signal: the negated top bucket is the original lowest non-empty bucket."""
    from investbot.evaluation.metrics import quantile_buckets, top_membership
    s = pd.DataFrame([[1, 1, 1, -1, -1, -1, 1, -1, 1, 1]], dtype=float)
    y = pd.DataFrame([[0.02] * 10])
    members_neg = top_membership(quantile_buckets(-s, y, min_names=5)).iloc[0].to_numpy()
    assert (members_neg == (s.iloc[0].to_numpy() == -1)).all()
