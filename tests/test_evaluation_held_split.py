"""Held-portfolio survivorship split (task 470): descriptive, partitions OOS periods only."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from investbot.evaluation.evaluate import evaluate, headline
from investbot.evaluation.metrics import (
    HELD_SPLIT_KEYS,
    SURVIVORSHIP_KEYS,
    TRADING_DAYS,
    held_complete_split,
    held_periods,
    held_portfolio,
)
from investbot.evaluation.report import RunInfo, held_split_rows, render_report
from tests.test_evaluation import PRICE_STEMS, TR_STEMS, label_columns


def _periods(n: int = 30, horizon: int = 5, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n * horizon)[::horizon]
    univ = rng.normal(0.002, 0.02, n)
    net = univ + rng.normal(0.001, 0.01, n)
    return pd.DataFrame({"gross": net, "turnover": np.zeros(n), "net": net, "universe": univ},
                        index=idx)


def test_keys_and_order():
    assert HELD_SPLIT_KEYS[:5] == ("n_periods_pre_complete", "cagr_net_pre_complete",
                                   "cagr_universe_pre_complete", "excess_cagr_pre_complete",
                                   "info_ratio_net_pre_complete")
    assert HELD_SPLIT_KEYS[5:] == tuple(k.replace("pre_", "post_") for k in HELD_SPLIT_KEYS[:5])
    p = _periods()
    assert tuple(held_complete_split(p, 5, p.index[10])) == HELD_SPLIT_KEYS


def test_pre_and_post_growth_multiply_to_the_full_window_growth():
    h = 5
    p = _periods(40, h)
    out = held_complete_split(p, h, p.index[17])
    n_pre, n_post = out["n_periods_pre_complete"], out["n_periods_post_complete"]
    assert (n_pre, n_post) == (17, 23)
    growth = ((1 + out["cagr_net_pre_complete"]) ** (n_pre * h / TRADING_DAYS)
              * (1 + out["cagr_net_post_complete"]) ** (n_post * h / TRADING_DAYS))
    assert growth == pytest.approx(float(np.prod(1 + p["net"])), abs=1e-12, rel=0)
    ug = ((1 + out["cagr_universe_pre_complete"]) ** (n_pre * h / TRADING_DAYS)
          * (1 + out["cagr_universe_post_complete"]) ** (n_post * h / TRADING_DAYS))
    assert ug == pytest.approx(float(np.prod(1 + p["universe"])), abs=1e-12, rel=0)
    for s in ("pre", "post"):
        assert out[f"excess_cagr_{s}_complete"] == pytest.approx(
            out[f"cagr_net_{s}_complete"] - out[f"cagr_universe_{s}_complete"], abs=1e-15)


def test_all_periods_before_the_cut_leave_post_empty():
    p = _periods()
    out = held_complete_split(p, 5, p.index[-1] + pd.Timedelta(days=1))
    assert out["n_periods_post_complete"] == 0
    for k in HELD_SPLIT_KEYS[6:]:
        assert math.isnan(out[k])
    assert out["n_periods_pre_complete"] == len(p)
    assert np.isfinite(out["cagr_net_pre_complete"])


def test_fewer_than_two_periods_is_nan():
    p = _periods()
    out = held_complete_split(p, 5, p.index[-1])      # one period on the post side
    assert out["n_periods_post_complete"] == 1
    assert all(math.isnan(out[k]) for k in HELD_SPLIT_KEYS[6:])


def test_cut_on_a_rebalance_date_counts_as_post():
    p = _periods()
    cut = p.index[12]
    out = held_complete_split(p, 5, cut.date())
    assert out["n_periods_pre_complete"] == 12 and out["n_periods_post_complete"] == len(p) - 12
    # a cut between rebalances: the period that started before the cut stays "pre"
    mid = cut - pd.Timedelta(days=1)
    assert held_complete_split(p, 5, mid.date())["n_periods_pre_complete"] == 12


def test_all_post_matches_held_portfolio():
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2021-01-01", periods=120)
    cols = [f"S{i}" for i in range(12)]
    Y = pd.DataFrame(rng.normal(0.001, 0.02, (120, 12)), index=dates, columns=cols)
    members = pd.DataFrame(rng.uniform(size=(120, 12)) < 0.3, index=dates, columns=cols)
    h = 5
    p = held_periods(members, Y, h)
    full = held_portfolio(members, Y, h, bootstrap=False)
    out = held_complete_split(p, h, dates[0] - pd.Timedelta(days=1))
    assert out["info_ratio_net_post_complete"] == full["info_ratio_net"]
    assert out["cagr_net_post_complete"] == full["cagr_net"]
    assert out["cagr_universe_post_complete"] == full["cagr_universe"]
    assert out["n_periods_post_complete"] == full["n_periods"]


def test_truncating_after_the_cut_leaves_pre_unchanged():
    p = _periods(60)
    cut = p.index[25]
    full = held_complete_split(p, 5, cut)
    short = held_complete_split(p.iloc[:40], 5, cut)
    for k in HELD_SPLIT_KEYS[:5]:
        assert short[k] == full[k]


@pytest.fixture(scope="module")
def tables():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2021-01-01", "2023-06-30")
    idx = pd.MultiIndex.from_product([dates, [f"S{i:02d}" for i in range(14)]],
                                     names=["date", "ticker"])
    n = len(idx)
    labels = pd.DataFrame({c: rng.normal(0, 0.02, n)
                           for c in label_columns((1, 5), PRICE_STEMS)}, index=idx)
    labels = labels.assign(**{c.replace(p, t, 1): labels[c] for p, t in zip(PRICE_STEMS, TR_STEMS)
                              for c in label_columns((1, 5), (p,))})
    features = pd.DataFrame({"TECH_SCORE": rng.normal(size=n),
                             "ORACLE": labels["FWD_TR_RET_5"].to_numpy(),
                             "REGIME": rng.integers(0, 2, n).astype(float),
                             "REGIME_CONFIDENCE": rng.uniform(size=n)}, index=idx)
    cut = pd.Timestamp("2022-10-03").date()
    table, splits = evaluate(features, labels, (1, 5), complete_from=cut,
                             signals=["TECH_SCORE", "ORACLE"], robust=False)
    return table, splits, cut


def test_evaluate_columns_last_and_consistent_with_the_headline_row(tables):
    table, _, _ = tables
    assert list(table.columns[-14:]) == [*SURVIVORSHIP_KEYS, *HELD_SPLIT_KEYS]
    head = headline(table)
    rest = table.drop(head.index)
    assert head[list(HELD_SPLIT_KEYS)].notna().any(axis=None)
    assert rest[list(HELD_SPLIT_KEYS)].isna().all(axis=None)
    h = head.set_index(["signal", "horizon"])
    # the two sides cover every period of the headline held portfolio
    assert ((h["n_periods_pre_complete"] + h["n_periods_post_complete"])
            == h["held_n_periods"]).all()


def test_report_table_lists_rows_that_beat_the_universe(tables):
    table, splits, cut = tables
    head = headline(table).copy()
    head["promising"] = False
    head["held_cagr_universe"] = 0.0
    head["held_cagr_net"] = -0.1
    head.loc[head.index[(head["signal"] == "ORACLE") & (head["horizon"] == 5)], "held_cagr_net"] = 0.2
    assert set(held_split_rows(head)["signal"]) == {"TECH_SCORE", "ORACLE"}
    rows = held_split_rows(head)
    assert rows.index.is_unique
    assert rows[rows["signal"] == "ORACLE"]["horizon"].tolist() == [5]   # beats, not promising
    t2 = table.copy()
    t2.loc[head.index, "promising"] = False
    t2.loc[head.index, "held_cagr_universe"] = 0.0
    t2.loc[head.index, "held_cagr_net"] = head["held_cagr_net"]
    info = RunInfo(14, pd.Timestamp("2021-01-04"), pd.Timestamp("2023-06-30"), (1, 5), None,
                   complete_from=cut)
    md = render_report(t2, splits, info, pd.Timestamp("2023-07-01").date())
    sec = md.split("## Survivorship split (descriptive)")[1].split("\n## ")[0]
    sub = sec.split(f"### Held portfolio before / from {cut} (descriptive)")[1]
    assert "rebalance date" in sub and "survivorship-inflated" in sub
    assert "| ORACLE | 5 |" in sub and "| TECH_SCORE | 1 |" in sub
    assert "| ORACLE | 1 |" not in sub


def test_report_without_the_new_keys_prints_the_ic_table_only(tables):
    table, splits, cut = tables
    old = table.drop(columns=list(HELD_SPLIT_KEYS))
    info = RunInfo(14, pd.Timestamp("2021-01-04"), pd.Timestamp("2023-06-30"), (1, 5), None,
                   complete_from=cut)
    md = render_report(old, splits, info, pd.Timestamp("2023-07-01").date())
    sec = md.split("## Survivorship split (descriptive)")[1].split("\n## ")[0]
    assert "mean IC before" in sec and "Held portfolio before" not in sec
