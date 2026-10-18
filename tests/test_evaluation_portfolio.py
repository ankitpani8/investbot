"""
Tests for the open-to-open label, the continuously-held portfolio and the
pooled IC (task 150). Synthetic data only, no network or DB.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from investbot.evaluation.evaluate import evaluate, headline
from investbot.evaluation.labels import forward_returns, label_columns
from investbot.evaluation.metrics import (
    daily_ic,
    held_periods,
    held_portfolio,
    ic_summary,
    pooled_ic,
)
from investbot.evaluation.stats import newey_west_tstat


def _bars(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    vol = rng.integers(100_000, 1_000_000, n).astype(float)
    idx = pd.bdate_range("2020-01-01", periods=n, name="date")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": vol}, index=idx)


def _wide(a, cols=None) -> pd.DataFrame:
    a = np.asarray(a, dtype=np.float64)
    cols = cols or [f"T{i:02d}" for i in range(a.shape[1])]
    return pd.DataFrame(a, index=pd.bdate_range("2024-01-01", periods=len(a)), columns=cols)


# ──────────────────────────────────────────────────────────
# FWD_OO_RET_h
# ──────────────────────────────────────────────────────────

def test_open_to_open_label_by_hand():
    idx = pd.bdate_range("2024-01-01", periods=7)
    bars = pd.DataFrame({"open": [10, 11, 12, 13, 14, 15, 16.0],
                         "close": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5]}, index=idx)
    y = forward_returns(bars, (1, 2))
    # t = 0: enter open[1] = 11, exit open[2] = 12 (h=1) / open[3] = 13 (h=2)
    assert y["FWD_OO_RET_1"][0] == pytest.approx(12 / 11 - 1)
    assert y["FWD_OO_RET_2"][0] == pytest.approx(13 / 11 - 1)
    assert y["FWD_OO_RET_2"][3] == pytest.approx(16 / 14 - 1)
    # last h + 1 rows have no label
    assert np.isnan(y["FWD_OO_RET_1"][-2:]).all() and np.isfinite(y["FWD_OO_RET_1"][-3])
    assert np.isnan(y["FWD_OO_RET_2"][-3:]).all() and np.isfinite(y["FWD_OO_RET_2"][-4])
    # the existing labels are unchanged
    assert y["FWD_RET_1"][0] == pytest.approx(11.5 / 11 - 1)
    assert y["FWD_CC_RET_2"][3] == pytest.approx(15.5 / 13.5 - 1)
    assert list(y) == label_columns((1, 2))
    assert label_columns((5,)) == ["FWD_RET_5", "FWD_CC_RET_5", "FWD_OO_RET_5"]


def test_open_to_open_periods_tile_the_timeline():
    bars = _bars(60, seed=3)
    h = 5
    y = forward_returns(bars, (h,))[f"FWD_OO_RET_{h}"]
    starts = np.arange(0, 40, h)
    compounded = np.prod(1 + y[starts]) - 1
    assert compounded == pytest.approx(bars["open"].iloc[41] / bars["open"].iloc[1] - 1)


@pytest.mark.parametrize("h", [1, 5, 20])
def test_open_to_open_reads_only_opens_t1_and_t1h(h):
    bars = _bars(80)
    t = 30
    col = f"FWD_OO_RET_{h}"
    base = forward_returns(bars, (h,))[col][t]
    # bars <= t, bars after t+1+h and every non-open column do not matter
    other = bars.copy()
    other.iloc[: t + 1] *= 1.7
    other.iloc[t + h + 2:] *= 0.3
    other["close"] *= 1.3
    assert forward_returns(other, (h,))[col][t] == pytest.approx(base)
    # truncating right after t+1+h does not change it
    assert forward_returns(bars.iloc[: t + h + 2], (h,))[col][t] == pytest.approx(base)
    # the entry open (t+1) and the exit open (t+1+h) do
    for k in (t + 1, t + 1 + h):
        moved = bars.copy()
        moved.iloc[k, moved.columns.get_loc("open")] *= 1.1
        assert forward_returns(moved, (h,))[col][t] != pytest.approx(base)


def test_labels_leave_bars_and_features_untouched():
    pytest.importorskip("talib")
    from investbot.evaluation.panel import asset_features, asset_labels
    bars = _bars(400, seed=11)
    before = bars.copy()
    f0 = asset_features(bars)
    labels = asset_labels(bars, (1, 5))
    pd.testing.assert_frame_equal(bars, before)
    pd.testing.assert_frame_equal(asset_features(bars), f0)
    assert not set(f0.columns) & set(labels.columns)
    assert "FWD_OO_RET_5" in labels and not any(c.startswith("FWD_") for c in f0.columns)


# ──────────────────────────────────────────────────────────
# held_portfolio
# ──────────────────────────────────────────────────────────

def test_held_portfolio_two_assets_by_hand():
    # 4 rows, h = 2: rebalances on rows 0 and 2; rows 1 and 3 must be ignored
    m = _wide([[1, 0], [0, 1], [0, 1], [1, 1]], ["A", "B"]).astype(bool)
    y = _wide([[0.10, 0.00], [9.0, 9.0], [0.05, -0.20], [9.0, 9.0]], ["A", "B"])
    c = 0.01
    p = held_periods(m, y, horizon=2, cost=c)
    assert list(p.index) == [m.index[0], m.index[2]]
    np.testing.assert_allclose(p["gross"], [0.10, -0.20])
    np.testing.assert_allclose(p["turnover"], [1.0, 1.0])          # A out, B in
    np.testing.assert_allclose(p["net"], [0.09, -0.21])
    np.testing.assert_allclose(p["universe"], [0.05, -0.075])

    out = held_portfolio(m, y, horizon=2, cost=c)
    years = 2 * 2 / 252
    assert out["n_periods"] == 2
    assert out["turnover"] == pytest.approx(1.0)
    assert out["cagr_gross"] == pytest.approx((1.10 * 0.80) ** (1 / years) - 1)
    assert out["cagr_net"] == pytest.approx((1.09 * 0.79) ** (1 / years) - 1)
    assert out["cagr_universe"] == pytest.approx((1.05 * 0.925) ** (1 / years) - 1)
    assert out["max_drawdown_net"] == pytest.approx(-0.21)        # 1.09 -> 1.09 * 0.79
    net = np.array([0.09, -0.21])
    act = net - np.array([0.05, -0.075])
    assert out["vol_ann_net"] == pytest.approx(net.std(ddof=1) * math.sqrt(126))
    assert out["info_ratio_net"] == pytest.approx(act.mean() / act.std(ddof=1) * math.sqrt(126))


def test_held_portfolio_drawdown_counts_first_period_loss():
    m = _wide([[1, 0], [1, 0], [1, 0]], ["A", "B"]).astype(bool)
    y = _wide([[-0.10, 0.0], [0.05, 0.0], [0.02, 0.0]], ["A", "B"])
    out = held_portfolio(m, y, horizon=1, cost=0.0)
    assert out["max_drawdown_net"] == pytest.approx(-0.10)
    assert out["turnover"] == pytest.approx(0.0)


def test_held_portfolio_whole_universe_costs_only_first_period():
    rng = np.random.default_rng(1)
    y = _wide(rng.normal(0, 0.02, (30, 6)))
    m = pd.DataFrame(True, index=y.index, columns=y.columns)
    c = 0.0032
    p = held_periods(m, y, horizon=5, cost=c)
    diff = (p["net"] - p["universe"]).to_numpy()
    assert diff[0] == pytest.approx(-c)
    np.testing.assert_allclose(diff[1:], 0.0, atol=1e-15)
    np.testing.assert_allclose(p["turnover"].to_numpy()[1:], 0.0)
    out = held_portfolio(m, y, horizon=5, cost=c)
    assert out["n_periods"] == 6
    assert out["cagr_gross"] == pytest.approx(out["cagr_universe"])
    assert out["cagr_net"] < out["cagr_universe"]


def test_held_portfolio_skips_missing_labels_and_empty_periods():
    m = _wide([[1, 1], [0, 0], [1, 0]], ["A", "B"]).astype(bool)
    y = _wide([[0.02, np.nan], [0.0, 0.0], [np.nan, 0.01]], ["A", "B"])
    p = held_periods(m, y, horizon=1, cost=0.0)
    assert list(p.index) == [m.index[0]]                 # row 1 empty, row 2 member has no label
    assert p["gross"].iloc[0] == pytest.approx(0.02)
    empty = held_portfolio(m.iloc[[1]], y.iloc[[1]], horizon=1)
    assert empty["n_periods"] == 0 and math.isnan(empty["cagr_net"])


# ──────────────────────────────────────────────────────────
# pooled_ic
# ──────────────────────────────────────────────────────────

def test_pooled_ic_equals_mean_ic_with_constant_cross_section():
    rng = np.random.default_rng(7)
    s = _wide(rng.normal(size=(40, 12)))
    y = _wide(rng.normal(size=(40, 12)))
    for h in (1, 5):
        out = pooled_ic(s, y, horizon=h)
        ic = daily_ic(s, y, min_names=10)
        assert abs(out["pooled_ic"] - ic.mean()) < 1e-12
        assert out["pooled_t_nw"] == pytest.approx(ic_summary(ic, h)["ic_t_nw"], rel=1e-10)
        assert out["pooled_t_nw"] == pytest.approx(newey_west_tstat(ic.to_numpy(), h - 1),
                                                   rel=1e-10)
    assert out["pooled_n_dates"] == 40 and out["pooled_n_obs"] == 480


def test_pooled_ic_on_thin_cross_sections_matches_brute_force():
    rng = np.random.default_rng(9)
    n_dates, n_names = 50, 12
    s = rng.normal(size=(n_dates, n_names))
    y = 0.5 * s + rng.normal(size=(n_dates, n_names))
    keep = rng.integers(3, 10, n_dates)                      # 3..9 names per date
    for d in range(n_dates):
        drop = rng.permutation(n_names)[keep[d]:]
        s[d, drop] = np.nan
    S, Y = _wide(s), _wide(y)
    assert daily_ic(S, Y).isna().all()                       # MIN_NAMES = 10
    out = pooled_ic(S, Y, horizon=1)
    assert np.isfinite(out["pooled_ic"]) and np.isfinite(out["pooled_t_nw"])
    assert out["pooled_ic"] > 0.2 and out["pooled_t_nw"] > 3
    assert out["pooled_n_dates"] == n_dates and out["pooled_n_obs"] == int(keep.sum())
    # brute force with pandas ranks
    num = ss = yy = 0.0
    for d in range(n_dates):
        ok = S.iloc[d].notna()
        n = int(ok.sum())
        rs = (S.iloc[d][ok].rank() - 1) / (n - 1) - 0.5
        ry = (Y.iloc[d][ok].rank() - 1) / (n - 1) - 0.5
        rs, ry = rs - rs.mean(), ry - ry.mean()
        num += float((rs * ry).sum())
        ss += float((rs * rs).sum())
        yy += float((ry * ry).sum())
    assert out["pooled_ic"] == pytest.approx(num / math.sqrt(ss * yy), rel=1e-12)


def test_pooled_ic_min_names_ties_and_empty():
    s = _wide([[1, 2, np.nan, np.nan], [1, 1, 1, 2], [3, 3, 3, 3]])
    y = _wide([[1, 2, 3, 4], [4, 3, 2, 1], [1, 2, 3, 4]])
    out = pooled_ic(s, y, horizon=1)
    # row 0 has 2 names (< 3), row 2 has no S variance: only row 1 is used
    assert out["pooled_n_dates"] == 1 and out["pooled_n_obs"] == 4
    exp = pd.Series([1, 1, 1, 2.0]).rank().corr(pd.Series([4, 3, 2, 1.0]).rank())
    assert out["pooled_ic"] == pytest.approx(exp)
    none = pooled_ic(s.iloc[[0]], y.iloc[[0]], horizon=1)
    assert none["pooled_n_dates"] == 0 and math.isnan(none["pooled_ic"])


# ──────────────────────────────────────────────────────────
# evaluate() and the report
# ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_eval():
    rng = np.random.default_rng(4)
    dates = pd.bdate_range("2021-01-01", "2023-12-29")
    tickers = [f"S{i:02d}" for i in range(20)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    n = len(idx)
    labels = pd.DataFrame({c: rng.normal(0, 0.02, n) for c in label_columns((1, 5))}, index=idx)
    regime = rng.choice([0.0, 1.0, 2.0], n, p=[0.45, 0.45, 0.10])
    features = pd.DataFrame({
        "TECH_SCORE": rng.normal(size=n),
        "ORACLE": labels["FWD_OO_RET_5"].to_numpy(),
        "REGIME": regime,
        "REGIME_CONFIDENCE": rng.uniform(size=n),
    }, index=idx)
    return evaluate(features, labels, (1, 5))


def test_evaluate_adds_held_and_pooled_columns(small_eval):
    table, _ = small_eval
    held = [c for c in table.columns if c.startswith("held_")]
    assert {"held_cagr_net", "held_cagr_universe", "held_max_drawdown_net",
            "held_info_ratio_net", "held_vol_ann_net", "held_n_periods"} <= set(held)
    assert {"lo_gross_ann", "lo_net_ann", "top_turnover"} <= set(table.columns)
    head = headline(table).set_index(["signal", "horizon"])
    assert head["held_n_periods"].min() > 0
    # an oracle on the open-to-open label beats the universe when held
    assert head.loc[("ORACLE", 5), "held_cagr_net"] > head.loc[("ORACLE", 5), "held_cagr_universe"]
    assert head.loc[("ORACLE", 5), "held_info_ratio_net"] > 2
    reg = table["subset"].str.startswith("regime:")
    assert table.loc[reg, "pooled_n_dates"].gt(0).all()
    assert table.loc[~reg, ["pooled_ic", "pooled_t_nw"]].isna().all().all()
    # the thin regime (10% of names, ~2 per date) has no daily IC but a pooled one
    chaotic = table[table["subset"] == "regime:chaotic"]
    assert not chaotic.empty and (chaotic["n_days"] == 0).all()
    assert chaotic["pooled_ic"].notna().all() and chaotic["pooled_n_dates"].gt(0).all()
    # BH family is unchanged: only the headline rows carry q-values
    assert table.loc[table["ic_q"].notna(), "subset"].eq("oos").all()


def test_report_has_held_table_and_pooled_regime_columns(small_eval):
    from investbot.evaluation.report import RunInfo, render_report
    table, splits = small_eval
    md = render_report(table, splits, RunInfo(20, pd.Timestamp("2021-01-01"),
                                              pd.Timestamp("2023-12-29"), (1, 5), None),
                       pd.Timestamp("2024-01-02").date())
    assert "## Held portfolio (open-to-open, compounded)" in md
    assert "open→close per period" in md
    assert "pooled IC" in md
    assert "## Verdict" in md and "## By regime (OOS)" in md


def test_report_path_does_not_overwrite(tmp_path):
    from datetime import date

    from investbot.evaluation.report import report_paths
    d = date(2026, 10, 18)
    rep, csv = report_paths(tmp_path, tmp_path, d)
    assert rep.name == "eval_20261018.md" and csv.name == "eval_metrics_20261018.csv"
    rep.write_text("x")
    rep2, csv2 = report_paths(tmp_path, tmp_path, d)
    assert rep2.name == "eval_20261018_b.md" and csv2.name == "eval_metrics_20261018_b.csv"
    rep2.write_text("y")
    assert report_paths(tmp_path, tmp_path, d)[0].name == "eval_20261018_c.md"
