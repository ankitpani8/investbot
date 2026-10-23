"""
KPI summary of simulator runs (investbot/backtest/summary.py). Synthetic fixtures only.
"""

import copy
import dataclasses
import math
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from investbot.backtest import summary
from investbot.backtest.simulator import SimulationResult, daily_returns, simulate
from investbot.backtest.summary import (
    GO_NO_GO,
    KPI_KEYS,
    RISK_FREE_ANNUAL,
    TRADING_DAYS_PER_YEAR,
    compare,
    concentration,
    fy_tax_table,
    go_no_go,
    kpis,
    monthly_table,
    tax_alpha,
)
from investbot.evaluation import metrics, performance
from investbot.trading.tax_rules import regime_for
from investbot.trading.transaction_costs import deductible_costs, estimate_costs

C = 500_000.0


def _same(a, b):
    """Equality of two kpi values / dicts with NaN == NaN."""
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, rel=1e-12, abs=1e-12)
    return a == b


def assert_kpis_equal(x: dict, y: dict, skip=()):
    assert list(x) == list(y)
    for k in x:
        if k not in skip:
            assert _same(x[k], y[k]), (k, x[k], y[k])


# --- fixtures ----------------------------------------------------------------------------


def hand_case():
    """3 assets, two years, flat prices with three steps, one dividend, one FY rollover.

    2024-01-02 buy 0.3 each at 100 / 200 / 50; asset 1 jumps to 120 (2024-11-01) and is sold
    on 2024-12-03 (short-term gain, FY 2024-25 paid on 2025-04-01); asset 2 pays 5 on
    2024-09-02; asset 2 -> 220 (2025-06-02), asset 3 -> 100 (2025-09-01). Held to the end.
    """
    dates = pd.bdate_range("2024-01-01", "2025-12-31")
    idx = pd.Series(range(len(dates)), index=dates)

    def path(base, steps):
        p = pd.Series(float(base), index=dates)
        for d, v in steps:
            p[p.index >= pd.Timestamp(d)] = float(v)
        return p

    close = pd.DataFrame(
        {
            1: path(100, [("2024-11-01", 120)]),
            2: path(200, [("2025-06-02", 220)]),
            3: path(50, [("2025-09-01", 100)]),
        }
    )
    open_ = close.copy()
    divs = pd.DataFrame(0.0, index=dates, columns=[1, 2, 3])
    divs.loc["2024-09-02", 2] = 5.0
    reg = regime_for("equity", date(2024, 1, 2))
    buy_cost = estimate_costs("BUY", 150_000.0, reg).total
    cash1 = C - 450_000 - 3 * buy_cost
    cash_after_div = cash1 + 5.0 * 750
    nav_open = cash_after_div + 1500 * 120 + 750 * 200 + 3000 * 50
    w = pd.DataFrame(
        {1: [0.3, 0.0], 2: [0.3, 750 * 200 / nav_open], 3: [0.3, 3000 * 50 / nav_open]},
        index=pd.to_datetime(["2024-01-01", "2024-12-02"]),
    )
    res = simulate(w, open_, close, divs, initial_capital=C)
    return res, {"buy_cost": buy_cost, "cash_after_div": cash_after_div, "reg": reg, "idx": idx}


def random_case(tax=True, seed=11, n=700):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n)
    steps = rng.normal(0.0004, 0.012, size=(n, 3))
    close = pd.DataFrame(100.0 * np.exp(np.cumsum(steps, axis=0)), index=dates, columns=[1, 2, 3])
    open_ = close.shift(1).fillna(close.iloc[0])
    divs = pd.DataFrame(0.0, index=dates, columns=[1, 2, 3])
    divs.iloc[200, 0] = 1.5
    rebal = dates[::21]
    raw = rng.dirichlet(np.ones(4), size=len(rebal))[:, :3]  # 4th = cash
    w = pd.DataFrame(raw, index=rebal, columns=[1, 2, 3])
    return simulate(w, open_, close, divs, initial_capital=C, tax=tax), (w, open_, close, divs)


# --- constants and shape -----------------------------------------------------------------


def test_constants_are_the_shared_ones():
    assert TRADING_DAYS_PER_YEAR == 252
    assert TRADING_DAYS_PER_YEAR is metrics.TRADING_DAYS
    assert RISK_FREE_ANNUAL is metrics.RISK_FREE_RATE_ANNUAL
    src = Path(summary.__file__).read_text()
    assert not re.search(r"^(TRADING_DAYS_PER_YEAR|RISK_FREE_ANNUAL)\s*[:=]", src, re.M)


def test_no_dropna_in_module():
    assert "dropna" not in Path(summary.__file__).read_text()


def test_kpi_keys_and_order():
    res, _ = random_case()
    k = kpis(res)
    assert list(k) == list(KPI_KEYS) == [
        "start", "end", "years", "initial_nav", "terminal_value", "cagr", "ann_vol", "sharpe",
        "sortino", "max_drawdown", "max_dd_duration_days", "calmar", "var95", "cvar95",
        "total_costs", "total_dividends", "total_tax_paid", "terminal_tax_accrued",
        "liquidation_tax", "turnover_annual", "win_rate", "avg_win_loss_ratio", "profit_factor",
        "n_realisations", "n_stale_episodes",
    ]
    assert kpis(res, after_tax=False).keys() == k.keys()


# --- hand-computable case ----------------------------------------------------------------


def test_hand_computed_kpis():
    res, h = hand_case()
    reg, buy_cost = h["reg"], h["buy_cost"]

    # costs and dividends, from first principles
    sell_reg = regime_for("equity", date(2024, 12, 3))
    sell1 = estimate_costs("SELL", 1500 * 120.0, sell_reg)
    total_costs = 3 * buy_cost + sell1.total
    total_div = 5.0 * 750

    # the asset-1 sale: short-term gain, FY 2024-25 tax paid 2025-04-01
    basis = 150_000 + deductible_costs(estimate_costs("BUY", 150_000.0, reg))
    gain = 180_000 - basis - deductible_costs(sell1)
    tax_paid = gain * 0.20 * 1.04
    cash = h["cash_after_div"] + 180_000 - sell1.total - tax_paid

    # terminal liquidation of assets 2 and 3 on 2025-12-31 (held > 365 days: long term)
    end_reg = regime_for("equity", date(2025, 12, 31))
    ltcg = 0.0
    for q, buy_px, sell_px in ((750, 200.0, 220.0), (3000, 50.0, 100.0)):
        bought = q * buy_px
        b = bought + deductible_costs(estimate_costs("BUY", bought, reg))
        s = estimate_costs("SELL", q * sell_px, end_reg)
        ltcg += q * sell_px - b - deductible_costs(s)
    liq_tax = max(0.0, ltcg - 125_000.0) * 0.125 * 1.04
    assert liq_tax > 0
    nav_end = cash + 750 * 220 + 3000 * 100
    terminal = nav_end - liq_tax

    years = (date(2025, 12, 31) - date(2024, 1, 1)).days / 365.25
    cagr = (terminal / C) ** (1 / years) - 1

    k = kpis(res)
    assert k["start"] == pd.Timestamp("2024-01-01") and k["end"] == pd.Timestamp("2025-12-31")
    assert k["years"] == pytest.approx(years, abs=1e-12)
    assert k["initial_nav"] == pytest.approx(C, abs=1e-9)
    assert k["total_costs"] == pytest.approx(total_costs, abs=1e-6)
    assert k["total_dividends"] == pytest.approx(total_div, abs=1e-9)
    assert k["total_tax_paid"] == pytest.approx(tax_paid, abs=1e-6)
    assert k["terminal_tax_accrued"] == 0.0
    assert k["liquidation_tax"] == pytest.approx(liq_tax, abs=1e-6)
    assert k["terminal_value"] == pytest.approx(terminal, abs=1e-6)
    assert k["cagr"] == pytest.approx(cagr, rel=1e-12)
    assert k["n_realisations"] == 1
    assert k["n_stale_episodes"] == 0
    # one winning trade only: win rate is NaN with < 2 trades (performance.trade_stats)
    assert math.isnan(k["win_rate"])

    # turnover: 3 buys + 1 sell, over mean nav and years
    traded = 3 * 150_000 + 180_000
    assert k["turnover_annual"] == pytest.approx(traded / res.daily["nav"].mean() / years, rel=1e-12)

    pre = kpis(res, after_tax=False)
    assert pre["terminal_value"] == pytest.approx(nav_end, abs=1e-6)

    t = fy_tax_table(res)
    assert list(t.index) == ["2023-24", "2024-25", "2025-26"]
    assert t.loc["2024-25", "realized_stcg"] == pytest.approx(gain, abs=1e-6)
    assert t.loc["2024-25", "estimated_tax_liability"] == pytest.approx(tax_paid, abs=1e-6)
    assert list(t["paid"]) == [True, True, False]


def test_after_tax_vs_pre_tax_differ_by_exactly_the_tax_terms():
    res, _ = random_case(tax=True)
    a, p = kpis(res, after_tax=True), kpis(res, after_tax=False)
    assert a["liquidation_tax"] > 0 or a["terminal_tax_accrued"] > 0
    tax_terms = float(res.daily["tax_accrued"].iloc[-1]) + res.liquidation_tax
    assert p["terminal_value"] - a["terminal_value"] == pytest.approx(tax_terms, abs=1e-9)
    assert a["terminal_value"] == pytest.approx(
        res.daily["nav"].iloc[-1] - res.daily["tax_accrued"].iloc[-1] - res.liquidation_tax
    )
    same = ("terminal_value", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown",
            "max_dd_duration_days", "calmar", "var95", "cvar95", "initial_nav")
    for key in KPI_KEYS:
        if key not in same:
            assert _same(a[key], p[key]), key
    assert a["total_tax_paid"] > 0
    assert p["cagr"] > a["cagr"]


def test_tax_false_run_has_equal_after_and_pre_tax():
    res, _ = random_case(tax=False)
    assert res.fiscal_years == {}
    a, p = kpis(res, after_tax=True), kpis(res, after_tax=False)
    assert_kpis_equal(a, p)
    assert a["total_tax_paid"] == 0.0 and a["liquidation_tax"] == 0.0
    table = fy_tax_table(res)
    assert len(table) == 0 and list(table.columns) == summary.FY_TAX_COLUMNS
    row = compare({"naive": res}).loc["naive"]
    assert row["tax_drag_cagr"] == 0.0


def test_return_based_figures_match_independent_numpy():
    res, _ = random_case()
    k = kpis(res)
    r = (res.daily["nav_after_tax"] / res.daily["nav_after_tax"].shift(1) - 1.0).to_numpy()[1:]
    assert k["ann_vol"] == pytest.approx(r.std(ddof=1) * math.sqrt(252), rel=1e-12)
    rf = (1 + RISK_FREE_ANNUAL) ** (1 / 252) - 1
    ex = r - rf
    assert k["sharpe"] == pytest.approx(ex.mean() / ex.std(ddof=1) * math.sqrt(252), rel=1e-12)
    dd = np.sqrt(np.mean(np.minimum(ex, 0) ** 2))
    assert k["sortino"] == pytest.approx(ex.mean() / dd * math.sqrt(252), rel=1e-12)
    curve = res.daily["nav_after_tax"].to_numpy()
    assert k["max_drawdown"] == pytest.approx((curve / np.maximum.accumulate(curve) - 1).min(), rel=1e-9)
    q = np.quantile(r, 0.05)
    assert k["var95"] == pytest.approx(-q, rel=1e-12)
    assert k["cvar95"] == pytest.approx(-r[r <= q].mean(), rel=1e-12)
    assert k["calmar"] == pytest.approx(k["cagr"] / abs(k["max_drawdown"]), rel=1e-12)
    assert k["max_dd_duration_days"] > 0
    gains = np.array([x.gain for x in res.realisations])
    assert k["n_realisations"] == len(gains) > 2
    assert k["win_rate"] == pytest.approx((gains > 0).mean())
    assert k["profit_factor"] == pytest.approx(gains[gains > 0].sum() / abs(gains[gains < 0].sum()))
    assert k["avg_win_loss_ratio"] == pytest.approx(
        gains[gains > 0].mean() / abs(gains[gains < 0].mean())
    )
    # the rf parameter is honoured
    assert kpis(res, rf_annual=0.0)["sharpe"] > k["sharpe"]


# --- monthly and FY tables ---------------------------------------------------------------


def test_monthly_table_grid_and_year_total():
    res, _ = random_case()
    tbl = monthly_table(res)
    assert list(tbl.columns) == [*range(1, 13), "year"]
    r = daily_returns(res)
    for (y, m), grp in r.groupby([r.index.year, r.index.month]):
        expect = np.prod(1 + grp.to_numpy()[np.isfinite(grp.to_numpy())]) - 1
        assert tbl.loc[y, m] == pytest.approx(expect, rel=1e-12)
    # the grid starts in January 2021 and ends in 2023 before December: those months are NaN
    assert list(tbl.index) == [2021, 2022, 2023]
    assert tbl.loc[2023].iloc[:12].isna().any()
    for y in tbl.index:
        months = tbl.loc[y, list(range(1, 13))].dropna()
        assert tbl.loc[y, "year"] == pytest.approx(np.prod(1 + months.to_numpy()) - 1, rel=1e-12)
    # compounding the months equals compounding the days
    total = (1 + tbl["year"]).prod() - 1
    assert total == pytest.approx(np.nanprod(1 + r.to_numpy()) - 1, rel=1e-9)
    pre = monthly_table(res, after_tax=False)
    assert not pre.equals(tbl)


def test_monthly_table_single_row_is_empty_grid():
    res, _ = hand_case()
    short = dataclasses.replace(res, daily=res.daily.iloc[:1])
    t = monthly_table(short)
    assert len(t) == 0 and list(t.columns) == [*range(1, 13), "year"]


def test_fy_tax_table_columns_and_paid_flag():
    res, _ = random_case()
    t = fy_tax_table(res)
    assert list(t.index) == list(res.fiscal_years)
    assert list(t["paid"]) == [True] * (len(t) - 1) + [False]
    for label, fy in res.fiscal_years.items():
        assert t.loc[label, "realized_stcg"] == fy.realized_stcg
        assert t.loc[label, "realized_ltcg"] == fy.realized_ltcg
        assert t.loc[label, "ltcg_exemption_used"] == fy.ltcg_exemption_used
        assert t.loc[label, "realized_stcl"] == fy.realized_stcl
        assert t.loc[label, "carried_forward_ltcl"] == fy.carried_forward_ltcl
        assert t.loc[label, "estimated_tax_liability"] == fy.estimated_tax_liability
    # the paid liabilities are exactly what left the cash
    assert t.loc[t["paid"], "estimated_tax_liability"].sum() == pytest.approx(
        res.daily["tax_paid"].sum(), abs=1e-6
    )
    assert t["paid"].dtype == bool


# --- compare -----------------------------------------------------------------------------


def test_compare_columns_drag_and_excess():
    taxed, _ = random_case(tax=True)
    naive, _ = random_case(tax=False)
    runs = {"strat": taxed, "bench": naive}
    snapshot = copy.deepcopy(runs)
    c = compare(runs, benchmark="bench")
    assert list(c.index) == ["strat", "bench"]
    after = [f"{k}_after_tax" for k in KPI_KEYS]
    pre = [f"{k}_pre_tax" for k in KPI_KEYS]
    assert list(c.columns) == [
        *after, *pre, "tax_drag_cagr", "excess_cagr_after_tax", "ruined_on"
    ]
    row = c.loc["strat"]
    ka, kp = kpis(taxed, after_tax=True), kpis(taxed, after_tax=False)
    assert row["cagr_after_tax"] == ka["cagr"] and row["cagr_pre_tax"] == kp["cagr"]
    assert row["tax_drag_cagr"] == pytest.approx(kp["cagr"] - ka["cagr"])
    assert row["tax_drag_cagr"] > 0
    assert row["excess_cagr_after_tax"] == pytest.approx(
        ka["cagr"] - kpis(naive, after_tax=True)["cagr"]
    )
    assert c.loc["bench", "excess_cagr_after_tax"] == 0.0
    # inputs untouched, no benchmark -> no excess column
    for k in runs:
        pd.testing.assert_frame_equal(runs[k].daily, snapshot[k].daily)
        assert runs[k].liquidation_tax == snapshot[k].liquidation_tax
    assert "excess_cagr_after_tax" not in compare(runs).columns
    with pytest.raises(ValueError):
        compare(runs, benchmark="nope")


def test_tax_alpha_is_the_after_tax_cagr_difference():
    a, _ = random_case(tax=True, seed=11)
    b, _ = random_case(tax=True, seed=12)
    assert tax_alpha(a, b) == pytest.approx(kpis(a)["cagr"] - kpis(b)["cagr"])


# --- go / no-go --------------------------------------------------------------------------


def test_go_no_go_thresholds_table():
    assert GO_NO_GO["excess_cagr_after_tax"]["minimum"] == 0.02
    assert GO_NO_GO["excess_cagr_after_tax"]["target"] == 0.05
    assert (GO_NO_GO["max_drawdown"]["minimum"], GO_NO_GO["max_drawdown"]["target"]) == (0.25, 0.20)
    assert (GO_NO_GO["sharpe"]["minimum"], GO_NO_GO["sharpe"]["target"]) == (0.8, 1.2)
    assert (GO_NO_GO["win_rate"]["minimum"], GO_NO_GO["win_rate"]["target"]) == (0.45, 0.55)
    assert (GO_NO_GO["tax_alpha"]["minimum"], GO_NO_GO["tax_alpha"]["target"]) == (0.005, 0.015)


def test_go_no_go_missing_metrics_are_none_not_false():
    row = pd.Series({"sharpe_after_tax": 1.0})
    t = go_no_go(row)
    assert list(t.columns) == ["metric", "value", "minimum", "target", "meets_minimum", "meets_target"]
    assert list(t["metric"]) == list(GO_NO_GO)
    by = t.set_index("metric")
    assert by.loc["sharpe", "meets_minimum"] is True and by.loc["sharpe", "meets_target"] is False
    for m in ("excess_cagr_after_tax", "max_drawdown", "win_rate", "tax_alpha"):
        assert math.isnan(by.loc[m, "value"])
        assert by.loc[m, "meets_minimum"] is None and by.loc[m, "meets_target"] is None
    # NaN present in the row behaves like a missing metric
    nan_row = pd.Series({"sharpe_after_tax": math.nan, "win_rate_after_tax": math.inf})
    t2 = go_no_go(nan_row).set_index("metric")
    assert t2.loc["sharpe", "meets_minimum"] is None and t2.loc["win_rate", "meets_target"] is None


def test_go_no_go_boundaries_are_strict():
    def run(**kw):
        return go_no_go(pd.Series(kw)).set_index("metric")

    t = run(excess_cagr_after_tax=0.02, sharpe_after_tax=0.8, win_rate_after_tax=0.45,
            tax_alpha=0.005, max_drawdown_after_tax=-0.25)
    assert not t["meets_minimum"].astype(bool).any()
    assert not t["meets_target"].astype(bool).any()
    t = run(excess_cagr_after_tax=0.05, sharpe_after_tax=1.2, win_rate_after_tax=0.55,
            tax_alpha=0.015, max_drawdown_after_tax=-0.20)
    assert t["meets_minimum"].astype(bool).all()          # target values pass the minimums
    assert not t["meets_target"].astype(bool).any()       # but sit exactly on the target
    eps = 1e-9
    t = run(excess_cagr_after_tax=0.05 + eps, sharpe_after_tax=1.2 + eps, win_rate_after_tax=0.55 + eps,
            tax_alpha=0.015 + eps, max_drawdown_after_tax=-(0.20 - eps))
    assert t["meets_target"].astype(bool).all()
    # drawdown is read as a loss magnitude and passes for a smaller one only
    t = run(max_drawdown_after_tax=-0.30)
    assert t.loc["max_drawdown", "value"] == 0.30
    assert t.loc["max_drawdown", "meets_minimum"] is False


def test_go_no_go_on_a_compare_row():
    res, _ = random_case()
    c = compare({"s": res}, benchmark="s")
    t = go_no_go(c.loc["s"]).set_index("metric")
    assert t.loc["sharpe", "value"] == c.loc["s", "sharpe_after_tax"]
    assert t.loc["max_drawdown", "value"] == abs(c.loc["s", "max_drawdown_after_tax"])
    assert t.loc["excess_cagr_after_tax", "value"] == 0.0
    assert t.loc["tax_alpha", "meets_minimum"] is None


# --- no lookahead ------------------------------------------------------------------------


def _slice_result(res: SimulationResult, last: pd.Timestamp, liquidation_tax: float) -> SimulationResult:
    t = res.trades
    return SimulationResult(
        daily=res.daily.loc[:last],
        trades=t[pd.to_datetime(t["date"]) <= last],
        realisations=[r for r in res.realisations if pd.Timestamp(r.sell_date) <= last],
        fiscal_years={},
        stale_marks=res.stale_marks[pd.to_datetime(res.stale_marks["first_stale_date"]) <= last],
        liquidation_tax=liquidation_tax,
    )


@pytest.mark.parametrize("cut", [250, 400, 600])
def test_kpis_truncation_invariant(cut):
    res, (w, open_, close, divs) = random_case()
    s = res.daily.index[cut]
    trunc = simulate(
        w.loc[:s], open_.loc[:s], close.loc[:s], divs.loc[:s], initial_capital=C
    )
    # without the terminal liquidation term the truncated run is the full run's prefix
    sliced = _slice_result(res, s, trunc.liquidation_tax)
    assert_kpis_equal(kpis(trunc, after_tax=False), kpis(sliced, after_tax=False))
    assert_kpis_equal(kpis(trunc, after_tax=True), kpis(sliced, after_tax=True))
    # and with the liquidation term removed on both sides the equality is of every field
    no_liq = dataclasses.replace(trunc, liquidation_tax=0.0)
    sliced0 = _slice_result(res, s, 0.0)
    assert_kpis_equal(kpis(no_liq), kpis(sliced0))
    # the terminal-liquidation term is the only difference to the full run's sliced numbers
    full_prefix = kpis(_slice_result(res, s, 0.0))
    assert kpis(trunc)["terminal_value"] == pytest.approx(
        full_prefix["terminal_value"] - trunc.liquidation_tax, abs=1e-9
    )
    for k in ("sharpe", "sortino", "ann_vol", "max_drawdown", "var95", "cvar95", "total_costs"):
        assert _same(kpis(trunc)[k], full_prefix[k]), k


def test_inputs_not_mutated():
    res, _ = random_case()
    before = copy.deepcopy(res)
    kpis(res), kpis(res, after_tax=False), monthly_table(res), fy_tax_table(res)
    compare({"a": res}, benchmark="a")
    pd.testing.assert_frame_equal(res.daily, before.daily)
    pd.testing.assert_frame_equal(res.trades, before.trades)
    assert res.realisations == before.realisations
    assert res.fiscal_years.keys() == before.fiscal_years.keys()


def test_single_session_run_gives_nan_not_errors():
    res, _ = hand_case()
    one = dataclasses.replace(res, daily=res.daily.iloc[:1], liquidation_tax=0.0)
    k = kpis(one)
    assert k["years"] == 0.0
    for key in ("cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar", "var95",
                "turnover_annual"):
        assert math.isnan(k[key]), key
    with pytest.raises(ValueError):
        kpis(dataclasses.replace(res, daily=res.daily.iloc[:0]))


def test_concentration_wraps_herfindahl():
    w = pd.DataFrame({1: [0.5, 1.0], 2: [0.5, 0.0]}, index=pd.to_datetime(["2024-01-01", "2024-02-01"]))
    c = concentration(w)
    assert c.tolist() == [0.5, 1.0]
    pd.testing.assert_series_equal(c, performance.herfindahl(w))
