"""
Ruined portfolio in the backtest simulator (task 610): NAV <= RUIN_NAV is a flagged outcome
(``SimulationResult.ruined_on``), the portfolio is liquidated and never buys again, and the
summary functions cope with a curve that reaches 0 or goes negative. Synthetic fixtures only.
"""

import math
import warnings
from datetime import date

import numpy as np
import pandas as pd
import pytest

from investbot.backtest import simulator, summary
from investbot.backtest.simulator import RUIN_NAV, SimulationResult, daily_returns, simulate
from investbot.trading.tax_rules import regime_for
from investbot.trading.transaction_costs import estimate_costs


def frames(dates, closes: dict[int, list[float]], opens: dict[int, list[float]] | None = None):
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    close = pd.DataFrame(closes, index=idx, dtype=float)
    open_ = pd.DataFrame(opens if opens is not None else closes, index=idx, dtype=float)
    return open_, close


def schedule(rows: list[list[float]], idx, assets=(1, 2, 3)) -> pd.DataFrame:
    return pd.DataFrame(rows, index=idx[: len(rows)], columns=list(assets), dtype=float)


def held_after(res: SimulationResult, d) -> dict[int, float]:
    """Quantity per asset after every trade dated <= d."""
    t = res.trades[pd.to_datetime(res.trades["date"]) <= pd.Timestamp(d)]
    signed = np.where(t["side"] == "BUY", t["quantity"], -t["quantity"])
    return pd.Series(signed, index=t["asset_id"]).groupby(level=0).sum().to_dict()


def buys_after(res: SimulationResult, d) -> pd.DataFrame:
    t = res.trades
    return t[(t["side"] == "BUY") & (pd.to_datetime(t["date"]) > pd.Timestamp(d))]


# --- fixtures ------------------------------------------------------------------------------

def rotation_case(tax: bool = True):
    """₹180, three assets at a flat ₹10, all-in in one asset, rotating every session.

    Every session sells the whole holding and buys the next asset with all the cash, so each
    session pays two flat brokerages (₹20 + GST) and the NAV falls by ~₹47.4 a session until
    a holding worth less than one sell leg's cost is sold: cash < 0 with nothing held.
    """
    dates = pd.bdate_range("2025-01-01", periods=12)
    flat = [10.0] * len(dates)
    open_, close = frames(dates, {1: flat, 2: flat, 3: flat})
    rot = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] * 4
    return simulate(schedule(rot, dates), open_, close, initial_capital=180.0, tax=tax), dates


def drift_inputs():
    """₹200 equal-weight across three assets, rebalanced daily; assets 1 and 2 oscillate.

    Every rebalance trims the overweight legs at a flat ₹20 brokerage each, and the buys of
    the underweight legs are scaled to 0 once the cash cannot pay their brokerage, so the NAV
    bleeds to <= 0 at the close of 2025-01-07. Asset 3 has no valid open on 2025-01-08.
    """
    dates = pd.bdate_range("2025-01-01", periods=8)
    closes = {1: [10, 12, 10, 12, 10, 12, 10, 12], 2: [10, 8, 10, 8, 10, 8, 10, 8], 3: [10] * 8}
    opens = {k: list(v) for k, v in closes.items()}
    opens[3][5] = np.nan
    open_, close = frames(dates, closes, opens)
    return schedule([[1 / 3] * 3] * len(dates), dates), open_, close


def gap_inputs():
    """Same as ``drift_inputs`` but every open gaps to ₹1 on 2025-01-06 (closes unchanged)."""
    dates = pd.bdate_range("2025-01-01", periods=8)
    closes = {1: [10, 12, 10, 12, 10, 12, 10, 12], 2: [10, 8, 10, 8, 10, 8, 10, 8], 3: [10] * 8}
    opens = {k: list(v) for k, v in closes.items()}
    for k in opens:
        opens[k][3] = 1.0
    open_, close = frames(dates, closes, opens)
    return schedule([[1 / 3] * 3] * len(dates), dates), open_, close


# --- ruin date, liquidation, no buys -------------------------------------------------------

@pytest.mark.parametrize("tax", [True, False])
def test_rotation_ruin_hand_derived(tax):
    res, dates = rotation_case(tax)
    reg = regime_for("equity", date(2025, 1, 2))
    b = estimate_costs("BUY", 0.0, reg).total
    a = estimate_costs("BUY", 1.0, reg).total - b

    # The recurrence by hand: sell the held value v, then buy with the cash left (scaled so
    # buy value + its costs fit in the cash), while the cash can pay a buy's flat costs.
    cash, held, navs, ruined = 180.0, 0.0, [180.0], None
    for k in range(1, len(dates)):
        if ruined is not None:
            navs.append(cash)
            continue
        if held:
            cash += held - estimate_costs("SELL", held, reg).total
            held = 0.0
        if cash > b:
            held = (cash - b) / (1.0 + a) * (1.0 - 1e-12)
            cash -= held + estimate_costs("BUY", held, reg).total
        navs.append(cash + held)
        if cash + held <= 0:
            ruined = dates[k].date()
    # 180 -> ~156.2 held -> ~108.7 -> ~61.3 -> ~14.1, and selling ₹14.1 costs ~₹23.6
    assert ruined == date(2025, 1, 8)
    assert res.ruined_on == ruined
    assert res.daily["nav"].tolist() == pytest.approx(navs, abs=1e-9)
    assert res.daily["nav"].iloc[-1] < 0                  # written to the last session, negative
    assert len(res.daily) == len(dates)
    assert buys_after(res, res.ruined_on).empty            # the schedule keeps rotating: ignored
    assert not (pd.to_datetime(res.trades["date"]) > pd.Timestamp(res.ruined_on)).any()


@pytest.mark.parametrize("tax", [True, False])
def test_ruin_at_close_liquidates_next_open_and_waits_for_a_valid_open(tax):
    w, open_, close = drift_inputs()
    res = simulate(w, open_, close, initial_capital=200.0, tax=tax)
    nav = res.daily["nav"]
    # first close <= 0 is 2025-01-07; every earlier close and rebalance open is positive
    assert (nav.loc[:"2025-01-06"] > RUIN_NAV).all() and nav.loc["2025-01-07"] <= RUIN_NAV
    assert res.ruined_on == date(2025, 1, 7)
    assert buys_after(res, res.ruined_on).empty
    # 01-08 is the first open after the ruin: assets 1 and 2 are sold in full there; asset 3
    # has no valid open that day and is sold in full on 01-09.
    before = held_after(res, "2025-01-07")
    t = res.trades.assign(date=pd.to_datetime(res.trades["date"]))
    day8 = t[t["date"] == "2025-01-08"]
    assert set(day8["asset_id"]) == {1, 2} and (day8["side"] == "SELL").all()
    assert day8.set_index("asset_id")["quantity"].to_dict() == pytest.approx(
        {1: before[1], 2: before[2]}, rel=1e-12)
    day9 = t[t["date"] == "2025-01-09"]
    assert day9["asset_id"].tolist() == [3] and day9["side"].tolist() == ["SELL"]
    assert day9["quantity"].iloc[0] == pytest.approx(before[3], rel=1e-12)
    assert held_after(res, "2025-01-08")[3] == pytest.approx(before[3], rel=1e-12)
    for q in held_after(res, "2025-01-09").values():
        assert q == pytest.approx(0.0, abs=1e-12)
    assert (res.daily.loc["2025-01-09":, "holdings_value"] == 0).all()
    assert (res.daily.loc["2025-01-09":, "costs"].iloc[1:] == 0).all()   # nothing left to trade
    assert len(res.daily) == len(open_) and res.daily["nav"].iloc[-1] < 0


@pytest.mark.parametrize("tax", [True, False])
def test_ruin_at_a_rebalance_open_sells_at_that_open(tax):
    w, open_, close = gap_inputs()
    res = simulate(w, open_, close, initial_capital=200.0, tax=tax)
    d3 = res.daily.loc["2025-01-03"]
    assert d3["nav"] > RUIN_NAV                              # the previous close was fine
    held = held_after(res, "2025-01-03")
    nav_open = d3["cash"] + sum(q * 1.0 for q in held.values())   # every open is ₹1 on 01-06
    assert d3["cash"] < 0 and nav_open <= RUIN_NAV
    assert res.ruined_on == date(2025, 1, 6)
    t = res.trades.assign(date=pd.to_datetime(res.trades["date"]))
    day6 = t[t["date"] == "2025-01-06"]
    assert (day6["side"] == "SELL").all() and (day6["price"] == 1.0).all()
    assert day6.set_index("asset_id")["quantity"].to_dict() == pytest.approx(held, rel=1e-12)
    assert (t["date"] > "2025-01-06").sum() == 0
    assert (res.daily.loc["2025-01-06":, "holdings_value"] == 0).all()


def test_tax_still_paid_after_ruin():
    """Gain realised in FY 2024-25, ruin in March 2025, the FY tax is paid on 1 April 2025.

    Asset 1 is bought at ₹10 and sold at ₹40 (STCG). Asset 2 then takes 99% at ₹10 and closes
    at ₹0.001; dropping the 1% asset-3 leg costs more than it raises, so the close of 03-27
    is <= 0. Asset 2 has no valid open until 04-01, so it is sold only then, in FY 2025-26:
    its loss cannot offset the FY 2024-25 gain, whose tax is paid on 04-01 from cash.
    """
    dates = pd.bdate_range("2025-03-24", periods=10)       # 03-24 .. 04-04
    n = len(dates)
    c1 = [10, 40] + [40] * (n - 2)
    c2 = [10, 10, 0.001] + [0.001] * (n - 3)
    c3 = [10] * n
    o2 = [10, 10, 10] + [np.nan] * 3 + [0.001] * (n - 6)    # no open 03-27, 03-28, 03-31
    open_, close = frames(dates, {1: c1, 2: c2, 3: c3},
                          {1: [10, 10] + [40] * (n - 2), 2: o2, 3: c3})
    w = schedule([[1, 0, 0], [0, 0.99, 0.01], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
                  [0, 1, 0], [0, 1, 0]], dates)
    res = simulate(w, open_, close, initial_capital=200.0, tax=True)
    assert res.ruined_on == date(2025, 3, 27)
    fy = res.fiscal_years["2024-25"]
    assert fy.realized_stcg > 0 and fy.estimated_tax_liability > 0
    paid = res.daily["tax_paid"]
    assert paid.loc["2025-04-01"] == pytest.approx(fy.estimated_tax_liability, abs=1e-9)
    assert paid.drop(pd.Timestamp("2025-04-01")).eq(0).all()
    t = res.trades.assign(date=pd.to_datetime(res.trades["date"]))
    sale = t[(t["asset_id"] == 2) & (t["side"] == "SELL")]
    assert sale["date"].tolist() == [pd.Timestamp("2025-04-01")]
    assert buys_after(res, res.ruined_on).empty
    assert res.daily.loc["2025-04-01", "cash"] < 0
    naive = simulate(w, open_, close, initial_capital=200.0, tax=False)
    assert naive.ruined_on == date(2025, 3, 27) and naive.daily["tax_paid"].eq(0).all()


def test_sell_fifo_never_asked_for_more_than_is_held(monkeypatch):
    calls = []
    real = simulator.sell_fifo

    def checked(lots, quantity, *args, **kwargs):
        held = sum(lot.remaining_qty for lot in lots)
        calls.append((quantity, held))
        assert 0 < quantity <= held
        return real(lots, quantity, *args, **kwargs)

    monkeypatch.setattr(simulator, "sell_fifo", checked)
    for inputs in (drift_inputs(), gap_inputs()):
        for tax in (True, False):
            simulate(*inputs, initial_capital=200.0, tax=tax)
    rotation_case(True)
    assert calls


# --- no lookahead --------------------------------------------------------------------------

@pytest.mark.parametrize("make", [drift_inputs, gap_inputs])
@pytest.mark.parametrize("tax", [True, False])
def test_truncation_invariance_with_ruin(make, tax):
    w, open_, close = make()
    full = simulate(w, open_, close, initial_capital=200.0, tax=tax)
    assert full.ruined_on is not None
    for s in open_.index[:-1]:
        cut = simulate(w.loc[:s], open_.loc[:s], close.loc[:s], initial_capital=200.0, tax=tax)
        pd.testing.assert_frame_equal(cut.daily, full.daily.loc[:s])
        n = len(cut.trades)
        if n:
            pd.testing.assert_frame_equal(cut.trades, full.trades.iloc[:n])
        assert (pd.to_datetime(full.trades["date"].iloc[n:]) > s).all()
        if s.date() >= full.ruined_on:
            assert cut.ruined_on == full.ruined_on
        else:
            assert cut.ruined_on is None


# --- no ruin: unchanged --------------------------------------------------------------------

def test_no_ruin_is_bit_identical_to_the_pre_ruin_simulator():
    """The hand-checked fixture of test_backtest_simulator, exact values of the task-600 code."""
    dates = pd.bdate_range("2025-01-01", periods=10)
    open_, close = frames(
        dates,
        {1: [100, 101, 102, 103, 104, 108, 109, 110, 111, 112],
         2: [200, 202, 204, 206, 208, 210, 212, 214, 216, 218]},
        {1: [100, 100, 102, 103, 104, 110, 109, 110, 111, 112],
         2: [200, 200, 204, 206, 208, 205, 212, 214, 216, 218]},
    )
    w = pd.DataFrame({1: [0.4, 0.0], 2: [0.4, 0.8]}, index=dates[[0, 4]])
    res = simulate(w, open_, close, initial_capital=500_000.0)
    assert res.ruined_on is None
    assert res.daily["nav"].tolist() == [
        500000.0, 503476.988, 507476.988, 511476.988, 515476.988, 534179.5374876007,
        538273.0164183325, 542366.4953490642, 546459.9742797959, 550553.4532105275,
    ]
    assert res.daily["cash"].tolist() == [500000.0] + [99476.98800000001] * 4 + [
        104364.24976077146] * 5
    assert res.daily["tax_accrued"].tolist() == [0.0] * 5 + [4140.489059200001] * 5
    assert res.trades["quantity"].tolist() == [2000.0, 1000.0, 2000.0, 1046.739465365854]
    assert res.trades["costs"].tolist() == [261.506, 261.506, 252.2966, 278.85123922851204]
    assert res.liquidation_tax == 6539.644704676764


def test_daily_returns_nan_off_a_non_positive_base():
    w, open_, close = drift_inputs()
    res = simulate(w, open_, close, initial_capital=200.0)
    r = daily_returns(res, after_tax=False)
    nav = res.daily["nav"]
    assert r.loc["2025-01-07"] == pytest.approx(nav.loc["2025-01-07"] / nav.loc["2025-01-06"] - 1)
    assert r.loc["2025-01-07"] <= -1.0
    assert r.loc["2025-01-08":].isna().all()
    assert r.loc[:"2025-01-06"].iloc[1:].notna().all()


# --- summary on a ruined result ------------------------------------------------------------

def _zero_curve_result() -> SimulationResult:
    """A hand-built curve that hits exactly 0 and stays there."""
    idx = pd.bdate_range("2025-01-01", periods=6)
    nav = np.array([100.0, 90.0, 50.0, 0.0, 0.0, 0.0])
    zeros = np.zeros(len(idx))
    daily = pd.DataFrame({"cash": nav, "holdings_value": zeros, "nav": nav, "costs": zeros,
                          "dividends": zeros, "tax_paid": zeros, "tax_accrued": zeros,
                          "nav_after_tax": nav}, index=idx, columns=simulator.DAILY_COLUMNS)
    trades = pd.DataFrame(columns=simulator.TRADE_COLUMNS)
    return SimulationResult(daily=daily, trades=trades, ruined_on=date(2025, 1, 6))


def test_summary_functions_on_a_negative_curve():
    w, open_, close = drift_inputs()
    ruined = simulate(w, open_, close, initial_capital=200.0)
    fine = simulate(w.iloc[:1], open_, close, initial_capital=200.0)   # one rebalance, held
    assert fine.ruined_on is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for after in (True, False):
            k = summary.kpis(ruined, after_tax=after)
            assert math.isnan(k["cagr"]) and k["terminal_value"] < 0
            assert k["max_drawdown"] < -1.0
            assert math.isnan(k["calmar"])
            assert math.isfinite(k["sharpe"]) and math.isfinite(k["sortino"])
            summary.monthly_table(ruined, after_tax=after)
        fy = summary.fy_tax_table(ruined)
        c = summary.compare({"ruined": ruined, "fine": fine}, benchmark="fine")
        g = summary.go_no_go(c.loc["ruined"])
    assert list(fy.index) == ["2024-25"]
    assert c.columns[-1] == "ruined_on"
    assert c.loc["ruined", "ruined_on"] == "2025-01-07"
    assert c.loc["fine", "ruined_on"] is None
    assert math.isnan(c.loc["ruined", "excess_cagr_after_tax"])
    assert g.set_index("metric").loc["max_drawdown", "meets_minimum"] is False
    assert g.set_index("metric").loc["excess_cagr_after_tax", "meets_minimum"] is None


def test_summary_functions_on_a_curve_that_reaches_zero():
    res = _zero_curve_result()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        k = summary.kpis(res)
        m = summary.monthly_table(res)
        c = summary.compare({"z": res})
        summary.go_no_go(c.loc["z"])
        summary.fy_tax_table(res)
    assert k["max_drawdown"] == -1.0
    assert k["cagr"] == -1.0
    assert k["calmar"] == -1.0
    assert m.loc[2025, 1] == pytest.approx(-1.0)
    assert c.loc["z", "ruined_on"] == "2025-01-06"
