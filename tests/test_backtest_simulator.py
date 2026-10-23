"""
After-tax portfolio simulator (investbot/backtest/simulator.py). Synthetic fixtures only.
"""

import copy
import inspect
from datetime import date

import numpy as np
import pandas as pd
import pytest

from investbot.backtest import simulator
from investbot.backtest.simulator import DAILY_COLUMNS, daily_returns, simulate
from investbot.evaluation import performance
from investbot.trading.lot_tracker import Lot, sell_fifo
from investbot.trading.tax_ledger import compute_fiscal_year_tax
from investbot.trading.tax_rules import regime_for
from investbot.trading.transaction_costs import deductible_costs, estimate_costs

C = 500_000.0


def frames(dates, prices: dict[int, list[float]], opens: dict[int, list[float]] | None = None):
    """open_ and close frames; opens default to the closes (flat intraday)."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    close = pd.DataFrame(prices, index=idx, dtype=float)
    open_ = pd.DataFrame(opens if opens is not None else prices, index=idx, dtype=float)
    return open_, close


def weights(rows: dict[str, dict[int, float]], assets) -> pd.DataFrame:
    w = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=assets)
    w.index = pd.to_datetime(w.index)
    return w


# --- hand-checked fixture ---------------------------------------------------------------


def test_hand_checked_two_assets_one_rebalance():
    dates = pd.bdate_range("2025-01-01", periods=10)
    a_close = [100, 101, 102, 103, 104, 108, 109, 110, 111, 112]
    a_open = [100, 100, 102, 103, 104, 110, 109, 110, 111, 112]
    b_close = [200, 202, 204, 206, 208, 210, 212, 214, 216, 218]
    b_open = [200, 200, 204, 206, 208, 205, 212, 214, 216, 218]
    open_, close = frames(dates, {1: a_close, 2: b_close}, {1: a_open, 2: b_open})
    w = weights({str(dates[0].date()): {1: 0.4, 2: 0.4}, str(dates[4].date()): {1: 0.0, 2: 0.8}}, [1, 2])
    res = simulate(w, open_, close, initial_capital=C)
    reg = regime_for("equity", date(2025, 1, 2))

    # Session 1 (2025-01-02): buy 200,000 of each at the open.
    buy_cost = estimate_costs("BUY", 200_000.0, reg).total
    assert buy_cost == pytest.approx(261.506, abs=1e-9)  # 200 + 30 + 6.5 + 0.2 + 20 + 4.806
    qa, qb = 200_000 / 100, 200_000 / 200
    cash1 = C - 400_000 - 2 * buy_cost
    d = res.daily
    assert d["cash"].iloc[0] == pytest.approx(C, abs=1e-6)
    assert d["cash"].iloc[1] == pytest.approx(cash1, abs=1e-6)
    assert d["costs"].iloc[1] == pytest.approx(2 * buy_cost, abs=1e-6)
    assert d["nav"].iloc[1] == pytest.approx(cash1 + qa * 101 + qb * 202, abs=1e-6)

    # Session 5 (2025-01-08): sell all of A at 110, top B up to 0.8 x NAV at 205.
    nav_open = cash1 + qa * 110 + qb * 205
    sell = estimate_costs("SELL", qa * 110, reg)
    cash5 = cash1 + qa * 110 - sell.total
    qb_target = 0.8 * nav_open / 205
    buy2 = estimate_costs("BUY", (qb_target - qb) * 205, reg).total
    cash5 -= (qb_target - qb) * 205 + buy2
    assert d["cash"].iloc[5] == pytest.approx(cash5, abs=1e-6)
    assert d["costs"].iloc[5] == pytest.approx(sell.total + buy2, abs=1e-6)
    assert d["nav"].iloc[5] == pytest.approx(cash5 + qb_target * 210, abs=1e-6)
    assert d["nav"].iloc[-1] == pytest.approx(cash5 + qb_target * 218, abs=1e-6)

    t = res.trades
    assert list(t["side"]) == ["BUY", "BUY", "SELL", "BUY"]
    assert list(t["asset_id"]) == [1, 2, 1, 2]
    assert t["quantity"].tolist() == pytest.approx([qa, qb, qa, qb_target - qb], abs=1e-9)
    assert t["costs"].tolist() == pytest.approx([buy_cost, buy_cost, sell.total, buy2], abs=1e-9)

    # One short-term realisation on A.
    (r,) = res.realisations
    basis = 200_000 + deductible_costs(estimate_costs("BUY", 200_000.0, reg))
    gain = qa * 110 - basis - deductible_costs(sell)
    assert r.term == "STCG" and r.asset_id == 1 and r.quantity == pytest.approx(qa)
    assert r.cost_basis == pytest.approx(basis, abs=1e-6)
    assert r.gain == pytest.approx(gain, abs=1e-6)
    tax = gain * 0.20 * 1.04
    assert d["tax_accrued"].iloc[4] == 0.0
    assert d["tax_accrued"].iloc[5:].tolist() == pytest.approx([tax] * 5, abs=1e-6)
    assert (d["nav_after_tax"] == d["nav"] - d["tax_accrued"]).all()
    assert (d["tax_paid"] == 0).all()
    assert list(d.columns) == DAILY_COLUMNS

    # Terminal liquidation of B at 218: B was bought on two dates, both short term.
    assert res.liquidation_tax > 0
    assert set(res.fiscal_years) == {"2024-25"}
    assert res.fiscal_years["2024-25"].estimated_tax_liability == pytest.approx(tax, abs=1e-6)


def test_buys_scaled_so_cash_stays_non_negative():
    dates = pd.bdate_range("2025-01-01", periods=5)
    open_, close = frames(dates, {1: [100] * 5, 2: [50] * 5})
    w = weights({"2025-01-01": {1: 0.5, 2: 0.5}}, [1, 2])
    res = simulate(w, open_, close, initial_capital=C)
    reg = regime_for("equity", date(2025, 1, 2))
    a = estimate_costs("BUY", 1.0, reg).total - estimate_costs("BUY", 0.0, reg).total
    b = estimate_costs("BUY", 0.0, reg).total
    k = (C - 2 * b) / (C * (1 + a))
    assert res.daily["cash"].iloc[1] == pytest.approx(0.0, abs=1e-6)
    assert res.daily["cash"].min() >= 0.0
    assert res.trades["value"].tolist() == pytest.approx([k * C / 2] * 2, abs=1e-6)
    assert res.trades["costs"].tolist() == pytest.approx(
        [estimate_costs("BUY", k * C / 2, reg).total] * 2, abs=1e-9
    )


def test_inputs_not_mutated():
    dates = pd.bdate_range("2025-01-01", periods=6)
    open_, close = frames(dates, {1: [100, np.nan, 102, 103, 104, 105]})
    divs = pd.DataFrame({1: [0, 0, 1.0, np.nan, 0, 0]}, index=open_.index, dtype=float)
    w = weights({"2025-01-01": {1: 0.5}, "2025-01-03": {1: np.nan}}, [1])
    before = [x.copy() for x in (w, open_, close, divs)]
    fmv = {1: 99.0}
    simulate(w, open_, close, divs, fmv_2018_01_31=fmv)
    for x, y in zip((w, open_, close, divs), before):
        pd.testing.assert_frame_equal(x, y)
    assert fmv == {1: 99.0}


# --- tax --------------------------------------------------------------------------------


def test_short_term_gain_paid_on_first_april_session():
    dates = ["2025-01-10", "2025-01-13", "2025-02-14", "2025-02-17", "2025-03-28", "2025-04-02", "2025-04-03"]
    open_, close = frames(dates, {1: [100, 100, 120, 120, 120, 120, 120]})
    w = weights({"2025-01-10": {1: 0.5}, "2025-02-14": {1: 0.0}}, [1])
    res = simulate(w, open_, close, initial_capital=C)
    fy = res.fiscal_years["2024-25"]
    assert fy.realized_stcg > 0
    d = res.daily
    assert d.loc["2025-04-02", "tax_paid"] == pytest.approx(fy.estimated_tax_liability, abs=1e-9)
    assert d["tax_paid"].sum() == pytest.approx(fy.estimated_tax_liability, abs=1e-9)
    # Accrued until 31 March, paid on 2 April: after-tax NAV does not jump at the payment.
    assert d.loc["2025-03-28", "tax_accrued"] == pytest.approx(fy.estimated_tax_liability)
    assert d.loc["2025-04-02", "tax_accrued"] == 0.0
    assert d.loc["2025-04-02", "nav_after_tax"] == pytest.approx(d.loc["2025-03-28", "nav_after_tax"])
    assert d.loc["2025-04-02", "cash"] == pytest.approx(
        d.loc["2025-03-28", "cash"] - fy.estimated_tax_liability
    )
    assert set(res.fiscal_years) == {"2024-25", "2025-26"}
    assert res.fiscal_years["2025-26"].estimated_tax_liability == 0.0


def test_loss_carried_forward_offsets_later_gain():
    dates = ["2025-01-10", "2025-01-13", "2025-02-17", "2025-04-02", "2025-05-02", "2025-06-02", "2025-06-03"]
    a = [100, 100, 80, 80, 80, 80, 80]
    b = [50, 50, 50, 50, 50, 70, 70]
    open_, close = frames(dates, {1: a, 2: b})
    w = weights(
        {"2025-01-10": {1: 0.5}, "2025-02-14": {1: 0.0}, "2025-04-02": {2: 0.4}, "2025-06-02": {2: 0.0}},
        [1, 2],
    )
    res = simulate(w, open_, close, initial_capital=C)
    fy1, fy2 = res.fiscal_years["2024-25"], res.fiscal_years["2025-26"]
    assert fy1.estimated_tax_liability == 0.0
    loss = fy1.cf_stcl_by_year["2024-25"]
    assert loss == pytest.approx(fy1.realized_stcl) and loss > 0
    gain = fy2.realized_stcg
    assert gain > loss
    assert fy2.estimated_tax_liability == pytest.approx((gain - loss) * 0.20 * 1.04, abs=1e-6)
    assert fy2.carried_forward_stcl == 0.0
    assert res.daily["tax_paid"].sum() == 0.0  # FY 2025-26 is still open at the end
    assert res.daily["tax_accrued"].iloc[-1] == pytest.approx(fy2.estimated_tax_liability)


def test_long_term_gain_uses_exemption():
    dates = ["2024-01-10", "2024-01-11", "2025-01-11", "2025-06-02", "2025-06-03"]
    open_, close = frames(dates, {1: [100, 100, 150, 200, 200]})
    w = weights({"2024-01-10": {1: 0.5}, "2025-05-30": {1: 0.0}}, [1])
    res = simulate(w, open_, close, initial_capital=C)
    (r,) = res.realisations
    # 2025-05-30 is not a session: handled like 2025-01-11, so it executes on 2025-06-02
    assert r.term == "LTCG" and r.sell_date == date(2025, 6, 2)
    fy = res.fiscal_years["2025-26"]
    assert fy.realized_ltcg == pytest.approx(r.gain)
    assert fy.ltcg_exemption_used == 125_000
    assert fy.estimated_tax_liability == pytest.approx((r.gain - 125_000) * 0.125 * 1.04, abs=1e-6)


def test_tax_shortfall_sells_pro_rata_in_new_fy():
    # Fully invested at the gain, so the April tax must be raised by selling.
    dates = ["2025-01-10", "2025-01-13", "2025-02-14", "2025-02-17", "2025-03-28", "2025-04-02", "2025-04-03"]
    a = [100, 100, 150, 150, 150, 150, 150]
    b = [100, 100, 100, 100, 100, 100, 100]
    open_, close = frames(dates, {1: a, 2: b})
    w = weights({"2025-01-10": {1: 1.0}, "2025-02-14": {2: 1.0}}, [1, 2])
    res = simulate(w, open_, close, initial_capital=C)
    due = res.fiscal_years["2024-25"].estimated_tax_liability
    d = res.daily
    assert due > 0 and d.loc["2025-03-28", "cash"] < due
    assert d.loc["2025-04-02", "tax_paid"] == pytest.approx(due)
    assert d.loc["2025-04-02", "cash"] == pytest.approx(0.0, abs=1e-6)
    assert (d["cash"] >= 0).all()
    sales = res.trades[(res.trades["date"] == date(2025, 4, 2))]
    assert list(sales["side"]) == ["SELL"] and list(sales["asset_id"]) == [2]
    # The tax sale realises a small loss in FY 2025-26 (sell costs on a flat price).
    new = [r for r in res.realisations if r.sell_date == date(2025, 4, 2)]
    assert new and all(r.gain <= 0 for r in new)


def test_grandfathered_lot_stepped_up_only_with_fmv():
    dates = ["2017-06-01", "2017-06-02", "2018-01-31", "2019-06-03", "2019-06-04"]
    open_, close = frames(dates, {7: [100, 100, 140, 150, 150]})
    w = weights({"2017-06-01": {7: 0.5}, "2019-06-03": {7: 0.0}}, [7])
    plain = simulate(w, open_, close, initial_capital=C)
    gf = simulate(w, open_, close, initial_capital=C, fmv_2018_01_31={7: 140.0})
    (rp,), (rg,) = plain.realisations, gf.realisations
    assert rp.term == rg.term == "LTCG"
    assert not rp.grandfathered and rg.grandfathered
    assert rg.cost_basis == pytest.approx(rg.quantity * 140.0)
    assert rp.cost_basis < rg.cost_basis
    assert rp.cost_basis == pytest.approx(rp.quantity * 100 + plain.trades["costs"].iloc[0] - 0.001 * rp.quantity * 100)
    assert gf.daily["tax_accrued"].iloc[-1] < plain.daily["tax_accrued"].iloc[-1]


def test_tax_false_is_tax_naive_with_same_costs():
    dates = ["2025-01-10", "2025-01-13", "2025-02-14", "2025-02-17", "2025-03-28", "2025-04-02", "2025-04-03"]
    open_, close = frames(dates, {1: [100, 100, 150, 150, 150, 150, 150], 2: [100] * 7})
    w = weights({"2025-01-10": {1: 1.0}, "2025-02-14": {2: 1.0}, "2025-04-02": {1: 0.5, 2: 0.5}}, [1, 2])
    taxed = simulate(w, open_, close)
    naive = simulate(w, open_, close, tax=False)
    assert (naive.daily["tax_paid"] == 0).all() and (naive.daily["tax_accrued"] == 0).all()
    assert naive.liquidation_tax == 0.0 and naive.fiscal_years == {}
    assert (naive.daily["nav_after_tax"] == naive.daily["nav"]).all()
    first_pay = taxed.daily.index[taxed.daily["tax_paid"] > 0][0]
    before = taxed.daily.index < first_pay
    np.testing.assert_allclose(naive.daily["costs"][before], taxed.daily["costs"][before], atol=1e-12)
    assert taxed.daily.loc[first_pay, "nav"] < naive.daily.loc[first_pay, "nav"]


# --- lookahead --------------------------------------------------------------------------


def _random_fixture(seed=3, n=320, n_assets=4):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-10-01", periods=n)
    rets = rng.normal(0.0004, 0.015, size=(n, n_assets))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=range(1, n_assets + 1))
    open_ = close.shift(1) * np.exp(rng.normal(0, 0.004, size=(n, n_assets)))
    open_.iloc[0] = close.iloc[0]
    divs = pd.DataFrame(0.0, index=dates, columns=close.columns)
    divs.iloc[n // 3, 0] = 2.0
    divs.iloc[3 * n // 4, 2] = 1.5
    weekly = dates[dates.weekday == 4]
    raw = rng.uniform(0, 1, size=(len(weekly), n_assets))
    w = pd.DataFrame(raw / raw.sum(axis=1, keepdims=True) * 0.95, index=weekly, columns=close.columns)
    return w, open_, close, divs


def test_truncation_invariance():
    w, open_, close, divs = _random_fixture()
    full = simulate(w, open_, close, divs)
    assert (full.daily["tax_paid"] > 0).any()  # the window crosses a paid FY boundary
    # both FYs have a dividend, so tax_accrued and tax_paid carry dividend tax (task 620)
    assert all(t > 0 for _, t in full.dividend_tax.values())
    # tax_accrued (compressed realisations) agrees with the full-list ledger
    last_fy = list(full.fiscal_years)[-1]
    assert full.daily["tax_accrued"].iloc[-1] == pytest.approx(
        full.fiscal_years[last_fy].estimated_tax_liability + full.dividend_tax[last_fy][1],
        rel=1e-12, abs=1e-6,
    )
    paid = full.daily["tax_paid"][full.daily["tax_paid"] > 0]
    assert paid.iloc[0] == pytest.approx(
        full.fiscal_years["2024-25"].estimated_tax_liability + full.dividend_tax["2024-25"][1]
    )
    for T in (close.index[40], close.index[127], close.index[128], close.index[300]):
        trunc = simulate(w[w.index <= T], open_[:T], close[:T], divs[:T])
        pd.testing.assert_frame_equal(trunc.daily, full.daily.loc[:T], rtol=0, atol=1e-9)


def test_changing_weight_row_t_leaves_rows_up_to_t_unchanged():
    w, open_, close, divs = _random_fixture()
    full = simulate(w, open_, close, divs)
    t = w.index[20]
    w2 = w.copy()
    w2.loc[t] = [0.9, 0.0, 0.0, 0.1]
    changed = simulate(w2, open_, close, divs)
    pd.testing.assert_frame_equal(changed.daily.loc[:t], full.daily.loc[:t], rtol=0, atol=1e-9)
    assert not np.allclose(changed.daily["nav"].iloc[-1], full.daily["nav"].iloc[-1])


def test_changing_prices_after_t_leaves_rows_up_to_t_unchanged():
    w, open_, close, divs = _random_fixture()
    full = simulate(w, open_, close, divs)
    t = close.index[150]
    o2, c2 = open_.copy(), close.copy()
    o2.loc[o2.index > t] *= 1.3
    c2.loc[c2.index > t] *= 0.7
    changed = simulate(w, o2, c2, divs)
    pd.testing.assert_frame_equal(changed.daily.loc[:t], full.daily.loc[:t], rtol=0, atol=1e-9)
    assert not np.allclose(changed.daily["nav"].iloc[-1], full.daily["nav"].iloc[-1])


def test_decision_on_non_session_executes_after_last_session_before_it():
    dates = pd.bdate_range("2025-01-01", periods=8)  # 2025-01-04/05 are a weekend
    open_, close = frames(dates, {1: [100] * 8})
    res = simulate(weights({"2025-01-04": {1: 0.5}}, [1]), open_, close)
    assert list(res.trades["date"]) == [date(2025, 1, 6)]
    # the same decision dated Friday 2025-01-03 executes at the same open
    fri = simulate(weights({"2025-01-03": {1: 0.5}}, [1]), open_, close)
    pd.testing.assert_frame_equal(res.daily, fri.daily)


# --- missing data, dividends, validation ------------------------------------------------


def test_nan_open_skips_only_that_leg():
    dates = pd.bdate_range("2025-01-01", periods=6)
    a_open = [100, np.nan, 100, 100, 100, 100]
    open_, close = frames(dates, {1: [100] * 6, 2: [50] * 6}, {1: a_open, 2: [50] * 6})
    w = weights({"2025-01-01": {1: 0.4, 2: 0.4}}, [1, 2])
    res = simulate(w, open_, close)
    t = res.trades
    assert list(t["asset_id"]) == [2] and t["date"].iloc[0] == date(2025, 1, 2)
    assert t["value"].iloc[0] == pytest.approx(0.4 * C)  # NAV at the open is all cash


def test_stale_holding_marked_at_last_close():
    dates = pd.bdate_range("2025-01-01", periods=8)
    a = [100, 100, 104, 105, np.nan, np.nan, np.nan, np.nan]
    open_, close = frames(dates, {1: a, 2: [50] * 8})
    w = weights({"2025-01-01": {1: 0.5, 2: 0.4}, "2025-01-06": {1: 0.5, 2: 0.3}}, [1, 2])
    res = simulate(w, open_, close)
    qa = res.trades.loc[res.trades["asset_id"] == 1, "quantity"].sum()
    assert res.daily["holdings_value"].iloc[-1] == pytest.approx(
        qa * 105 + res.trades.loc[res.trades["asset_id"] == 2, "quantity"].iloc[0] * 50
        - res.trades.loc[(res.trades["asset_id"] == 2) & (res.trades["side"] == "SELL"), "quantity"].sum() * 50
    )
    assert (res.trades["asset_id"] == 1).sum() == 1  # never traded again: no valid open
    sm = res.stale_marks
    assert len(sm) == 1
    assert sm.iloc[0].to_dict() == {
        "asset_id": 1,
        "first_stale_date": date(2025, 1, 7),
        "last_valid_date": date(2025, 1, 6),
        "last_valid_close": 105.0,
    }


def test_dividend_credited_on_previous_close_quantity():
    dates = pd.bdate_range("2025-01-01", periods=5)
    open_, close = frames(dates, {1: [100] * 5})
    divs = pd.DataFrame({1: [0, 0, 3.0, 0, 0]}, index=open_.index, dtype=float)
    w = weights({"2025-01-01": {1: 0.5}, "2025-01-02": {1: 0.0}}, [1])  # sold at the ex-date open
    res = simulate(w, open_, close, divs)
    q = res.trades["quantity"].iloc[0]
    assert res.daily["dividends"].tolist() == pytest.approx([0, 0, 3.0 * q, 0, 0])
    no_div = simulate(w, open_, close)
    assert res.daily["cash"].iloc[-1] - no_div.daily["cash"].iloc[-1] == pytest.approx(3.0 * q)


def test_first_execution_before_2004_10_01_raises():
    dates = ["2004-09-28", "2004-09-29", "2004-10-01"]
    open_, close = frames(dates, {1: [100, 100, 100]})
    with pytest.raises(ValueError, match="2004-10-01"):
        simulate(weights({"2004-09-28": {1: 0.5}}, [1]), open_, close)
    # a decision executing on 2004-10-01 is fine, earlier sessions just hold cash
    res = simulate(weights({"2004-09-29": {1: 0.5}}, [1]), open_, close)
    assert list(res.trades["date"]) == [date(2004, 10, 1)]


@pytest.mark.parametrize(
    "rows, msg",
    [
        ({"2025-01-01": {1: -0.1}}, ">= 0"),
        ({"2025-01-01": {1: 0.7, 2: 0.4}}, "sum"),
    ],
)
def test_bad_weights_raise(rows, msg):
    dates = pd.bdate_range("2025-01-01", periods=3)
    open_, close = frames(dates, {1: [100] * 3, 2: [100] * 3})
    with pytest.raises(ValueError, match=msg):
        simulate(weights(rows, [1, 2]), open_, close)
    with pytest.raises(ValueError, match="no prices"):
        simulate(weights({"2025-01-01": {9: 0.5}}, [9]), open_, close)


def test_weight_sum_tolerance_accepted():
    dates = pd.bdate_range("2025-01-01", periods=3)
    open_, close = frames(dates, {1: [100] * 3, 2: [100] * 3})
    simulate(weights({"2025-01-01": {1: 0.5, 2: 0.5 + 5e-10}}, [1, 2]), open_, close)


# --- horizons ---------------------------------------------------------------------------


@pytest.mark.parametrize("freq", ["daily", "weekly", "monthly"])
def test_schedules_of_any_horizon(freq):
    w_all, open_, close, divs = _random_fixture(seed=11, n=130)
    sessions = close.index
    if freq == "daily":
        dec = sessions
    elif freq == "weekly":
        dec = sessions[sessions.weekday == 4]
    else:
        dec = sessions.to_series().groupby(sessions.to_period("M")).max().values
    rng = np.random.default_rng(5)
    raw = rng.uniform(0, 1, size=(len(dec), 4))
    w = pd.DataFrame(raw / raw.sum(axis=1, keepdims=True), index=pd.DatetimeIndex(dec), columns=close.columns)
    res = simulate(w, open_, close, divs, min_trade_value=1_000.0)
    exec_dates = {sessions[sessions.searchsorted(t, side="right")].date() for t in w.index if t < sessions[-1]}
    assert set(res.trades["date"]) <= exec_dates
    assert len(set(res.trades["date"])) == len(exec_dates)
    assert res.daily["cash"].min() >= 0.0
    r = daily_returns(res)
    assert np.isnan(r.iloc[0]) and r.iloc[1:].notna().all()
    assert np.isfinite(performance.sharpe(r, periods_per_year=252))


def test_constant_weights_flat_prices_trade_only_once():
    dates = pd.bdate_range("2025-01-01", periods=30)
    open_, close = frames(dates, {1: [100] * 30, 2: [37.5] * 30, 3: [1234.0] * 30})
    w = pd.DataFrame({1: 0.2, 2: 0.3, 3: 0.5}, index=dates)
    res = simulate(w, open_, close)
    assert set(res.trades["date"]) == {date(2025, 1, 2)}
    assert len(res.trades) == 3


def test_daily_returns_pre_and_after_tax():
    w, open_, close, divs = _random_fixture()
    res = simulate(w, open_, close, divs)
    pre = daily_returns(res, after_tax=False)
    post = daily_returns(res)
    assert pre.index.equals(res.daily.index)
    assert np.isnan(pre.iloc[0]) and np.isnan(post.iloc[0])
    np.testing.assert_allclose(pre.iloc[1:], res.daily["nav"].pct_change().iloc[1:])
    assert not np.allclose(pre.iloc[1:], post.iloc[1:])


def test_liquidation_tax_matches_manual_sale():
    dates = ["2025-01-10", "2025-01-13", "2025-03-03"]
    open_, close = frames(dates, {1: [100, 100, 130]})
    res = simulate(weights({"2025-01-10": {1: 0.5}}, [1]), open_, close)
    reg = regime_for("equity", date(2025, 3, 3))
    (buy,) = res.trades.itertuples()
    lot = Lot("x", 1, date(2025, 1, 13), 100.0, buy.quantity, buy.quantity,
              buy_costs=deductible_costs(estimate_costs("BUY", buy.value, reg)))
    rs, _ = sell_fifo([lot], buy.quantity, 130.0, date(2025, 3, 3), reg,
                      sell_costs=estimate_costs("SELL", buy.quantity * 130, reg))
    expected = compute_fiscal_year_tax(rs, "2024-25").estimated_tax_liability
    assert res.liquidation_tax == pytest.approx(expected, abs=1e-6)
    assert res.daily["tax_accrued"].iloc[-1] == 0.0


def test_module_is_pure():
    src = inspect.getsource(simulator)
    for banned in ("dropna(", "sqlalchemy", "investbot.data", "yfinance", "requests"):
        assert banned not in src
