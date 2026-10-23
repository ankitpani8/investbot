"""
Dividend-income tax in the backtest simulator (task 620). Synthetic fixtures only.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from investbot.backtest.simulator import simulate
from investbot.trading.dividend_tax import DIVIDEND_SLAB_RATE
from investbot.trading.tax_rules import regime_for
from investbot.trading.transaction_costs import estimate_costs
from tests.test_backtest_simulator import _random_fixture, frames, weights

C = 500_000.0


def _one_dividend(start, end, ex_date, dps, w_rows, price=100.0, capital=C):
    dates = pd.bdate_range(start, end)
    open_, close = frames(dates, {1: [price] * len(dates)})
    divs = pd.DataFrame(0.0, index=dates, columns=[1])
    divs.loc[ex_date, 1] = dps
    return weights(w_rows, [1]), open_, close, divs


def test_fy_2021_22_dividend_accrued_on_ex_date_and_paid_first_april_session():
    # Buy 0.5 x C at 100 on 2022-01-03 (flat prices, no sale, so no capital-gains tax).
    # 4.0 per share on 2022-02-01 -> gross 4 x q; FY 2021-22 slab 30% + 4% cess.
    w, open_, close, divs = _one_dividend(
        "2021-12-31", "2022-04-08", "2022-02-01", 4.0, {"2021-12-31": {1: 0.5}}
    )
    res = simulate(w, open_, close, divs, initial_capital=C)
    d = res.daily
    q = res.trades["quantity"].iloc[0]
    assert q == pytest.approx(2_500.0)
    gross = 4.0 * q
    dtax = gross * 0.30 * 1.04  # 3,120.0 for q = 2,500
    assert dtax == pytest.approx(3_120.0, abs=1e-9)

    ex = pd.Timestamp("2022-02-01")
    before, after = d.loc[: ex - pd.Timedelta(days=1)], d.loc[ex:"2022-03-31"]
    assert (before["tax_accrued"] == 0.0).all()
    assert d.loc[ex, "tax_accrued"] - d["tax_accrued"].shift(1).loc[ex] == pytest.approx(dtax, abs=1e-9)
    assert after["tax_accrued"].tolist() == pytest.approx([dtax] * len(after), abs=1e-9)
    assert (d["nav_after_tax"] == d["nav"] - d["tax_accrued"]).all()
    assert d.loc[ex, "dividends"] == pytest.approx(gross, abs=1e-9)

    # paid on Friday 2022-04-01, the first session of FY 2022-23, from cash (no sale)
    apr = pd.Timestamp("2022-04-01")
    assert d["tax_paid"][d["tax_paid"] != 0].index.tolist() == [apr]
    assert d.loc[apr, "tax_paid"] == pytest.approx(dtax, abs=1e-9)
    assert d.loc[apr, "cash"] == pytest.approx(d.loc["2022-03-31", "cash"] - dtax, abs=1e-9)
    assert (d.loc[apr:, "tax_accrued"] == 0.0).all()
    assert len(res.trades) == 1
    assert d["nav_after_tax"].iloc[-1] == pytest.approx(C - res.trades["costs"].iloc[0] + gross - dtax,
                                                        abs=1e-6)

    assert res.dividend_tax["2021-22"] == pytest.approx((gross, dtax))
    assert res.dividend_tax["2022-23"] == (0.0, 0.0)
    assert list(res.dividend_tax) == list(res.fiscal_years)
    assert res.fiscal_years["2021-22"].estimated_tax_liability == 0.0  # kept apart
    assert res.liquidation_tax == 0.0


def test_dividend_tax_paid_with_pro_rata_cover_sale_when_cash_is_short():
    # Fully invested again after the dividend (0.999 x NAV), so April cash cannot cover it.
    w, open_, close, divs = _one_dividend(
        "2021-12-31", "2022-04-08", "2022-02-01", 4.0,
        {"2021-12-31": {1: 0.5}, "2022-02-10": {1: 0.999}},
    )
    res = simulate(w, open_, close, divs, initial_capital=C)
    d, t = res.daily, res.trades
    apr = pd.Timestamp("2022-04-01")
    dtax = res.dividend_tax["2021-22"][1]
    assert dtax > d.loc["2022-03-31", "cash"] > 0
    (sale,) = t[(t["date"] == date(2022, 4, 1))].itertuples()
    assert sale.side == "SELL"
    shortfall = dtax - d.loc["2022-03-31", "cash"]
    net = sale.value - estimate_costs("SELL", sale.value, regime_for("equity", date(2022, 4, 1))).total
    assert net == pytest.approx(shortfall, rel=1e-9)
    assert d.loc[apr, "tax_paid"] == pytest.approx(dtax)
    assert d.loc[apr, "cash"] == pytest.approx(0.0, abs=1e-6)


def test_fy_2018_19_dividend_below_ten_lakh_untaxed():
    w, open_, close, divs = _one_dividend(
        "2018-06-01", "2019-04-05", "2018-09-03", 10.0, {"2018-06-01": {1: 0.5}}
    )
    res = simulate(w, open_, close, divs, initial_capital=C)
    gross, dtax = res.dividend_tax["2018-19"]
    assert gross == pytest.approx(25_000.0) and dtax == 0.0
    assert (res.daily["tax_accrued"] == 0.0).all() and (res.daily["tax_paid"] == 0.0).all()


def test_fy_2018_19_dividend_above_ten_lakh_s115bbda():
    # 1e8 capital, 0.5 invested at 100 -> 500,000 shares x 3.0 = 15 lakh gross;
    # 10% on the 5 lakh above the threshold, 4% cess (FY 2018-19) = 52,000.
    w, open_, close, divs = _one_dividend(
        "2018-06-01", "2019-04-05", "2018-09-03", 3.0, {"2018-06-01": {1: 0.5}}, capital=1e8
    )
    res = simulate(w, open_, close, divs, initial_capital=1e8)
    gross, dtax = res.dividend_tax["2018-19"]
    assert gross == pytest.approx(1_500_000.0)
    assert dtax == pytest.approx(52_000.0)
    assert res.daily.loc["2019-04-01", "tax_paid"] == pytest.approx(52_000.0)


def test_open_fy_dividend_tax_is_terminal_tax_accrued_and_respects_slab_and_surcharge():
    w, open_, close, divs = _one_dividend(
        "2023-06-01", "2023-12-29", "2023-09-01", 2.0, {"2023-06-01": {1: 0.5}}
    )
    res = simulate(w, open_, close, divs, initial_capital=C, dividend_slab_rate=0.20,
                   surcharge_rate=0.10)
    gross = 2.0 * res.trades["quantity"].iloc[0]
    dtax = gross * 0.20 * 1.10 * 1.04
    assert res.dividend_tax == {"2023-24": pytest.approx((gross, dtax))}
    assert res.daily["tax_accrued"].iloc[-1] == pytest.approx(dtax)
    assert (res.daily["tax_paid"] == 0.0).all()
    assert res.liquidation_tax == 0.0


def test_tax_false_never_taxes_dividends():
    w, open_, close, divs = _random_fixture()
    res = simulate(w, open_, close, divs, tax=False)
    assert res.dividend_tax == {} and res.fiscal_years == {}
    assert (res.daily["tax_accrued"] == 0.0).all() and (res.daily["tax_paid"] == 0.0).all()
    none = simulate(w, open_, close, divs, tax=False, dividend_slab_rate=None)
    pd.testing.assert_frame_equal(res.daily, none.daily, check_exact=True)
    pd.testing.assert_frame_equal(res.trades, none.trades, check_exact=True)


def test_none_and_tax_false_bit_identical_to_pre_task_outputs():
    # Pinned from the simulator at the parent commit of task 620 (dividends untaxed) on
    # _random_fixture() (seed 3, 320 sessions, two dividends).
    w, open_, close, divs = _random_fixture()
    pinned = {
        (True, None): (648360.6693067154, 623168.4671842011, 10105.281141153631,
                       3184538.017924468, 7471.852397998377, 748.0715167908638),
        (False, DIVIDEND_SLAB_RATE): (660583.4254076559, 660583.4254076559, 0.0, 0.0,
                                      7536.106842884967, 0.0),
    }
    for (tax, rate), want in pinned.items():
        r = simulate(w, open_, close, divs, tax=tax, dividend_slab_rate=rate)
        d = r.daily
        got = (float(d["nav"].iloc[-1]), float(d["nav_after_tax"].iloc[-1]),
               float(d["tax_paid"].sum()), float(d["tax_accrued"].sum()),
               float(d["dividends"].sum()), r.liquidation_tax)
        assert got == want
        assert len(r.trades) == 256
    untaxed = simulate(w, open_, close, divs, dividend_slab_rate=None)
    assert all(t == 0.0 for _, t in untaxed.dividend_tax.values())
    assert sum(g for g, _ in untaxed.dividend_tax.values()) == pytest.approx(
        untaxed.daily["dividends"].sum()
    )


def test_truncation_invariance_with_dividend_tax():
    # dividends in FY 2024-25 and FY 2025-26; cut on, just before and after each ex-date and
    # around the April payment
    w, open_, close, divs = _random_fixture()
    full = simulate(w, open_, close, divs)
    ex = [i for i in range(len(divs)) if divs.iloc[i].any()]
    april = int(np.flatnonzero(full.daily["tax_paid"].to_numpy() > 0)[0])
    for i in (ex[0] - 1, ex[0], ex[0] + 1, april - 1, april, ex[1], len(divs) - 2):
        T = close.index[i]
        trunc = simulate(w[w.index <= T], open_[:T], close[:T], divs[:T])
        pd.testing.assert_frame_equal(trunc.daily, full.daily.loc[:T], rtol=0, atol=1e-9)
        # a dividend after T never changes the rows up to T
        later = divs.copy()
        later.loc[later.index > T] *= 5.0
        bumped = simulate(w, open_, close, later)
        pd.testing.assert_frame_equal(bumped.daily.loc[:T], full.daily.loc[:T], rtol=0, atol=1e-9)
