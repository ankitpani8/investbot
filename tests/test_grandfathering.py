"""
Grandfathered cost of acquisition, s.55(2)(ac) / s.112A.

The four worked examples are the ones in the CBDT FAQ on the LTCG tax on listed equity
(4 February 2018), per share: cost of acquisition = max(actual cost, min(FMV on
31-Jan-2018, sale value)).
"""

import copy
from datetime import date

import numpy as np
import pandas as pd
import pytest

from investbot.trading import grandfathering as gf
from investbot.trading.grandfathering import applies, fmv_31_jan_2018, grandfathered_cost
from investbot.trading.lot_tracker import Lot, sell_fifo, unrealised_by_lot
from investbot.trading.tax_ledger import compute_fiscal_year_tax
from investbot.trading.tax_rules import regime_for

BUY = date(2017, 6, 1)
SELL = date(2019, 6, 3)


@pytest.mark.parametrize(
    "actual, fmv, sale, cost, gain",
    [
        (100, 200, 250, 200, 50),  # FMV above cost, sale above FMV: gain from FMV
        (100, 200, 150, 150, 0),  # sale below FMV: cost capped at the sale value, no gain
        (100, 50, 150, 100, 50),  # FMV below cost: actual cost stays
        (100, 200, 50, 100, -50),  # sale below actual cost: loss on actual cost
    ],
)
def test_cbdt_faq_examples(actual, fmv, sale, cost, gain):
    assert grandfathered_cost(actual, fmv, sale) == cost
    assert sale - grandfathered_cost(actual, fmv, sale) == gain


def test_constants():
    assert gf.GF_ACQUIRED_BEFORE == date(2018, 2, 1)
    assert gf.GF_FMV_DATE == date(2018, 1, 31)
    assert gf.GF_TRANSFER_FROM == date(2018, 4, 1)


def test_applies_boundaries():
    assert applies(date(2018, 1, 31), date(2018, 4, 1), "LTCG")
    assert not applies(date(2018, 2, 1), date(2019, 6, 3), "LTCG")
    assert not applies(date(2018, 1, 31), date(2018, 3, 31), "LTCG")
    assert not applies(date(2017, 1, 1), date(2019, 6, 3), "STCG")
    assert not applies(date(2017, 1, 1), date(2019, 6, 3), "FLAT")


def bars(rows):
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d, _ in rows], name="timestamp")
    return pd.DataFrame({"high": [h for _, h in rows], "close": [h - 1 for _, h in rows]}, index=idx)


BARS = [("2018-01-29", 98.0), ("2018-01-30", 99.0), ("2018-01-31", 105.0)]


def test_fmv_picks_31_jan_high():
    assert fmv_31_jan_2018(bars(BARS)) == 105.0


def test_fmv_falls_back_to_previous_trading_day():
    assert fmv_31_jan_2018(bars(BARS[:2] + [("2018-02-01", 120.0)])) == 99.0


def test_fmv_none_without_earlier_bar():
    assert fmv_31_jan_2018(bars([("2018-02-01", 120.0), ("2018-02-02", 121.0)])) is None
    assert fmv_31_jan_2018(bars([])) is None
    assert fmv_31_jan_2018(pd.DataFrame({"high": []}, index=pd.DatetimeIndex([]))) is None


@pytest.mark.parametrize("bad", [np.nan, 0.0, -3.0])
def test_fmv_none_for_bad_high(bad):
    assert fmv_31_jan_2018(bars(BARS[:2] + [("2018-01-31", bad)])) is None


def test_fmv_truncation_invariant_and_pure():
    base = bars(BARS)
    snapshot = base.copy()
    later = bars([("2018-02-01", 999.0), ("2018-03-01", np.nan), ("2019-01-01", 1.0)])
    for extra in (later, bars([("2018-02-05", 1e9)])):
        assert fmv_31_jan_2018(pd.concat([base, extra])) == fmv_31_jan_2018(base) == 105.0
    assert fmv_31_jan_2018(pd.concat([later, base])) == 105.0  # unsorted index
    pd.testing.assert_frame_equal(base, snapshot)


def test_fmv_intraday_timestamps_on_the_day_count():
    b = bars([("2018-01-31 00:00", 105.0)])
    assert fmv_31_jan_2018(b) == 105.0


def lot(buy=BUY, price=100.0, qty=10.0, lot_id="L1"):
    return Lot(lot_id, 1, buy, price, qty, qty)


def sell(lots, qty, price, on, fmv=200.0):
    return sell_fifo(lots, qty, price, on, regime_for("equity", on), fmv_2018_01_31=fmv)


def test_sell_fifo_grandfathers_long_term_sale_after_april_2018():
    (r,), _ = sell([lot()], 10, 250.0, SELL)
    assert r.grandfathered and r.term == "LTCG"
    assert r.cost_basis == 2000.0 and r.gain == 500.0


def test_sell_fifo_cbdt_cases_through_lots():
    for fmv, price, gain in [(200, 150, 0), (50, 150, 500), (200, 50, -500)]:
        (r,), _ = sell([lot()], 10, price, SELL, fmv=fmv)
        assert r.gain == gain


def test_sale_before_april_2018_not_grandfathered():
    (r,), _ = sell([lot(buy=date(2016, 6, 1))], 10, 250.0, date(2018, 3, 15))
    assert r.term == "LTCG" and not r.grandfathered
    assert r.cost_basis == 1000.0 and r.gain == 1500.0


def test_lot_bought_on_or_after_feb_2018_not_grandfathered():
    (r,), _ = sell([lot(buy=date(2018, 2, 1))], 10, 250.0, SELL)
    assert not r.grandfathered and r.cost_basis == 1000.0


def test_short_term_sale_never_grandfathered():
    # bought 2018-01-31, sold 2018-06-01: short-term
    (r,), _ = sell([lot(buy=date(2018, 1, 31))], 10, 250.0, date(2018, 6, 1))
    assert r.term == "STCG" and not r.grandfathered and r.cost_basis == 1000.0


def test_none_fmv_is_identical_to_before():
    lots = [lot(), lot(buy=date(2018, 3, 1), price=120.0, lot_id="L2")]
    old = sell_fifo(lots, 15, 250.0, SELL, regime_for("equity", SELL))
    new = sell_fifo(lots, 15, 250.0, SELL, regime_for("equity", SELL), fmv_2018_01_31=None)
    assert old == new
    assert not any(r.grandfathered for r in new[0])


def test_mixed_fifo_sale_grandfathers_only_the_old_lot():
    lots = [lot(buy=date(2018, 2, 1), price=100.0, lot_id="L2"), lot()]
    snapshot = copy.deepcopy(lots)
    real, _ = sell(lots, 20, 250.0, SELL)
    assert [r.lot_id for r in real] == ["L1", "L2"]
    assert [r.grandfathered for r in real] == [True, False]
    assert [r.cost_basis for r in real] == [2000.0, 1000.0]
    assert [r.gain for r in real] == [500.0, 1500.0]
    assert lots == snapshot


def test_partial_lot_and_buy_costs_pro_rata():
    l = Lot("L1", 1, BUY, 100.0, 10, 10, buy_costs=50.0)
    (r,), _ = sell([l], 4, 150.0, SELL, fmv=90.0)  # FMV below cost: 4*100 + 20 stays
    assert r.grandfathered
    assert r.cost_basis == 420.0
    (r,), _ = sell([l], 4, 150.0, SELL, fmv=125.0)  # 4 * 125 = 500 > 420
    assert r.cost_basis == 500.0 and r.gain == 100.0


def test_unrealised_by_lot_column():
    lots = [lot(), lot(buy=date(2018, 2, 1), lot_id="L2"), lot(buy=date(2025, 1, 1), lot_id="L3")]
    reg = regime_for("equity", SELL)
    df = unrealised_by_lot(lots, 250.0, SELL, reg, fmv_2018_01_31=200.0)
    assert list(df["lot_id"]) == ["L1", "L2", "L3"]
    assert list(df["grandfathered_cost_basis"]) == [2000.0, 1000.0, 1000.0]
    assert list(df["cost_basis"]) == [1000.0, 1000.0, 1000.0]  # unchanged column
    off = unrealised_by_lot(lots, 250.0, SELL, reg)
    assert "grandfathered_cost_basis" not in off.columns


def test_grandfathered_gain_flows_into_fiscal_year_tax():
    on = date(2025, 8, 1)  # FY 2025-26: LTCG 12.5% above 1.25L
    lots = [Lot("L1", 1, date(2016, 1, 4), 100.0, 5000, 5000)]
    plain, _ = sell_fifo(lots, 5000, 250.0, on, regime_for("equity", on))
    step, _ = sell(lots, 5000, 250.0, on)
    assert plain[0].gain == 750_000.0 and step[0].gain == 250_000.0
    t_plain = compute_fiscal_year_tax(plain, "2025-26", cess_rate=0.0)
    t_step = compute_fiscal_year_tax(step, "2025-26", cess_rate=0.0)
    assert t_plain.estimated_tax_liability == pytest.approx((750_000 - 125_000) * 0.125)
    assert t_step.estimated_tax_liability == pytest.approx((250_000 - 125_000) * 0.125)
