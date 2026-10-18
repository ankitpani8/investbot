import copy
from datetime import date

import pytest

from investbot.trading import lot_tracker
from investbot.trading.lot_tracker import Lot, sell_fifo, unrealised_by_lot
from investbot.trading.tax_rules import regime_params
from investbot.trading.transaction_costs import estimate_costs

EQ = regime_params("IN_EQUITY_LISTED")
CRYPTO = regime_params("IN_CRYPTO_VDA")
SELL_DATE = date(2026, 6, 1)


def three_lots():
    return [
        Lot("L1", 1, date(2024, 1, 10), 100.0, 10, 10),  # long-term on SELL_DATE
        Lot("L2", 1, date(2025, 5, 31), 120.0, 10, 10),  # bought 1 yr + 1 day earlier: long-term
        Lot("L3", 1, date(2025, 9, 1), 150.0, 10, 10),  # short-term
    ]


def test_partial_sale_across_three_lots():
    lots = three_lots()
    real, new = sell_fifo(lots, 25, 200.0, SELL_DATE, EQ)
    assert [r.lot_id for r in real] == ["L1", "L2", "L3"]
    assert [r.quantity for r in real] == [10, 10, 5]
    assert [r.cost_basis for r in real] == [1000.0, 1200.0, 750.0]
    assert [r.sale_value for r in real] == [2000.0, 2000.0, 1000.0]
    assert [r.gain for r in real] == [1000.0, 800.0, 250.0]
    assert [r.term for r in real] == ["LTCG", "LTCG", "STCG"]
    assert [l.remaining_qty for l in new] == [0, 0, 5]
    assert all(r.sell_date == SELL_DATE and r.asset_id == 1 for r in real)


def test_inputs_unchanged():
    lots = three_lots()
    snapshot = copy.deepcopy(lots)
    _, new = sell_fifo(lots, 15, 200.0, SELL_DATE, EQ)
    assert lots == snapshot
    assert new is not lots and all(n is not o for n, o in zip(new, lots))


def test_fifo_by_buy_date_not_list_order():
    lots = list(reversed(three_lots()))
    real, new = sell_fifo(lots, 12, 200.0, SELL_DATE, EQ)
    assert [r.lot_id for r in real] == ["L1", "L2"]
    assert [r.quantity for r in real] == [10, 2]
    assert [l.lot_id for l in new] == ["L3", "L2", "L1"]  # input order kept
    assert [l.remaining_qty for l in new] == [10, 8, 0]


def test_skips_exhausted_lots_and_sells_everything():
    lots = three_lots()
    _, after_first = sell_fifo(lots, 10, 200.0, SELL_DATE, EQ)
    real, after_all = sell_fifo(after_first, 20, 200.0, SELL_DATE, EQ)
    assert [r.lot_id for r in real] == ["L2", "L3"]
    assert all(l.remaining_qty == 0 for l in after_all)


def test_oversell_raises_and_validation():
    lots = three_lots()
    with pytest.raises(ValueError):
        sell_fifo(lots, 30.5, 200.0, SELL_DATE, EQ)
    with pytest.raises(ValueError):
        sell_fifo(lots, 0, 200.0, SELL_DATE, EQ)
    with pytest.raises(ValueError):
        sell_fifo([], 1, 200.0, SELL_DATE, EQ)
    mixed = three_lots() + [Lot("X", 2, date(2025, 1, 1), 1.0, 1, 1)]
    with pytest.raises(ValueError):
        sell_fifo(mixed, 1, 200.0, SELL_DATE, EQ)


def test_loss_is_negative_gain():
    real, _ = sell_fifo(three_lots(), 25, 90.0, SELL_DATE, EQ)
    assert [r.gain for r in real] == [-100.0, -300.0, -300.0]
    assert [r.term for r in real] == ["LTCG", "LTCG", "STCG"]  # LTCL, LTCL, STCL


def test_stt_not_deductible_and_costs_pro_rata():
    costs = estimate_costs("SELL", 5000.0, EQ)  # 25 units x 200
    deductible = costs.total - costs.stt
    real, _ = sell_fifo(three_lots(), 25, 200.0, SELL_DATE, EQ, sell_costs=costs)
    assert sum(r.sell_costs for r in real) == pytest.approx(deductible)
    assert sum(r.sell_costs for r in real) < costs.total
    assert [r.sell_costs for r in real] == pytest.approx(
        [deductible * 10 / 25, deductible * 10 / 25, deductible * 5 / 25]
    )
    for r in real:
        assert r.gain == pytest.approx(r.sale_value - r.cost_basis - r.sell_costs)


def test_buy_costs_added_to_cost_basis_pro_rata():
    lots = [Lot("A", 1, date(2025, 9, 1), 100.0, 10, 10, buy_costs=30.0)]
    real, new = sell_fifo(lots, 4, 100.0, SELL_DATE, EQ)
    assert real[0].cost_basis == pytest.approx(400.0 + 12.0)
    assert real[0].gain == pytest.approx(-12.0)
    real2, _ = sell_fifo(new, 6, 100.0, SELL_DATE, EQ)
    assert real2[0].cost_basis == pytest.approx(600.0 + 18.0)  # the two parts sum to 30


def test_term_boundary_on_sale_date():
    lot = [Lot("A", 1, date(2025, 3, 15), 100.0, 1, 1)]
    assert sell_fifo(lot, 1, 110.0, date(2026, 3, 15), EQ)[0][0].term == "STCG"
    assert sell_fifo(lot, 1, 110.0, date(2026, 3, 16), EQ)[0][0].term == "LTCG"


def test_crypto_flat_term_and_no_cost_deduction():
    lots = [Lot("C1", 9, date(2020, 1, 1), 100.0, 2.5, 2.5)]
    costs = estimate_costs("SELL", 500.0, CRYPTO)
    real, new = sell_fifo(lots, 1.25, 200.0, SELL_DATE, CRYPTO, sell_costs=costs)
    assert real[0].term == "FLAT"
    assert real[0].sell_costs == 0.0  # Section 115BBH
    assert real[0].gain == pytest.approx(1.25 * 100.0)
    assert new[0].remaining_qty == pytest.approx(1.25)


def test_crypto_flat_term_cost_switch(monkeypatch):
    monkeypatch.setattr(lot_tracker, "FLAT_TERM_DEDUCTS_SELL_COSTS", True)
    lots = [Lot("C1", 9, date(2020, 1, 1), 100.0, 1, 1)]
    costs = estimate_costs("SELL", 200.0, CRYPTO)
    real, _ = sell_fifo(lots, 1, 200.0, SELL_DATE, CRYPTO, sell_costs=costs)
    assert real[0].sell_costs == pytest.approx(costs.total)


def test_fractional_quantity_rounding_does_not_leave_dust():
    lots = [Lot("A", 1, date(2025, 9, 1), 10.0, 0.3, 0.3)]
    _, new = sell_fifo(lots, 0.1 + 0.2, 10.0, SELL_DATE, EQ)
    assert new[0].remaining_qty == 0.0


def test_unrealised_by_lot():
    lots = three_lots()
    _, lots = sell_fifo(lots, 4, 200.0, SELL_DATE, EQ)  # L1 has 6 left
    lots[2].buy_costs = 15.0
    df = unrealised_by_lot(lots, 130.0, date(2026, 6, 1), EQ)
    assert list(df.columns) == [
        "lot_id", "remaining_qty", "cost_basis", "market_value",
        "unrealised_gain", "term_if_sold_now", "days_to_long_term",
    ]
    assert list(df["lot_id"]) == ["L1", "L2", "L3"]
    assert list(df["remaining_qty"]) == [6, 10, 10]
    assert list(df["cost_basis"]) == [600.0, 1200.0, 1515.0]
    assert list(df["market_value"]) == [780.0, 1300.0, 1300.0]
    assert list(df["unrealised_gain"]) == pytest.approx([180.0, 100.0, -215.0])
    assert list(df["term_if_sold_now"]) == ["LTCG", "LTCG", "STCG"]
    assert list(df["days_to_long_term"]) == [0, 0, (date(2026, 9, 2) - date(2026, 6, 1)).days]


def test_unrealised_skips_closed_lots_and_handles_empty():
    _, lots = sell_fifo(three_lots(), 30, 100.0, SELL_DATE, EQ)
    df = unrealised_by_lot(lots, 100.0, SELL_DATE, EQ)
    assert df.empty and "term_if_sold_now" in df.columns
    assert unrealised_by_lot([], 100.0, SELL_DATE, EQ).empty


def test_unrealised_crypto():
    lots = [Lot("C1", 9, date(2020, 1, 1), 100.0, 1, 1)]
    df = unrealised_by_lot(lots, 150.0, SELL_DATE, CRYPTO)
    assert df.loc[0, "term_if_sold_now"] == "FLAT" and df.loc[0, "days_to_long_term"] == 0
