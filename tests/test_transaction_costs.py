import pytest

from investbot.trading.tax_rules import regime_params
from investbot.trading.transaction_costs import (
    BROKERAGE_PER_ORDER,
    deductible_costs,
    estimate_costs,
    round_trip_cost_pct,
)

EQ = regime_params("IN_EQUITY_LISTED")
CRYPTO = regime_params("IN_CRYPTO_VDA")
V = 100_000.0


def test_equity_buy_hand_computed():
    c = estimate_costs("BUY", V, EQ)
    assert c.stt == pytest.approx(100.0)
    assert c.stamp_duty == pytest.approx(15.0)
    assert c.exchange_txn == pytest.approx(3.25)
    assert c.sebi_fee == pytest.approx(0.10)
    assert c.brokerage == pytest.approx(20.0)
    assert c.gst == pytest.approx(0.18 * (20.0 + 3.25 + 0.10))  # 4.203
    assert c.total == pytest.approx(100 + 15 + 3.25 + 0.10 + 20 + 4.203)
    assert c.tds == 0.0


def test_equity_sell_hand_computed():
    c = estimate_costs("SELL", V, EQ)
    assert c.stt == pytest.approx(100.0)
    assert c.stamp_duty == 0.0
    assert c.exchange_txn == pytest.approx(3.25)
    assert c.sebi_fee == pytest.approx(0.10)
    assert c.brokerage == pytest.approx(20.0)
    assert c.gst == pytest.approx(4.203)
    assert c.total == pytest.approx(127.553)
    assert c.tds == 0.0


def test_round_trip_in_plan_range():
    pct = round_trip_cost_pct(V, EQ)
    assert pct == pytest.approx((142.553 + 127.553) / V)
    assert 0.0025 <= pct <= 0.0035


def test_brokerage_is_flat_so_small_trades_cost_more_in_pct():
    assert round_trip_cost_pct(10_000, EQ) > round_trip_cost_pct(V, EQ) > round_trip_cost_pct(1_000_000, EQ)


def test_orders_and_brokerage_override():
    assert BROKERAGE_PER_ORDER == 20.0
    c = estimate_costs("SELL", V, EQ, orders=3, brokerage_per_order=0.0)
    assert c.brokerage == 0.0
    assert c.gst == pytest.approx(0.18 * (3.25 + 0.10))
    c3 = estimate_costs("SELL", V, EQ, orders=3)
    assert c3.brokerage == pytest.approx(60.0)


def test_crypto_sell_tds_excluded_from_total():
    c = estimate_costs("SELL", V, CRYPTO)
    assert c.tds == pytest.approx(1000.0)
    assert c.stt == 0.0 and c.stamp_duty == 0.0 and c.exchange_txn == 0.0 and c.sebi_fee == 0.0
    assert c.gst == pytest.approx(0.18 * 20.0)
    assert c.total == pytest.approx(20.0 + 3.6)  # no TDS in total


def test_crypto_buy_has_no_tds():
    assert estimate_costs("BUY", V, CRYPTO).tds == 0.0


def test_deductible_costs_exclude_stt():
    c = estimate_costs("SELL", V, EQ)
    assert deductible_costs(c) == pytest.approx(c.total - 100.0)
    assert deductible_costs(estimate_costs("SELL", V, CRYPTO)) == pytest.approx(23.6)


def test_zero_value_and_validation():
    c = estimate_costs("SELL", 0.0, EQ)
    assert c.stt == 0.0 and c.total == pytest.approx(20.0 * 1.18)
    with pytest.raises(ValueError):
        estimate_costs("HOLD", V, EQ)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        estimate_costs("BUY", -1.0, EQ)
    with pytest.raises(ValueError):
        round_trip_cost_pct(0.0, EQ)
