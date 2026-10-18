"""
Transaction-cost calculator (master plan §2.8 cost table).

Pure compute. Rates come from a ``TaxRegimeParams``; a rate of None counts as 0.
DP (depository participant) charges are not modelled yet (see docs/BACKLOG.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from investbot.trading.tax_rules import TaxRegimeParams

BROKERAGE_PER_ORDER = 20.0  # discount-broker flat fee in INR, plan §2.8


@dataclass
class CostBreakdown:
    """INR costs of one trade side. ``tds`` is a prepaid tax credit and is not in ``total``."""

    stt: float
    stamp_duty: float
    exchange_txn: float
    sebi_fee: float
    brokerage: float
    gst: float
    total: float
    tds: float = 0.0


def _rate(value: float | None) -> float:
    return 0.0 if value is None else value


def estimate_costs(
    side: Literal["BUY", "SELL"],
    value: float,
    regime: TaxRegimeParams,
    *,
    orders: int = 1,
    brokerage_per_order: float = BROKERAGE_PER_ORDER,
) -> CostBreakdown:
    """Costs of trading ``value`` INR on one side.

    STT uses the side's rate. Stamp duty is BUY only. Exchange and SEBI fees apply to both
    sides. GST = gst_rate x (brokerage + exchange_txn + sebi_fee). ``tds`` (SELL only,
    Section 194S) is excluded from ``total`` because it is later credited against tax.
    """
    if side not in ("BUY", "SELL"):
        raise ValueError(f"side must be 'BUY' or 'SELL', got {side!r}")
    if value < 0:
        raise ValueError("value must be >= 0")
    if orders < 0:
        raise ValueError("orders must be >= 0")

    buy = side == "BUY"
    stt = value * _rate(regime.stt_buy_rate if buy else regime.stt_sell_rate)
    stamp_duty = value * _rate(regime.stamp_duty_rate) if buy else 0.0
    exchange_txn = value * _rate(regime.exchange_txn_charge_rate)
    sebi_fee = value * _rate(regime.sebi_turnover_fee_rate)
    brokerage = orders * brokerage_per_order
    gst = _rate(regime.gst_rate) * (brokerage + exchange_txn + sebi_fee)
    total = stt + stamp_duty + exchange_txn + sebi_fee + brokerage + gst
    tds = 0.0 if buy else value * _rate(regime.tds_rate)
    return CostBreakdown(
        stt=stt,
        stamp_duty=stamp_duty,
        exchange_txn=exchange_txn,
        sebi_fee=sebi_fee,
        brokerage=brokerage,
        gst=gst,
        total=total,
        tds=tds,
    )


def deductible_costs(costs: CostBreakdown) -> float:
    """Costs deductible as expenditure on transfer: ``total - stt``.

    STT is not deductible (Section 48). TDS is already outside ``total``.
    """
    return costs.total - costs.stt


def round_trip_cost_pct(value: float, regime: TaxRegimeParams) -> float:
    """Buy plus sell ``total`` (one order each) as a fraction of ``value``."""
    if value <= 0:
        raise ValueError("value must be > 0")
    buy = estimate_costs("BUY", value, regime)
    sell = estimate_costs("SELL", value, regime)
    return (buy.total + sell.total) / value
