"""
FIFO tax-lot tracker.

Pure compute: no DB, no mutation of inputs. FIFO is the only disposal method. CBDT
Circular 768 (1998) mandates FIFO for demat holdings, for both the period of holding and
the cost, so there is no tax-optimal lot picking here (docs/DECISIONS_PENDING.md). Under
FIFO the tax levers are how much and when to sell, not which lots.

Cost conventions
- cost of acquisition = quantity x buy_price + the lot's ``buy_costs`` pro rata by
  quantity. ``buy_costs`` is the lot's total acquisition cost (brokerage, stamp duty,
  fees); STT is not part of the cost of acquisition, so callers pass
  ``deductible_costs(buy_breakdown)``.
- deductible sell costs = ``CostBreakdown.total - stt`` (STT is not deductible,
  Section 48), allocated to lots pro rata by quantity sold.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Literal

import pandas as pd

from investbot.trading.tax_rules import (
    TaxRegimeParams,
    days_to_long_term,
    is_long_term,
)
from investbot.trading.transaction_costs import CostBreakdown, deductible_costs

# Section 115BBH allows no deduction for crypto (VDA) other than the cost of acquisition,
# so FLAT-term realisations carry sell_costs = 0 unless this is switched on.
# See docs/DECISIONS_PENDING.md ("Transfer expenses on FLAT-taxed assets").
FLAT_TERM_DEDUCTS_SELL_COSTS = False

_QTY_EPS = 1e-9

Term = Literal["STCG", "LTCG", "FLAT"]


@dataclass
class Lot:
    lot_id: str
    asset_id: int
    buy_date: date
    buy_price: float
    quantity: float
    remaining_qty: float
    buy_costs: float = 0.0  # total acquisition costs of the original quantity


@dataclass
class Realisation:
    """Gain realised on (part of) one lot.

    ``gain = sale_value - cost_basis - sell_costs``; negative is a loss, and the
    STCL/LTCL split follows from the sign and ``term``.
    """

    lot_id: str
    asset_id: int
    buy_date: date
    sell_date: date
    quantity: float
    cost_basis: float
    sale_value: float
    sell_costs: float
    gain: float
    term: Term


def _term(buy_date: date, on: date, regime: TaxRegimeParams) -> Term:
    if regime.ltcg_holding_days is None:
        return "FLAT"
    return "LTCG" if is_long_term(buy_date, on, regime) else "STCG"


def _cost_basis(lot: Lot, qty: float) -> float:
    return qty * lot.buy_price + lot.buy_costs * qty / lot.quantity


def sell_fifo(
    lots: list[Lot],
    quantity: float,
    sell_price: float,
    sell_date: date,
    regime: TaxRegimeParams,
    *,
    sell_costs: CostBreakdown | None = None,
) -> tuple[list[Realisation], list[Lot]]:
    """Sell ``quantity`` units oldest-first.

    Returns (realisations, new lot states). The new list holds every input lot, in input
    order, with updated ``remaining_qty``; inputs are never mutated. Raises ValueError for
    a non-positive quantity, mixed assets, or selling more than is held.
    """
    if quantity <= 0:
        raise ValueError("quantity must be > 0")
    if sell_price < 0:
        raise ValueError("sell_price must be >= 0")
    if len({lot.asset_id for lot in lots}) > 1:
        raise ValueError("lots belong to more than one asset")
    held = sum(lot.remaining_qty for lot in lots)
    if quantity > held + _QTY_EPS:
        raise ValueError(f"cannot sell {quantity} units, only {held} held")

    deductible = deductible_costs(sell_costs) if sell_costs is not None else 0.0
    remaining_to_sell = quantity
    consumed: dict[int, float] = {}
    realisations: list[Realisation] = []

    order = sorted(range(len(lots)), key=lambda i: lots[i].buy_date)  # stable: ties keep input order
    for i in order:
        lot = lots[i]
        if remaining_to_sell <= _QTY_EPS:
            break
        if lot.remaining_qty <= 0:
            continue
        qty = min(lot.remaining_qty, remaining_to_sell)
        remaining_to_sell -= qty
        consumed[i] = qty

        term = _term(lot.buy_date, sell_date, regime)
        cost_basis = _cost_basis(lot, qty)
        sale_value = qty * sell_price
        lot_sell_costs = deductible * qty / quantity
        if term == "FLAT" and not FLAT_TERM_DEDUCTS_SELL_COSTS:
            lot_sell_costs = 0.0
        realisations.append(
            Realisation(
                lot_id=lot.lot_id,
                asset_id=lot.asset_id,
                buy_date=lot.buy_date,
                sell_date=sell_date,
                quantity=qty,
                cost_basis=cost_basis,
                sale_value=sale_value,
                sell_costs=lot_sell_costs,
                gain=sale_value - cost_basis - lot_sell_costs,
                term=term,
            )
        )

    new_lots: list[Lot] = []
    for i, lot in enumerate(lots):
        left = lot.remaining_qty - consumed.get(i, 0.0)
        if left < _QTY_EPS:
            left = 0.0
        new_lots.append(replace(lot, remaining_qty=left))
    return realisations, new_lots


_UNREALISED_COLUMNS = [
    "lot_id",
    "remaining_qty",
    "cost_basis",
    "market_value",
    "unrealised_gain",
    "term_if_sold_now",
    "days_to_long_term",
]


def unrealised_by_lot(
    lots: list[Lot], price: float, as_of: date, regime: TaxRegimeParams
) -> pd.DataFrame:
    """One row per open lot (FIFO order) valued at ``price``.

    ``unrealised_gain`` is market value less cost of acquisition, before any sell costs.
    """
    rows = []
    for lot in sorted(lots, key=lambda x: x.buy_date):
        if lot.remaining_qty <= 0:
            continue
        cost_basis = _cost_basis(lot, lot.remaining_qty)
        market_value = lot.remaining_qty * price
        rows.append(
            {
                "lot_id": lot.lot_id,
                "remaining_qty": lot.remaining_qty,
                "cost_basis": cost_basis,
                "market_value": market_value,
                "unrealised_gain": market_value - cost_basis,
                "term_if_sold_now": _term(lot.buy_date, as_of, regime),
                "days_to_long_term": days_to_long_term(lot.buy_date, as_of, regime),
            }
        )
    return pd.DataFrame(rows, columns=_UNREALISED_COLUMNS)
