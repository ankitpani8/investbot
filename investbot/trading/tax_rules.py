"""
Indian tax rules: regime parameters, fiscal year and the holding-period rule.

Pure compute. Reads ``investbot.config.tax_regimes.TAX_REGIMES`` (never the DB).

Holding period (Income-tax Act s.2(42A)): a listed share is long-term when held for
*more than* twelve months. That is a calendar-month test, not a 365-day count, so a buy
on 2024-03-15 becomes long-term from 2025-03-16 (leap years do not shift it). The month
count is derived from the regime as ``ltcg_holding_days // 365 * 12``.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, timedelta

from investbot.config.tax_regimes import TAX_REGIMES

# True: the sale must fall strictly after the calendar anniversary ("more than twelve
# months"), so the first long-term sale date is anniversary + 1 day.
# False: a sale on the anniversary itself already counts as long-term.
# See docs/DECISIONS_PENDING.md ("Holding-period rule").
LTCG_STRICTLY_AFTER_ANNIVERSARY = True


@dataclass(frozen=True)
class TaxRegimeParams:
    """One ``TAX_REGIMES`` entry, with the effective dates parsed."""

    country: str
    asset_class_name: str
    regime_name: str
    stcg_rate: float | None
    ltcg_rate: float | None
    ltcg_holding_days: int | None
    ltcg_annual_exemption: float
    flat_tax_rate: float | None
    loss_offset_allowed: bool
    stcl_offsets: str
    ltcl_offsets: str
    loss_carryforward_years: int
    wash_sale_days: int
    tds_rate: float | None
    stt_buy_rate: float | None
    stt_sell_rate: float | None
    stamp_duty_rate: float | None
    exchange_txn_charge_rate: float | None
    sebi_turnover_fee_rate: float | None
    gst_rate: float | None
    effective_from: date
    effective_to: date | None


def regime_params(regime_name: str) -> TaxRegimeParams:
    """Look up a regime by name in the config list. Raises ValueError if unknown."""
    for entry in TAX_REGIMES:
        if entry["regime_name"] == regime_name:
            fields = dict(entry)
            fields["effective_from"] = date.fromisoformat(entry["effective_from"])
            eff_to = entry["effective_to"]
            fields["effective_to"] = date.fromisoformat(eff_to) if eff_to else None
            return TaxRegimeParams(**fields)
    known = ", ".join(e["regime_name"] for e in TAX_REGIMES)
    raise ValueError(f"unknown tax regime {regime_name!r} (known: {known})")


def fiscal_year(d: date) -> str:
    """Indian fiscal year label (April to March): 2026-03-31 -> '2025-26'."""
    start = d.year if d.month >= 4 else d.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def _add_months(d: date, months: int) -> date:
    """Calendar-month addition; a day that does not exist clamps to month end (29 Feb -> 28 Feb)."""
    index = d.year * 12 + (d.month - 1) + months
    year, month = divmod(index, 12)
    month += 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def first_long_term_date(buy_date: date, regime: TaxRegimeParams) -> date | None:
    """First sale date taxed as long-term, or None if the regime has no long-term class."""
    if regime.ltcg_holding_days is None:
        return None
    months = regime.ltcg_holding_days // 365 * 12
    anniversary = _add_months(buy_date, months)
    if LTCG_STRICTLY_AFTER_ANNIVERSARY:
        return anniversary + timedelta(days=1)
    return anniversary


def is_long_term(buy_date: date, sell_date: date, regime: TaxRegimeParams) -> bool:
    """True if a sale on ``sell_date`` of a lot bought on ``buy_date`` is long-term."""
    first = first_long_term_date(buy_date, regime)
    return first is not None and sell_date >= first


def days_to_long_term(buy_date: date, as_of: date, regime: TaxRegimeParams) -> int:
    """Calendar days from ``as_of`` until a sale becomes long-term; 0 once long-term.

    A regime with no long-term class (crypto) also returns 0: there is no cliff to wait for.
    """
    first = first_long_term_date(buy_date, regime)
    if first is None:
        return 0
    return max(0, (first - as_of).days)
