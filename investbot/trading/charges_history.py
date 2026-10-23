"""
Dated history of the non-STT charges on NSE equity delivery trades.

Pure compute over ``config/charges_history/nse_equity_delivery_charges.csv``: stamp duty,
exchange transaction charge, SEBI turnover fee and service tax / GST, each with the dates it
was in force and the document that says so. A period without a sourced row has no value
(``charges_on`` returns None for it); nothing is filled in from memory.

Nothing in the cost model reads this module yet. ``with_dated_charges`` is the hook the
simulator could call per trade date (see docs/DECISIONS_PENDING.md "Dated non-STT charges: wiring").

Rates use the unit of the matching ``TaxRegimeParams`` field: a fraction of trade value, and
for ``gst_or_service_tax`` a fraction of the taxable charges (see the ``basis`` column).
``side`` is ``buy``, ``sell`` or ``both``; ``both`` counts as a row for each side.
"""

from __future__ import annotations

import dataclasses
from datetime import date
from pathlib import Path

import pandas as pd

from investbot.trading.tax_rules import TaxRegimeParams

CHARGES_CSV = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "charges_history"
    / "nse_equity_delivery_charges.csv"
)

COLUMNS = (
    "charge",
    "effective_from",
    "effective_to",
    "rate",
    "side",
    "basis",
    "source_url",
    "source_title",
    "note",
)
CHARGES = ("stamp_duty", "exchange_txn_charge", "sebi_turnover_fee", "gst_or_service_tax")
SIDES = ("buy", "sell", "both")

# charge -> the TaxRegimeParams field it replaces
REGIME_FIELDS = {
    "stamp_duty": "stamp_duty_rate",
    "exchange_txn_charge": "exchange_txn_charge_rate",
    "sebi_turnover_fee": "sebi_turnover_fee_rate",
    "gst_or_service_tax": "gst_rate",
}

# Periods inside a charge's sourced span that no row covers: (charge, side, first day, last day).
# Empty today; tests/test_charges_history.py checks this against the file.
KNOWN_GAPS: tuple[tuple[str, str, date, date], ...] = ()

# First day with a sourced value per (charge, side); every earlier day back to the start of the
# dated equity regimes (2004-10-01) is unsourced and listed in docs/DECISIONS_PENDING.md.
FIRST_SOURCED: dict[tuple[str, str], date] = {
    ("stamp_duty", "buy"): date(2005, 5, 7),
    ("stamp_duty", "sell"): date(2005, 5, 7),
    ("exchange_txn_charge", "buy"): date(2009, 9, 7),
    ("exchange_txn_charge", "sell"): date(2009, 9, 7),
    ("sebi_turnover_fee", "buy"): date(2009, 6, 29),
    ("sebi_turnover_fee", "sell"): date(2009, 6, 29),
    ("gst_or_service_tax", "buy"): date(2004, 10, 1),
    ("gst_or_service_tax", "sell"): date(2004, 10, 1),
}


def _parse_date(value: object, what: str, line: int) -> date | None:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise ValueError(f"line {line}: {what} {text!r} is not an ISO date (YYYY-MM-DD)") from None


def _covers(row: pd.Series, d: date) -> bool:
    end = row["effective_to"]
    return row["effective_from"] <= d and (end is None or d <= end)


def _sides(side: str) -> tuple[str, ...]:
    return ("buy", "sell") if side == "both" else (side,)


def load_charge_history(path: Path | None = None) -> pd.DataFrame:
    """Parse and validate the charges file. Raises ValueError on any malformed row.

    Dates become ``datetime.date`` (``None`` for an open ``effective_to``) in object columns,
    ``rate`` a float. Checks: ISO dates, ``effective_from <= effective_to``, known charge and
    side, rate in [0, 1), non-empty ``source_url`` and ``basis``, and no two rows covering the
    same day for one (charge, side), where ``both`` counts as ``buy`` and ``sell``.
    """
    src = CHARGES_CSV if path is None else Path(path)
    raw = pd.read_csv(src, dtype=str, keep_default_na=False)
    missing = [c for c in COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{src.name}: missing columns {missing}")

    records = []
    for i, r in raw.iterrows():
        line = int(i) + 2  # header is line 1
        charge = r["charge"].strip()
        if charge not in CHARGES:
            raise ValueError(f"line {line}: unknown charge {charge!r} (known: {', '.join(CHARGES)})")
        side = r["side"].strip()
        if side not in SIDES:
            raise ValueError(f"line {line}: side {side!r} must be one of {SIDES}")
        start = _parse_date(r["effective_from"], "effective_from", line)
        if start is None:
            raise ValueError(f"line {line}: effective_from is empty")
        end = _parse_date(r["effective_to"], "effective_to", line)
        if end is not None and start > end:
            raise ValueError(f"line {line}: effective_from {start} is after effective_to {end}")
        try:
            rate = float(r["rate"])
        except ValueError:
            raise ValueError(f"line {line}: rate {r['rate']!r} is not a number") from None
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"line {line}: rate {rate} is outside [0, 1)")
        if not r["source_url"].strip():
            raise ValueError(f"line {line}: source_url is empty")
        if not r["basis"].strip():
            raise ValueError(f"line {line}: basis is empty")
        records.append(
            {
                "charge": charge,
                "effective_from": start,
                "effective_to": end,
                "rate": rate,
                "side": side,
                "basis": r["basis"].strip(),
                "source_url": r["source_url"].strip(),
                "source_title": r["source_title"].strip(),
                "note": r["note"].strip(),
            }
        )
    frame = pd.DataFrame.from_records(records, columns=list(COLUMNS))
    frame = frame.astype({"effective_from": object, "effective_to": object})

    for (charge, side), group in _by_effective_side(frame):
        spans = sorted(zip(group["effective_from"], group["effective_to"]), key=lambda s: s[0])
        for (s1, e1), (s2, _e2) in zip(spans, spans[1:]):
            if e1 is None or s2 <= e1:
                raise ValueError(
                    f"{src.name}: overlapping rows for {charge}/{side}: "
                    f"{s1}..{e1 or 'open'} and from {s2}"
                )
    return frame.sort_values(["charge", "effective_from"], kind="stable").reset_index(drop=True)


def _by_effective_side(history: pd.DataFrame):
    """Yield ``((charge, side), rows)`` with a ``both`` row listed under buy and sell."""
    for charge in history["charge"].unique():
        rows = history[history["charge"] == charge]
        for side in ("buy", "sell"):
            yield (charge, side), rows[rows["side"].isin((side, "both"))]


def charges_on(
    d: date, history: pd.DataFrame | None = None, *, side: str = "buy"
) -> dict[str, float | None]:
    """The four charges in force on ``d`` for a ``side`` ('buy' or 'sell') trade.

    Uses only rows whose [effective_from, effective_to] contains ``d`` (both bounds
    inclusive). A charge with no sourced row on ``d`` is None. ``side`` defaults to 'buy'
    because ``TaxRegimeParams.stamp_duty_rate`` is charged on the buy leg.
    """
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    hist = load_charge_history() if history is None else history
    out: dict[str, float | None] = {}
    for charge in CHARGES:
        rows = hist[(hist["charge"] == charge) & hist["side"].isin((side, "both"))]
        hits = [r for _, r in rows.iterrows() if _covers(r, d)]
        if len(hits) > 1:
            raise ValueError(f"{len(hits)} {charge} rows cover {d} for side {side}")
        out[charge] = float(hits[0]["rate"]) if hits else None
    return out


def with_dated_charges(
    regime: TaxRegimeParams, d: date, history: pd.DataFrame | None = None
) -> tuple[TaxRegimeParams, list[str]]:
    """``regime`` with its four non-STT charge fields set to the sourced values on ``d``.

    Returns ``(new_regime, unsourced)``: ``unsourced`` lists the ``TaxRegimeParams`` fields
    left as ``regime``'s own copy because no sourced row covers ``d``. STT, tax rates and every
    other field are untouched. The buy-leg values are used (``estimate_costs`` charges stamp
    duty on BUY only), so the sell-side stamp duty of 2005-2020 is not represented.
    Raises ValueError for a non-equity regime: the history is NSE equity delivery only.
    """
    if regime.asset_class_name != "equity":
        raise ValueError(f"charge history covers equity only, got {regime.asset_class_name!r}")
    dated = charges_on(d, history, side="buy")
    updates: dict[str, float] = {}
    unsourced: list[str] = []
    for charge, field in REGIME_FIELDS.items():
        value = dated[charge]
        if value is None:
            unsourced.append(field)
        else:
            updates[field] = value
    return dataclasses.replace(regime, **updates), unsourced
