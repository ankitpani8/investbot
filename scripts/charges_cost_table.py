"""
Round-trip cost of a Rs 1,00,000 trade under each dated equity regime, as stored versus with the
sourced non-STT charges of ``investbot.trading.charges_history``.

Prints the markdown table used in docs/DECISIONS_PENDING.md ("Dated non-STT charges: wiring").
Pure: no DB, no network. Run: ``python -m scripts.charges_cost_table``.
"""

from __future__ import annotations

from datetime import date

from investbot.trading.charges_history import load_charge_history, with_dated_charges
from investbot.trading.tax_rules import regime_for
from investbot.trading.transaction_costs import round_trip_cost_pct

TRADE_VALUE = 100_000

# One session inside each dated equity regime period.
SAMPLE_DATES = (
    date(2005, 3, 1),
    date(2006, 1, 2),
    date(2007, 6, 1),
    date(2010, 6, 1),
    date(2014, 6, 2),
    date(2022, 6, 1),
    date(2025, 6, 2),
)


def cost_table_rows() -> list[dict]:
    """One dict per sample date: regime, stored and dated round-trip cost (fractions), unsourced fields."""
    history = load_charge_history()
    rows = []
    for d in SAMPLE_DATES:
        stored = regime_for("equity", d)
        dated, unsourced = with_dated_charges(stored, d, history)
        rows.append(
            {
                "date": d,
                "regime": stored.regime_name,
                "stored": round_trip_cost_pct(TRADE_VALUE, stored),
                "dated": round_trip_cost_pct(TRADE_VALUE, dated),
                "unsourced": unsourced,
            }
        )
    return rows


def render_table(rows: list[dict] | None = None) -> str:
    rows = cost_table_rows() if rows is None else rows
    lines = [
        "| Date | Regime | (i) as stored | (ii) with_dated_charges | Fields left as stored |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        left = ", ".join(r["unsourced"]) or "none"
        lines.append(
            f"| {r['date']} | {r['regime']} | {r['stored'] * 100:.4f}% | {r['dated'] * 100:.4f}% | {left} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(render_table())
