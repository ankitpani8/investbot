"""
Migration: per-company fiscal-year-end months for the Q4 lag (task 500).

Before task 500 a quarterly row got the 60-day annual lag only when its period end fell
in March, so a December-FY company's Q4 was treated as public after 45 days. This
recomputes ``available_from`` for every ``lodr_lag`` quarterly row with the months of
its asset's annual period ends plus March (``fiscal_year_end_months``).

Dates only move later: a row whose recomputed date is not later than the stored one is
left alone, so a second run does nothing. ``published`` rows, annual rows and rows with
``available_from`` NULL are never touched. Runs after ``migrate_financials_point_in_time``;
a table without ``period_end`` (or no table) is a no-op.
"""
import logging
from collections import defaultdict

from sqlalchemy import inspect, select, update
from sqlalchemy.engine import Engine

from investbot.data.models import Financial
from investbot.scoring.fundamental.point_in_time import (
    SOURCE_LODR_LAG,
    available_from,
    fiscal_year_end_months,
)

logger = logging.getLogger(__name__)

TABLE = "financials"


def migrate_financials_fiscal_year_end(engine: Engine) -> dict[str, int]:
    """Move ``lodr_lag`` quarterly rows of non-March fiscal-year-end months to the annual lag.

    Returns ``{"checked": n, "updated": k}``: ``n`` ``lodr_lag`` quarterly rows with a
    period end and an ``available_from`` were recomputed, ``k`` of them moved later.
    """
    counts = {"checked": 0, "updated": 0}
    insp = inspect(engine)
    if not insp.has_table(TABLE) or "period_end" not in {c["name"] for c in insp.get_columns(TABLE)}:
        return counts

    t = Financial.__table__
    with engine.begin() as conn:
        annual_ends = defaultdict(list)
        for asset_id, period_end in conn.execute(
            select(t.c.asset_id, t.c.period_end).where(
                t.c.period_type == "annual", t.c.period_end.isnot(None)
            )
        ):
            annual_ends[asset_id].append(period_end)
        months = {a: fiscal_year_end_months(ends) for a, ends in annual_ends.items()}
        default_months = fiscal_year_end_months([])

        quarters = conn.execute(
            select(t.c.id, t.c.asset_id, t.c.period_end, t.c.available_from).where(
                t.c.period_type == "quarterly",
                t.c.available_from_source == SOURCE_LODR_LAG,
                t.c.available_from.isnot(None),
                t.c.period_end.isnot(None),
            )
        ).all()
        for row_id, asset_id, period_end, current in quarters:
            counts["checked"] += 1
            new, _ = available_from(
                period_end, "quarterly",
                fiscal_year_end_months=months.get(asset_id, default_months),
            )
            if new > current:
                conn.execute(update(t).where(t.c.id == row_id).values(available_from=new))
                counts["updated"] += 1

    if counts["updated"]:
        logger.info(
            "financials: %d of %d lodr_lag quarterly rows moved to the annual lag "
            "(non-March fiscal year end)", counts["updated"], counts["checked"],
        )
    return counts
