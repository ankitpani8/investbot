"""
Migration: point-in-time financials (task 410).

The old ``financials`` table stored the yfinance period end in ``filing_date`` and was
unique on ``(asset_id, fiscal_year, period_type)``, which collapsed quarterly rows to one
per calendar year. The new table adds ``period_end``, ``available_from`` and
``available_from_source`` and is unique on ``(asset_id, period_end, period_type)``.

SQLite cannot change a unique constraint in place, so the table is rebuilt inside one
transaction: create the new table, copy every row (``period_end`` = old ``filing_date``,
``filing_date`` = NULL), drop the old table and rename the new one. Idempotent: when
``period_end`` already exists, or the table does not exist yet, it does nothing.
``financial_data_availability`` is not touched.
"""
import logging

from sqlalchemy import MetaData, Table, inspect, text
from sqlalchemy.engine import Engine

from investbot.data.models import Base, Financial
from investbot.scoring.fundamental.point_in_time import available_from

logger = logging.getLogger(__name__)

TABLE = "financials"
_TMP_TABLE = "financials_pit_new"


def needs_migration(engine: Engine) -> bool:
    """True when ``financials`` exists with the old schema (no ``period_end`` column)."""
    insp = inspect(engine)
    if not insp.has_table(TABLE):
        return False
    return "period_end" not in {c["name"] for c in insp.get_columns(TABLE)}


def _new_table_definition() -> Table:
    """The current ``Financial`` table under a temporary name.

    The copy lives in a private MetaData holding every model table, so its foreign key
    to ``assets`` resolves without registering the temporary table on ``Base.metadata``.
    """
    md = MetaData()
    for t in Base.metadata.sorted_tables:
        t.to_metadata(md)
    return Financial.__table__.to_metadata(md, name=_TMP_TABLE)


def migrate_financials_point_in_time(engine: Engine) -> dict[str, int]:
    """Rebuild an old-schema ``financials`` table under the point-in-time schema.

    Returns ``{"rows_copied": n, "rows_skipped": k}``. Every row is copied; ``rows_skipped``
    counts the copied rows whose availability could not be derived (NULL old
    ``filing_date`` or an unknown ``period_type``). They keep ``available_from`` NULL, so
    ``financials_as_of`` never returns them. A no-op returns zeros.
    """
    counts = {"rows_copied": 0, "rows_skipped": 0}
    if not needs_migration(engine):
        return counts

    with engine.connect() as conn:
        if conn.dialect.name == "sqlite":
            # pysqlite opens a transaction only before DML, so CREATE TABLE would autocommit.
            # An explicit BEGIN makes the whole rebuild roll back as one unit.
            conn.exec_driver_sql("BEGIN")
        old = Table(TABLE, MetaData(), autoload_with=conn, resolve_fks=False)
        new = _new_table_definition()
        new.drop(conn, checkfirst=True)
        new.create(conn)

        shared = [c.name for c in new.columns if c.name in old.columns]
        rows = []
        for r in conn.execute(old.select()).mappings():
            row = {name: r[name] for name in shared}
            period_end = r["filing_date"]   # the old column held the period end
            row["period_end"] = period_end
            row["filing_date"] = None
            row["available_from"] = None
            row["available_from_source"] = None
            if period_end is not None and r["period_type"] in ("annual", "quarterly"):
                row["available_from"], row["available_from_source"] = available_from(
                    period_end, r["period_type"]
                )
            else:
                counts["rows_skipped"] += 1
            rows.append(row)

        if rows:
            conn.execute(new.insert(), rows)
        counts["rows_copied"] = len(rows)

        n_new = conn.execute(text(f"SELECT COUNT(*) FROM {_TMP_TABLE}")).scalar_one()
        n_old = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE}")).scalar_one()
        if n_new != n_old:
            raise RuntimeError(f"financials migration copied {n_new} of {n_old} rows; rolled back")

        conn.execute(text(f"DROP TABLE {TABLE}"))
        conn.execute(text(f"ALTER TABLE {_TMP_TABLE} RENAME TO {TABLE}"))
        conn.commit()

    logger.info(
        "financials migrated to point-in-time schema: %d rows copied, %d without availability",
        counts["rows_copied"], counts["rows_skipped"],
    )
    return counts
