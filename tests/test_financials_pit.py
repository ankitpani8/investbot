"""
Point-in-time financials (task 410): `available_from`, `financials_as_of`, the
`financials_pit` migration and period_end-keyed ingestion.

SQLite in memory / tmp_path and a mocked yf.Ticker; no network, no real DB.
"""
from __future__ import annotations

import ast
import inspect as pyinspect
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from investbot.data.ingestion import financials as fin_mod
from investbot.data.migrations import financials_pit
from investbot.data.migrations.financials_pit import migrate_financials_point_in_time
from investbot.data.models import Asset, Base, Financial
from investbot.scoring.fundamental import point_in_time
from investbot.scoring.fundamental.point_in_time import (
    ANNUAL_LAG_DAYS,
    FISCAL_YEAR_END_MONTH,
    QUARTERLY_LAG_DAYS,
    available_from,
    financials_as_of,
)


# ── available_from ──

def test_constants():
    assert (QUARTERLY_LAG_DAYS, ANNUAL_LAG_DAYS, FISCAL_YEAR_END_MONTH) == (45, 60, 3)


def test_annual_lag_60():
    assert available_from(date(2025, 3, 31), "annual") == (date(2025, 5, 30), "lodr_lag")


@pytest.mark.parametrize("pe", [date(2025, 6, 30), date(2025, 9, 30), date(2025, 12, 31)])
def test_quarterly_lag_45(pe):
    assert available_from(pe, "quarterly") == (pe + timedelta(days=45), "lodr_lag")


def test_march_quarter_gets_annual_lag():
    # Q4 is published with the audited annual results.
    assert available_from(date(2026, 3, 31), "quarterly") == (date(2026, 5, 30), "lodr_lag")


def test_published_filing_date_wins():
    assert available_from(date(2025, 6, 30), "quarterly", date(2025, 7, 20)) == (
        date(2025, 7, 20), "published",
    )
    # Even a filing later than the LODR deadline is used as is.
    assert available_from(date(2025, 3, 31), "annual", date(2025, 6, 15)) == (
        date(2025, 6, 15), "published",
    )


def test_accepts_timestamps():
    assert available_from(pd.Timestamp("2025-06-30"), "quarterly") == (date(2025, 8, 14), "lodr_lag")


def test_filing_before_period_end_raises():
    with pytest.raises(ValueError):
        available_from(date(2025, 6, 30), "quarterly", date(2025, 6, 29))


@pytest.mark.parametrize("pt", ["monthly", "Annual", "", None])
def test_unknown_period_type_raises(pt):
    with pytest.raises(ValueError):
        available_from(date(2025, 6, 30), pt)


# ── financials_as_of ──

def _fin_frame() -> pd.DataFrame:
    """Unsorted mix of quarterly/annual rows with a NaN line item and an undated row."""
    rows = []
    for pe, pt, filing in [
        (date(2025, 12, 31), "quarterly", None),
        (date(2024, 3, 31), "annual", None),
        (date(2025, 3, 31), "quarterly", None),
        (date(2025, 6, 30), "quarterly", date(2025, 7, 25)),
        (date(2025, 3, 31), "annual", None),
        (date(2025, 9, 30), "quarterly", None),
        (date(2024, 12, 31), "quarterly", None),
    ]:
        avail, src = available_from(pe, pt, filing)
        rows.append({"period_end": pe, "period_type": pt, "available_from": avail,
                     "available_from_source": src, "revenue": float(pe.month * 100)})
    fin = pd.DataFrame(rows, index=[10, 11, 12, 13, 14, 15, 16])
    fin.loc[15, "revenue"] = np.nan
    undated = pd.DataFrame([{"period_end": None, "period_type": "quarterly", "available_from": None,
                             "available_from_source": None, "revenue": 1.0}], index=[17])
    return pd.concat([fin, undated])


def test_as_of_filters_and_sorts():
    fin = _fin_frame()
    out = financials_as_of(fin, date(2025, 11, 14))
    assert list(out["period_end"]) == [
        date(2024, 3, 31), date(2024, 12, 31), date(2025, 3, 31), date(2025, 3, 31),
        date(2025, 6, 30), date(2025, 9, 30),
    ]
    assert list(out.columns) == list(fin.columns)
    assert (out.dtypes == fin.dtypes).all()
    # NaN line item stays in; the row with no available_from never shows up.
    assert out["revenue"].isna().sum() == 1
    assert 17 not in out.index


def test_as_of_boundary_is_inclusive():
    fin = _fin_frame()
    assert date(2025, 9, 30) not in set(financials_as_of(fin, date(2025, 11, 13))["period_end"])
    assert date(2025, 9, 30) in set(financials_as_of(fin, date(2025, 11, 14))["period_end"])
    assert financials_as_of(fin, date(2024, 1, 1)).empty


def test_as_of_accepts_datetime_columns():
    fin = _fin_frame().iloc[:-1]
    fin = fin.assign(available_from=pd.to_datetime(fin["available_from"]),
                     period_end=pd.to_datetime(fin["period_end"]))
    out = financials_as_of(fin, date(2025, 11, 14))
    assert len(out) == 6
    assert out["period_end"].is_monotonic_increasing


def test_as_of_no_lookahead_truncation_invariance():
    """No row available after as_of is returned, and rows not yet public cannot change the answer."""
    fin = _fin_frame()
    avail = pd.to_datetime(fin["available_from"])
    for as_of in pd.date_range("2024-01-01", "2026-06-30", freq="D").date:
        out = financials_as_of(fin, as_of)
        assert (pd.to_datetime(out["available_from"]) <= pd.Timestamp(as_of)).all()
        truncated = fin.loc[(avail <= pd.Timestamp(as_of)).to_numpy()]
        pd.testing.assert_frame_equal(out, financials_as_of(truncated, as_of))
        # Rows from the future with arbitrary numbers do not leak in.
        poisoned = fin.copy()
        poisoned.loc[(avail > pd.Timestamp(as_of)).to_numpy(), "revenue"] = -1e12
        pd.testing.assert_frame_equal(out, financials_as_of(poisoned, as_of))


def test_point_in_time_module_is_pure():
    tree = ast.parse(pyinspect.getsource(point_in_time))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any(m.startswith(("sqlalchemy", "sqlite3", "investbot.data")) for m in imported)
    assert "dropna" not in pyinspect.getsource(point_in_time)


# ── migration ──

OLD_FINANCIALS_DDL = """
CREATE TABLE financials (
    id INTEGER NOT NULL,
    asset_id INTEGER NOT NULL,
    fiscal_year INTEGER NOT NULL,
    period_type VARCHAR,
    filing_date DATE,
    {line_items},
    PRIMARY KEY (id),
    UNIQUE (asset_id, fiscal_year, period_type),
    FOREIGN KEY(asset_id) REFERENCES assets (id)
)
"""

_NEW_ONLY = {"period_end", "available_from", "available_from_source"}
_KEY_COLS = {"id", "asset_id", "fiscal_year", "period_type", "filing_date"}
LINE_ITEMS = [c.name for c in Financial.__table__.columns if c.name not in _NEW_ONLY | _KEY_COLS]

# (id, asset_id, fiscal_year, period_type, old filing_date = period end, revenue)
OLD_ROWS = [
    (1, 1, 2025, "annual", "2025-03-31", 100.0),
    (2, 1, 2026, "annual", "2026-03-31", 110.0),
    (3, 1, 2025, "quarterly", "2025-03-31", 25.0),
    (4, 1, 2026, "quarterly", "2026-03-31", None),
    (5, 2, 2021, "annual", "2021-12-31", 50.0),
    (6, 2, 2025, "quarterly", "2025-12-31", 13.0),
]


def _old_schema_engine(url: str = "sqlite://"):
    engine = create_engine(url, future=True)
    Base.metadata.create_all(engine, tables=[Asset.__table__])   # SQLite does not enforce the FKs
    with engine.begin() as conn:
        conn.execute(text(OLD_FINANCIALS_DDL.format(
            line_items=",\n    ".join(f"{c} FLOAT" for c in LINE_ITEMS))))
        conn.execute(text("INSERT INTO assets (id, exchange_id, ticker, name, asset_type) VALUES "
                          "(1, 1, 'AAA.NS', 'A', 'stock'), (2, 1, 'BBB.NS', 'B', 'stock')"))
        for r in OLD_ROWS:
            conn.execute(text(
                "INSERT INTO financials (id, asset_id, fiscal_year, period_type, filing_date, "
                "revenue, net_income) VALUES (:i, :a, :fy, :pt, :fd, :rev, :ni)"
            ), {"i": r[0], "a": r[1], "fy": r[2], "pt": r[3], "fd": r[4], "rev": r[5], "ni": 7.0})
    return engine


def _dump(engine) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text("SELECT * FROM financials ORDER BY id"), conn)


def test_migration_rebuilds_old_schema_and_is_idempotent(tmp_path):
    engine = _old_schema_engine(f"sqlite:///{tmp_path / 'old.db'}")
    assert financials_pit.needs_migration(engine)

    counts = migrate_financials_point_in_time(engine)
    assert counts == {"rows_copied": len(OLD_ROWS), "rows_skipped": 0}
    assert not financials_pit.needs_migration(engine)

    df = _dump(engine)
    assert len(df) == len(OLD_ROWS)
    assert list(df["period_end"]) == [r[4] for r in OLD_ROWS]
    assert df["filing_date"].isna().all()
    assert list(df["available_from"]) == [
        "2025-05-30", "2026-05-30", "2025-05-30", "2026-05-30", "2022-03-01", "2026-02-14",
    ]
    assert (df["available_from_source"] == "lodr_lag").all()
    assert list(df["fiscal_year"]) == [r[2] for r in OLD_ROWS]
    assert list(df["revenue"].fillna(-1)) == [r[5] if r[5] is not None else -1 for r in OLD_ROWS]
    assert (df["net_income"] == 7.0).all()

    # New unique key: a second quarter in the same calendar year now fits.
    uniques = inspect(engine).get_unique_constraints("financials")
    assert [sorted(u["column_names"]) for u in uniques] == [["asset_id", "period_end", "period_type"]]
    Session = sessionmaker(bind=engine)
    with Session() as s:
        s.add(Financial(asset_id=1, fiscal_year=2025, period_type="quarterly",
                        period_end=date(2025, 6, 30)))
        s.commit()
    assert set(inspect(engine).get_table_names()) == {"assets", "financials"}

    before = _dump(engine)
    assert migrate_financials_point_in_time(engine) == {"rows_copied": 0, "rows_skipped": 0}
    pd.testing.assert_frame_equal(before, _dump(engine))


def test_migration_noop_on_new_schema_and_missing_table():
    engine = create_engine("sqlite://", future=True)
    assert migrate_financials_point_in_time(engine) == {"rows_copied": 0, "rows_skipped": 0}
    Base.metadata.create_all(engine)
    assert migrate_financials_point_in_time(engine) == {"rows_copied": 0, "rows_skipped": 0}


def test_migration_keeps_undatable_rows_without_availability():
    engine = _old_schema_engine()
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO financials (id, asset_id, fiscal_year, period_type, filing_date)"
                          " VALUES (99, 2, 2019, 'annual', NULL)"))
    assert migrate_financials_point_in_time(engine) == {"rows_copied": len(OLD_ROWS) + 1, "rows_skipped": 1}
    row = _dump(engine).set_index("id").loc[99]
    assert row["period_end"] is None and row["available_from"] is None


def test_migration_failure_rolls_back_everything(tmp_path, monkeypatch):
    engine = _old_schema_engine(f"sqlite:///{tmp_path / 'old.db'}")
    before = _dump(engine)

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(financials_pit, "available_from", boom)
    with pytest.raises(RuntimeError, match="boom"):
        migrate_financials_point_in_time(engine)

    assert set(inspect(engine).get_table_names()) == {"assets", "financials"}
    assert financials_pit.needs_migration(engine)
    pd.testing.assert_frame_equal(before, _dump(engine))


def test_init_db_runs_the_migration(monkeypatch):
    from investbot.data import db

    engine = _old_schema_engine()
    monkeypatch.setattr(db, "engine", engine)
    db.init_db()
    assert not financials_pit.needs_migration(engine)
    assert len(_dump(engine)) == len(OLD_ROWS)


# ── ingestion ──

QUARTER_ENDS = pd.to_datetime(["2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31", "2025-03-31"])


def _stmt(labels: dict[str, float], dates) -> pd.DataFrame:
    return pd.DataFrame({d: {k: v * (i + 1) for k, v in labels.items()} for i, d in enumerate(dates)})


class FakeTicker:
    def __init__(self, ticker):
        self.ticker = ticker
        self.quarterly_financials = _stmt({"Total Revenue": 100.0, "Net Income": 10.0}, QUARTER_ENDS)
        self.quarterly_balance_sheet = _stmt({"Total Assets": 1000.0}, QUARTER_ENDS[:4])
        self.quarterly_cashflow = _stmt({"Operating Cash Flow": 20.0}, QUARTER_ENDS)
        self.financials = _stmt({"Total Revenue": 400.0}, pd.to_datetime(["2025-03-31", "2026-03-31"]))
        self.balance_sheet = pd.DataFrame()
        self.cashflow = pd.DataFrame()


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(fin_mod.yf, "Ticker", FakeTicker)
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    s.add(Asset(exchange_id=1, ticker="AAA.NS", name="A", asset_type="stock", is_active=True))
    s.commit()
    yield s
    s.close()
    engine.dispose()


def test_ingest_stores_every_quarter_and_is_idempotent(session):
    asset = session.query(Asset).one()
    assert fin_mod.ingest_financials_for_asset(session, asset, "quarterly") == 5
    assert fin_mod.ingest_financials_for_asset(session, asset, "quarterly") == 0

    rows = session.query(Financial).order_by(Financial.period_end).all()
    assert [r.period_end for r in rows] == sorted(d.date() for d in QUARTER_ENDS)
    assert {r.period_type for r in rows} == {"quarterly"}
    assert all(r.filing_date is None for r in rows)
    assert [r.fiscal_year for r in rows] == [r.period_end.year for r in rows]
    assert all(r.available_from_source == "lodr_lag" for r in rows)
    assert [r.available_from for r in rows] == [
        date(2025, 5, 30), date(2025, 8, 14), date(2025, 11, 14), date(2026, 2, 14), date(2026, 5, 30),
    ]
    by_end = {r.period_end: r for r in rows}
    assert by_end[date(2025, 3, 31)].revenue == 500.0       # 5th column of the income statement
    assert by_end[date(2025, 3, 31)].total_assets is None   # absent from the balance sheet


def test_ingest_annual_and_quarterly_share_a_period_end(session):
    asset = session.query(Asset).one()
    assert fin_mod.ingest_financials_for_asset(session, asset, "annual") == 2
    assert fin_mod.ingest_financials_for_asset(session, asset, "quarterly") == 5
    assert fin_mod.ingest_financials_for_asset(session, asset, "annual") == 0
    annual = session.query(Financial).filter_by(period_type="annual").order_by(Financial.period_end).all()
    assert [(r.period_end, r.available_from) for r in annual] == [
        (date(2025, 3, 31), date(2025, 5, 30)), (date(2026, 3, 31), date(2026, 5, 30)),
    ]
