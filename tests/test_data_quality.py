"""
Data-quality tests: row validation, ingestion filtering, the DB audit and `main.py dq`.

Everything runs against an in-memory SQLite and a stubbed yfinance download,
so these need neither the real DB nor the network.
"""
from __future__ import annotations

import logging
import sys
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from investbot.data.ingestion import prices as prices_mod
from investbot.data.models import Asset, Base, PriceDaily, PriceIntraday
from investbot.data.quality import (
    INCONSISTENT_OHLC, MISSING_OHLC, MISSING_VOLUME, NEGATIVE_VOLUME,
    NON_POSITIVE_PRICE, ZERO_VOLUME_FLAT,
    audit_prices, purge_bad_rows, validate_price_row,
)

NAN = float("nan")
GOOD = (100.0, 105.0, 99.0, 103.0, 1_000)       # open, high, low, close, volume


# ──────────────────────────────────────────────────────────
# validate_price_row
# ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("bar, daily, expected", [
    (GOOD, True, None),
    (GOOD, False, None),
    # Incomplete OHLC: the production failure (NaN close, volume present).
    ((100.0, 105.0, 99.0, NAN, 1_000), True, MISSING_OHLC),
    ((None, 105.0, 99.0, 103.0, 1_000), True, MISSING_OHLC),
    ((100.0, float("inf"), 99.0, 103.0, 1_000), True, MISSING_OHLC),
    ((100.0, 105.0, 99.0, pd.NA, 1_000), True, MISSING_OHLC),
    ((100.0, 105.0, 99.0, NAN, 1_000), False, MISSING_OHLC),
    # Non-positive prices.
    ((0.0, 105.0, 99.0, 103.0, 1_000), True, NON_POSITIVE_PRICE),
    ((100.0, 105.0, -1.0, 103.0, 1_000), True, NON_POSITIVE_PRICE),
    # High/low must bracket open and close.
    ((100.0, 102.0, 99.0, 103.0, 1_000), True, INCONSISTENT_OHLC),    # close > high
    ((100.0, 105.0, 101.0, 103.0, 1_000), True, INCONSISTENT_OHLC),   # open < low
    ((100.0, 98.0, 99.0, 99.0, 1_000), True, INCONSISTENT_OHLC),      # high < low
    # float32-scale rounding noise is tolerated.
    ((100.0, 105.0, 99.0, 105.0 * (1 + 1e-9), 1_000), True, None),
    # Volume.
    ((100.0, 105.0, 99.0, 103.0, None), True, MISSING_VOLUME),
    ((100.0, 105.0, 99.0, 103.0, NAN), True, MISSING_VOLUME),
    ((100.0, 105.0, 99.0, 103.0, -5), True, NEGATIVE_VOLUME),
    # yfinance carry-forward placeholder: flat bar, no volume. Daily only.
    ((50.0, 50.0, 50.0, 50.0, 0), True, ZERO_VOLUME_FLAT),
    ((50.0, 50.0, 50.0, 50.0, 0), False, None),
    # A zero-volume bar with a real range is not the placeholder pattern.
    ((100.0, 105.0, 99.0, 103.0, 0), True, None),
])
def test_validate_price_row(bar, daily, expected):
    o, h, l, c, v = bar
    assert validate_price_row(o, h, l, c, v, daily=daily) == expected


# ──────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────

@pytest.fixture
def Session():
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, expire_on_commit=False)
    engine.dispose()


@pytest.fixture
def session(Session):
    s = Session()
    yield s
    s.close()


@pytest.fixture
def asset(session):
    a = Asset(exchange_id=1, ticker="TEST.NS", name="Test Co", asset_type="stock")
    session.add(a)
    session.commit()
    return a


def _yf_daily(rows: list[tuple]) -> pd.DataFrame:
    """rows: (date, open, high, low, close, adj_close, volume), yfinance column names."""
    idx = pd.DatetimeIndex([r[0] for r in rows])
    cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    return pd.DataFrame([r[1:] for r in rows], index=idx, columns=cols)


def _stored_dates(session, asset) -> list[date]:
    return list(session.execute(
        select(PriceDaily.date).where(PriceDaily.asset_id == asset.id).order_by(PriceDaily.date)
    ).scalars())


# ──────────────────────────────────────────────────────────
# Ingestion
# ──────────────────────────────────────────────────────────

def test_daily_ingestion_rejects_bad_rows_and_refetches_them(session, asset, monkeypatch, caplog):
    frame = _yf_daily([
        ("2026-10-12", 100, 105, 99, 103, 103, 1_000),
        ("2026-10-13", 103, 106, 101, 104, 104, 1_100),
        ("2026-10-14", 50, 50, 50, 50, 50, 0),            # zero-volume placeholder
        ("2026-10-15", 104, 100, 105, 103, 103, 900),     # high < low
        ("2026-10-16", 104, 108, 103, NAN, NAN, 1_200),   # NaN close: the production case
    ])
    monkeypatch.setattr(prices_mod, "_download_prices", lambda *a, **k: frame)

    with caplog.at_level(logging.WARNING, logger=prices_mod.logger.name):
        added = prices_mod.ingest_daily_prices(session, asset)

    assert added == 2
    assert _stored_dates(session, asset) == [date(2026, 10, 12), date(2026, 10, 13)]
    rows = session.execute(select(PriceDaily)).scalars().all()
    assert all(r.close is not None and r.open is not None for r in rows)

    msg = next(r.getMessage() for r in caplog.records if "rejected" in r.getMessage())
    assert "3 daily bar(s)" in msg
    assert MISSING_OHLC in msg and ZERO_VOLUME_FLAT in msg and INCONSISTENT_OHLC in msg

    # Rejected dates were never stored, so a later run picks them up once yfinance has complete data.
    fixed = _yf_daily([
        ("2026-10-12", 100, 105, 99, 103, 103, 1_000),
        ("2026-10-13", 103, 106, 101, 104, 104, 1_100),
        ("2026-10-16", 104, 108, 103, 107, 107, 1_200),
    ])
    monkeypatch.setattr(prices_mod, "_download_prices", lambda *a, **k: fixed)
    assert prices_mod.ingest_daily_prices(session, asset) == 1
    assert _stored_dates(session, asset) == [date(2026, 10, 12), date(2026, 10, 13), date(2026, 10, 16)]


def test_daily_ingestion_handles_yfinance_multiindex_columns(session, asset, monkeypatch):
    frame = _yf_daily([("2026-10-12", 100, 105, 99, 103, 103, 1_000)])
    frame.columns = pd.MultiIndex.from_product([frame.columns, ["TEST.NS"]])
    monkeypatch.setattr(prices_mod, "_download_prices", lambda *a, **k: frame)
    assert prices_mod.ingest_daily_prices(session, asset) == 1


def test_intraday_ingestion_rejects_incomplete_bars_but_keeps_flat_zero_volume(
    session, asset, monkeypatch
):
    idx = pd.DatetimeIndex(
        ["2026-10-16 09:15", "2026-10-16 09:30", "2026-10-16 09:45"], tz="Asia/Kolkata"
    )
    frame = pd.DataFrame(
        {
            "Open":   [100.0, 101.0, 102.0],
            "High":   [101.0, 101.0, 103.0],
            "Low":    [99.5,  101.0, 101.5],
            "Close":  [100.5, 101.0, NAN],      # 09:30 flat + zero volume (kept), 09:45 NaN close (dropped)
            "Volume": [500,   0,     400],
        },
        index=idx,
    )
    monkeypatch.setattr(prices_mod, "_download_prices", lambda *a, **k: frame)

    assert prices_mod.ingest_intraday_prices(session, asset) == 2
    stored = session.execute(select(PriceIntraday.timestamp).order_by(PriceIntraday.timestamp)).scalars().all()
    assert stored == [datetime(2026, 10, 16, 9, 15), datetime(2026, 10, 16, 9, 30)]


# ──────────────────────────────────────────────────────────
# Audit / purge
# ──────────────────────────────────────────────────────────

def _seed_mixed_rows(session, asset):
    session.add_all([
        PriceDaily(asset_id=asset.id, date=date(2026, 10, 14), open=100, high=105, low=99, close=103, adj_close=103, volume=10),
        PriceDaily(asset_id=asset.id, date=date(2026, 10, 15), open=50, high=50, low=50, close=50, adj_close=50, volume=0),
        PriceDaily(asset_id=asset.id, date=date(2026, 10, 16), open=100, high=105, low=99, close=None, adj_close=None, volume=10),
        PriceIntraday(asset_id=asset.id, timestamp=datetime(2026, 10, 16, 9, 15), interval="15m", open=100, high=101, low=99, close=100, volume=5),
        PriceIntraday(asset_id=asset.id, timestamp=datetime(2026, 10, 16, 9, 30), interval="15m", open=100, high=101, low=99, close=None, volume=5),
    ])
    session.commit()


def test_audit_reports_every_bad_row_with_reason(session, asset):
    _seed_mixed_rows(session, asset)
    bad = audit_prices(session)

    got = {(b.table, b.when, b.reason) for b in bad}
    assert got == {
        ("price_daily", date(2026, 10, 15), ZERO_VOLUME_FLAT),
        ("price_daily", date(2026, 10, 16), MISSING_OHLC),
        ("price_intraday", datetime(2026, 10, 16, 9, 30), MISSING_OHLC),
    }
    assert {b.ticker for b in bad} == {"TEST.NS"}


def test_purge_deletes_only_bad_rows(session, asset):
    _seed_mixed_rows(session, asset)
    assert purge_bad_rows(session, audit_prices(session)) == 3

    assert _stored_dates(session, asset) == [date(2026, 10, 14)]
    assert session.query(PriceIntraday).count() == 1
    assert audit_prices(session) == []


# ──────────────────────────────────────────────────────────
# main.py dq
# ──────────────────────────────────────────────────────────

@pytest.fixture
def main_mod(Session, monkeypatch):
    import main
    monkeypatch.setattr(main, "SessionLocal", Session)
    return main


def test_dq_command_exits_nonzero_on_bad_rows_and_purge_cleans(main_mod, Session, asset, monkeypatch, capsys):
    with Session() as s:
        _seed_mixed_rows(s, asset)

    monkeypatch.setattr(sys, "argv", ["main.py", "dq"])
    with pytest.raises(SystemExit) as exc:
        main_mod.cmd_dq()
    out = capsys.readouterr().out
    assert exc.value.code == 1
    assert "3 bad price row(s)" in out and MISSING_OHLC in out and ZERO_VOLUME_FLAT in out
    with Session() as s:
        assert s.query(PriceDaily).count() == 3            # read-only without --purge

    monkeypatch.setattr(sys, "argv", ["main.py", "dq", "--purge"])
    main_mod.cmd_dq()
    assert "Purged 3 row(s)" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["main.py", "dq"])
    main_mod.cmd_dq()                                       # clean now: returns without exiting
    assert "Data quality OK" in capsys.readouterr().out
