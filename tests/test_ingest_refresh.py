"""
Trailing-window refresh: revised bars heal, bad bars never overwrite good ones,
unfinished bars (today's daily before the close, the forming 15m candle) are
not stored, and a repeated run is a no-op.

In-memory SQLite plus a stubbed yfinance download; no network, no real DB.
"""
from __future__ import annotations

import logging
from datetime import date, datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from investbot.config import settings
from investbot.data.ingestion import prices as prices_mod
from investbot.data.models import Asset, Base, PriceDaily, PriceIntraday

NAN = float("nan")
# Saturday after the week of 2026-10-12, so every fixture bar is in the past.
WEEKEND = datetime(2026, 10, 17, 12, 0)


@pytest.fixture
def session():
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    yield s
    s.close()
    engine.dispose()


@pytest.fixture
def asset(session):
    a = Asset(exchange_id=1, ticker="TEST.NS", name="Test Co", asset_type="stock")
    session.add(a)
    session.commit()
    return a


def _stub(monkeypatch, frame: pd.DataFrame) -> None:
    monkeypatch.setattr(prices_mod, "_download_prices", lambda *a, **k: frame.copy())


def _daily(rows: list[tuple]) -> pd.DataFrame:
    """rows: (date, open, high, low, close, volume); adj_close = close."""
    idx = pd.DatetimeIndex([r[0] for r in rows])
    return pd.DataFrame(
        {
            "Open": [r[1] for r in rows], "High": [r[2] for r in rows],
            "Low": [r[3] for r in rows], "Close": [r[4] for r in rows],
            "Adj Close": [r[4] for r in rows], "Volume": [r[5] for r in rows],
        },
        index=idx,
    )


def _week(**overrides) -> pd.DataFrame:
    """Ten sessions 2026-10-05..16; overrides map 'YYYY-MM-DD' -> (o, h, l, c, v)."""
    days = pd.bdate_range("2026-10-05", "2026-10-16")
    rows = []
    for i, d in enumerate(days):
        key = d.strftime("%Y-%m-%d")
        bar = overrides.get(key, (100 + i, 105 + i, 99 + i, 103 + i, 1_000 + i))
        rows.append((key, *bar))
    return _daily(rows)


def _daily_rows(session) -> dict[date, tuple]:
    rows = session.execute(select(PriceDaily).order_by(PriceDaily.date)).scalars()
    return {r.date: (r.open, r.high, r.low, r.close, r.adj_close, r.volume) for r in rows}


def _intraday_rows(session) -> dict[datetime, tuple]:
    rows = session.execute(select(PriceIntraday).order_by(PriceIntraday.timestamp)).scalars()
    return {r.timestamp: (r.open, r.high, r.low, r.close, r.volume) for r in rows}


def test_settings_defaults():
    assert settings.DAILY_REFRESH_SESSIONS == 5
    assert settings.INTRADAY_REFRESH_DAYS == 2
    assert settings.DAILY_BAR_FINAL_AFTER == "16:00"


# ──────────────────────────────────────────────────────────
# Daily
# ──────────────────────────────────────────────────────────

def test_revised_daily_bar_in_window_is_updated(session, asset, monkeypatch):
    _stub(monkeypatch, _week())
    assert prices_mod.ingest_daily_prices(session, asset, now=WEEKEND) == 10

    revised = (200.0, 210.0, 190.0, 205.0, 5_000)
    _stub(monkeypatch, _week(**{"2026-10-16": revised, "2026-10-12": revised}))
    assert prices_mod.ingest_daily_prices(session, asset, now=WEEKEND) == 2

    stored = _daily_rows(session)
    assert len(stored) == 10
    assert stored[date(2026, 10, 16)] == (200.0, 210.0, 190.0, 205.0, 205.0, 5_000)
    # 2026-10-12 is the 5th-last session: inside the window.
    assert stored[date(2026, 10, 12)] == (200.0, 210.0, 190.0, 205.0, 205.0, 5_000)


def test_revision_outside_window_is_ignored(session, asset, monkeypatch):
    _stub(monkeypatch, _week())
    prices_mod.ingest_daily_prices(session, asset, now=WEEKEND)
    before = _daily_rows(session)

    # 2026-10-09 is the 6th-last session: older history is insert-only.
    _stub(monkeypatch, _week(**{"2026-10-09": (200.0, 210.0, 190.0, 205.0, 5_000)}))
    assert prices_mod.ingest_daily_prices(session, asset, now=WEEKEND) == 0
    assert _daily_rows(session) == before


def test_missing_bar_inside_and_outside_window_is_inserted(session, asset, monkeypatch):
    full = _week()
    _stub(monkeypatch, full.drop(pd.Timestamp("2026-10-06")).drop(pd.Timestamp("2026-10-15")))
    assert prices_mod.ingest_daily_prices(session, asset, now=WEEKEND) == 8

    _stub(monkeypatch, full)
    assert prices_mod.ingest_daily_prices(session, asset, now=WEEKEND) == 2
    assert len(_daily_rows(session)) == 10


def test_null_close_bar_does_not_overwrite_good_row(session, asset, monkeypatch, caplog):
    _stub(monkeypatch, _week())
    prices_mod.ingest_daily_prices(session, asset, now=WEEKEND)
    before = _daily_rows(session)

    _stub(monkeypatch, _week(**{"2026-10-16": (104.0, 108.0, 103.0, NAN, 1_200)}))
    with caplog.at_level(logging.INFO, logger=prices_mod.logger.name):
        assert prices_mod.ingest_daily_prices(session, asset, now=WEEKEND) == 0

    assert _daily_rows(session) == before
    assert any("inserted 0, updated 0, rejected 1" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("now, stored", [
    (datetime(2026, 10, 16, 9, 30), False),
    (datetime(2026, 10, 16, 15, 59), False),
    (datetime(2026, 10, 16, 16, 0), True),
    (datetime(2026, 10, 16, 18, 0), True),
    # Clock in UTC: 10:00 UTC is 15:30 IST (skip), 10:31 UTC is 16:01 IST (store).
    (pd.Timestamp("2026-10-16 10:00", tz="UTC").to_pydatetime(), False),
    (pd.Timestamp("2026-10-16 10:31", tz="UTC").to_pydatetime(), True),
])
def test_todays_daily_bar_waits_for_final_time(session, asset, monkeypatch, now, stored):
    _stub(monkeypatch, _week())
    prices_mod.ingest_daily_prices(session, asset, now=now)
    assert (date(2026, 10, 16) in _daily_rows(session)) is stored
    assert date(2026, 10, 15) in _daily_rows(session)


def test_partial_today_bar_is_stored_after_close(session, asset, monkeypatch):
    """Mid-session run skips today; the after-close run stores the final bar."""
    _stub(monkeypatch, _week(**{"2026-10-16": (110.0, 111.0, 109.0, 110.5, 300)}))
    prices_mod.ingest_daily_prices(session, asset, now=datetime(2026, 10, 16, 11, 0))
    assert date(2026, 10, 16) not in _daily_rows(session)

    _stub(monkeypatch, _week(**{"2026-10-16": (110.0, 115.0, 108.0, 114.0, 9_000)}))
    prices_mod.ingest_daily_prices(session, asset, now=datetime(2026, 10, 16, 16, 30))
    assert _daily_rows(session)[date(2026, 10, 16)] == (110.0, 115.0, 108.0, 114.0, 114.0, 9_000)


def test_daily_second_run_is_noop(session, asset, monkeypatch, caplog):
    frame = _week(**{"2026-10-14": (50, 50, 50, 50, 0)})        # one rejected placeholder too
    _stub(monkeypatch, frame)
    assert prices_mod.ingest_daily_prices(session, asset, now=WEEKEND) == 9
    first = _daily_rows(session)
    ids = session.execute(select(PriceDaily.id).order_by(PriceDaily.id)).scalars().all()

    with caplog.at_level(logging.INFO, logger=prices_mod.logger.name):
        assert prices_mod.ingest_daily_prices(session, asset, now=WEEKEND) == 0
    assert _daily_rows(session) == first
    assert session.execute(select(PriceDaily.id).order_by(PriceDaily.id)).scalars().all() == ids
    assert any("inserted 0, updated 0, rejected 1" in r.getMessage() for r in caplog.records)


# ──────────────────────────────────────────────────────────
# Intraday
# ──────────────────────────────────────────────────────────

def _intraday(days: list[str], slots: int = 4, close_bump: dict | None = None) -> pd.DataFrame:
    """`slots` 15m bars from 09:15 IST on each day, tz-aware like yfinance."""
    close_bump = close_bump or {}
    stamps = [
        pd.Timestamp(f"{d} 09:15", tz="Asia/Kolkata") + pd.Timedelta(minutes=15 * i)
        for d in days for i in range(slots)
    ]
    closes = [100.5 + close_bump.get(ts.strftime("%Y-%m-%d %H:%M"), 0.0) for ts in stamps]
    return pd.DataFrame(
        {
            "Open": 100.0, "High": [max(101.0, c) for c in closes], "Low": 99.5,
            "Close": closes, "Volume": 500,
        },
        index=pd.DatetimeIndex(stamps),
    )


DAYS = ["2026-10-13", "2026-10-14", "2026-10-15"]


def test_forming_15m_candle_is_dropped(session, asset, monkeypatch):
    _stub(monkeypatch, _intraday(["2026-10-16"]))       # 09:15, 09:30, 09:45, 10:00
    # 10:00 IST: the 10:00 candle ends at 10:15, still forming; 09:45 ended exactly now.
    assert prices_mod.ingest_intraday_prices(session, asset, now=datetime(2026, 10, 16, 10, 0)) == 3
    assert max(_intraday_rows(session)) == datetime(2026, 10, 16, 9, 45)

    prices_mod.ingest_intraday_prices(session, asset, now=datetime(2026, 10, 16, 10, 15))
    assert max(_intraday_rows(session)) == datetime(2026, 10, 16, 10, 0)


def test_intraday_utc_index_is_stored_as_ist_wall_time(session, asset, monkeypatch):
    frame = _intraday(["2026-10-15"], slots=1)
    frame.index = frame.index.tz_convert("UTC")
    _stub(monkeypatch, frame)
    prices_mod.ingest_intraday_prices(session, asset, now=WEEKEND)
    assert list(_intraday_rows(session)) == [datetime(2026, 10, 15, 9, 15)]


def test_intraday_revision_in_last_two_days_is_updated(session, asset, monkeypatch):
    _stub(monkeypatch, _intraday(DAYS))
    assert prices_mod.ingest_intraday_prices(session, asset, now=WEEKEND) == 12

    bump = {"2026-10-15 09:30": 1.0, "2026-10-14 09:15": 1.0, "2026-10-13 09:15": 1.0}
    _stub(monkeypatch, _intraday(DAYS, close_bump=bump))
    assert prices_mod.ingest_intraday_prices(session, asset, now=WEEKEND) == 2

    stored = _intraday_rows(session)
    assert stored[datetime(2026, 10, 15, 9, 30)][3] == 101.5
    assert stored[datetime(2026, 10, 14, 9, 15)][3] == 101.5
    assert stored[datetime(2026, 10, 13, 9, 15)][3] == 100.5    # outside window: insert-only


def test_intraday_null_close_does_not_overwrite_good_row(session, asset, monkeypatch):
    _stub(monkeypatch, _intraday(DAYS))
    prices_mod.ingest_intraday_prices(session, asset, now=WEEKEND)
    before = _intraday_rows(session)

    bad = _intraday(DAYS)
    bad.loc[pd.Timestamp("2026-10-15 09:30", tz="Asia/Kolkata"), "Close"] = NAN
    _stub(monkeypatch, bad)
    assert prices_mod.ingest_intraday_prices(session, asset, now=WEEKEND) == 0
    assert _intraday_rows(session) == before


def test_intraday_second_run_is_noop(session, asset, monkeypatch):
    _stub(monkeypatch, _intraday(DAYS))
    prices_mod.ingest_intraday_prices(session, asset, now=WEEKEND)
    first = _intraday_rows(session)
    assert prices_mod.ingest_intraday_prices(session, asset, now=WEEKEND) == 0
    assert _intraday_rows(session) == first
