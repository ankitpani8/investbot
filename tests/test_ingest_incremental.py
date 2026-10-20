"""
Incremental downloads: an asset with stored bars fetches only the overlap window and
ends up with the same rows as a full-period run; thin windows and new splits fall back
to one full-period request; ingest_all_prices reports inserted / updated / rejected.

In-memory SQLite plus a stubbed yfinance download; no network, no real DB.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from investbot.config import settings
from investbot.data.ingestion import prices as prices_mod
from investbot.data.models import Asset, Base, CorporateAction, PriceDaily, PriceIntraday

WEEKEND = datetime(2026, 10, 17, 12, 0)          # IST Saturday: every fixture bar is in the past
DAYS = pd.bdate_range("2026-08-24", "2026-10-16")


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


def _frame(days=DAYS, revisions: dict | None = None, actions: bool = True) -> pd.DataFrame:
    """Daily bars; `revisions` maps 'YYYY-MM-DD' -> new close."""
    n = len(days)
    close = 100.0 + np.arange(n)
    df = pd.DataFrame({
        "Open": close - 1, "High": close + 2, "Low": close - 2, "Close": close,
        "Adj Close": close, "Volume": 1_000 + np.arange(n),
    }, index=pd.DatetimeIndex(days))
    for d, c in (revisions or {}).items():
        df.loc[d, ["High", "Close", "Adj Close"]] = [c + 2, c, c]
    if actions:
        df["Dividends"] = 0.0
        df["Stock Splits"] = 0.0
    return df


class Server:
    """Fake yfinance: serves `frame` (sliced by `start`), records every request."""

    def __init__(self, frame, full=None):
        self.frame = frame
        self.full = full if full is not None else frame
        self.calls: list[dict] = []

    def __call__(self, ticker, period, interval, actions=False, start=None):
        self.calls.append(dict(ticker=ticker, period=period, interval=interval,
                               actions=actions, start=start))
        if start is None:
            return self.full.copy()
        return self.frame[self.frame.index >= pd.Timestamp(start)].copy()


def _serve(monkeypatch, frame, full=None) -> Server:
    server = Server(frame, full)
    monkeypatch.setattr(prices_mod, "_download_prices", server)
    return server


def _daily_rows(session) -> dict[date, tuple]:
    rows = session.execute(select(PriceDaily).order_by(PriceDaily.date)).scalars()
    return {r.date: (r.open, r.high, r.low, r.close, r.adj_close, r.volume) for r in rows}


def _actions(session) -> dict:
    return {(r.ex_date, r.action_type): r.value
            for r in session.execute(select(CorporateAction)).scalars()}


def _seed(session, asset, monkeypatch, frame):
    """Store `frame` through the normal (full-period, empty DB) path."""
    _serve(monkeypatch, frame)
    prices_mod.ingest_daily_prices(session, asset, now=WEEKEND)


# ──────────────────────────────────────────────────────────
# Settings and request arguments
# ──────────────────────────────────────────────────────────

def test_overlap_settings():
    assert settings.DAILY_INCREMENTAL_OVERLAP_DAYS == 14
    assert settings.INTRADAY_INCREMENTAL_OVERLAP_DAYS == 5
    assert settings.INTRADAY_MAX_LOOKBACK_DAYS == 60


def test_no_stored_rows_requests_the_full_period(session, asset, monkeypatch):
    server = _serve(monkeypatch, _frame())
    prices_mod.ingest_daily_prices(session, asset, now=WEEKEND)
    assert server.calls == [dict(ticker="TEST.NS", period=settings.DAILY_PERIOD, interval="1d",
                                 actions=True, start=None)]


def test_stored_rows_request_last_bar_minus_overlap(session, asset, monkeypatch):
    _seed(session, asset, monkeypatch, _frame(DAYS[DAYS <= "2026-10-09"]))
    server = _serve(monkeypatch, _frame())
    prices_mod.ingest_daily_prices(session, asset, now=WEEKEND)
    assert len(server.calls) == 1
    assert server.calls[0]["start"] == date(2026, 10, 9) - timedelta(days=14)
    assert server.calls[0]["interval"] == "1d" and server.calls[0]["actions"] is True


def test_download_prices_passes_start_instead_of_period(monkeypatch):
    seen = {}

    def fake_download(ticker, **kwargs):
        seen.update(kwargs)
        return pd.DataFrame({"Close": [1.0]})

    monkeypatch.setattr(prices_mod.yf, "download", fake_download)
    prices_mod._download_prices("X.NS", "5y", "1d", start=date(2026, 9, 1))
    assert seen["start"] == date(2026, 9, 1) and "period" not in seen
    assert seen["auto_adjust"] is False

    seen.clear()
    prices_mod._download_prices("X.NS", "5y", "1d", actions=True)
    assert seen["period"] == "5y" and "start" not in seen and seen["actions"] is True


# ──────────────────────────────────────────────────────────
# Same result as the full-period path
# ──────────────────────────────────────────────────────────

def test_incremental_stores_exactly_what_the_full_path_stores(monkeypatch):
    first = _frame(DAYS[DAYS <= "2026-10-09"])
    # Three new sessions, a revision of the last stored bar (in window) and of a bar 3 weeks back (outside).
    second = _frame(DAYS[DAYS <= "2026-10-14"], revisions={"2026-10-09": 150.0, "2026-09-15": 999.0})
    results = []
    for force_full in (True, False):
        engine = create_engine("sqlite://", future=True)
        Base.metadata.create_all(engine)
        s = sessionmaker(bind=engine, expire_on_commit=False)()
        a = Asset(exchange_id=1, ticker="TEST.NS", name="Test Co", asset_type="stock")
        s.add(a)
        s.commit()

        with monkeypatch.context() as m:
            m.setattr(prices_mod, "_download_prices", Server(first))
            prices_mod.ingest_daily_prices(s, a, now=WEEKEND)
            server = Server(second)
            m.setattr(prices_mod, "_download_prices", server)
            if force_full:
                m.setattr(prices_mod, "_download_start", lambda *args, **kw: None)
            res = prices_mod._ingest_daily(s, a, WEEKEND)
        results.append((_daily_rows(s), res.counts.inserted, res.counts.updated,
                        server.calls[0]["start"]))
        s.close()
        engine.dispose()

    (full_rows, f_ins, f_upd, f_start), (inc_rows, i_ins, i_upd, i_start) = results
    assert f_start is None and i_start == date(2026, 10, 9) - timedelta(days=14)
    assert inc_rows == full_rows
    assert (i_ins, i_upd) == (f_ins, f_upd) == (3, 1)
    assert inc_rows[date(2026, 10, 9)][3] == 150.0          # revised last bar healed
    assert inc_rows[date(2026, 9, 15)][3] != 999.0          # older history stays insert-only


def test_second_incremental_run_is_a_noop(session, asset, monkeypatch):
    _seed(session, asset, monkeypatch, _frame(DAYS[DAYS <= "2026-10-09"]))
    _serve(monkeypatch, _frame())
    assert prices_mod.ingest_daily_prices(session, asset, now=WEEKEND) == 5
    assert prices_mod.ingest_daily_prices(session, asset, now=WEEKEND) == 0


# ──────────────────────────────────────────────────────────
# Thin incremental window
# ──────────────────────────────────────────────────────────

def test_thin_incremental_frame_triggers_exactly_one_full_download(session, asset, monkeypatch, caplog):
    _seed(session, asset, monkeypatch, _frame(DAYS[DAYS <= "2026-10-09"]))
    full = _frame()
    thin = full.iloc[-3:]                                   # < DAILY_REFRESH_SESSIONS sessions
    server = Server(thin, full=full)
    monkeypatch.setattr(prices_mod, "_download_prices", server)

    with caplog.at_level(logging.INFO, logger=prices_mod.logger.name):
        res = prices_mod._ingest_daily(session, asset, WEEKEND)

    assert [c["start"] for c in server.calls] == [date(2026, 9, 25), None]
    assert server.calls[1]["period"] == settings.DAILY_PERIOD and server.calls[1]["actions"] is True
    assert (res.incremental_downloads, res.full_downloads) == (1, 1)
    assert set(_daily_rows(session)) == {d.date() for d in full.index}
    assert any(f"downloading {settings.DAILY_PERIOD}" in r.getMessage() for r in caplog.records if r.levelno == logging.INFO)


def test_empty_incremental_frame_falls_back_to_full(session, asset, monkeypatch):
    _seed(session, asset, monkeypatch, _frame(DAYS[DAYS <= "2026-10-09"]))
    server = Server(_frame().iloc[0:0], full=_frame())
    monkeypatch.setattr(prices_mod, "_download_prices", server)
    prices_mod.ingest_daily_prices(session, asset, now=WEEKEND)
    assert len(server.calls) == 2 and server.calls[1]["start"] is None


def test_fresh_listing_with_few_bars_does_not_loop(session, asset, monkeypatch):
    _seed(session, asset, monkeypatch, _frame(DAYS[:2]))
    server = Server(_frame(DAYS[:3]))                        # the full frame is thin too
    monkeypatch.setattr(prices_mod, "_download_prices", server)
    prices_mod.ingest_daily_prices(session, asset, now=WEEKEND)
    assert len(server.calls) == 2                            # one incremental + one full, no more
    assert len(_daily_rows(session)) == 3


# ──────────────────────────────────────────────────────────
# Unfinished bar
# ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("now, stored", [
    (datetime(2026, 10, 16, 11, 0), False),
    (datetime(2026, 10, 16, 16, 30), True),
])
def test_todays_bar_waits_for_final_time_on_the_incremental_path(
    session, asset, monkeypatch, now, stored
):
    _seed(session, asset, monkeypatch, _frame(DAYS[DAYS <= "2026-10-14"]))
    server = _serve(monkeypatch, _frame())                   # includes today's 10-16 bar
    prices_mod.ingest_daily_prices(session, asset, now=now)
    assert server.calls[0]["start"] == date(2026, 10, 14) - timedelta(days=14)   # incremental
    assert len(server.calls) == 1
    assert (date(2026, 10, 16) in _daily_rows(session)) is stored
    assert date(2026, 10, 15) in _daily_rows(session)


# ──────────────────────────────────────────────────────────
# Corporate actions
# ──────────────────────────────────────────────────────────

def _with_actions(frame: pd.DataFrame, divs: dict, splits: dict) -> pd.DataFrame:
    frame = frame.copy()
    for d, v in divs.items():
        frame.loc[d, "Dividends"] = v
    for d, v in splits.items():
        frame.loc[d, "Stock Splits"] = v
    return frame


def test_new_split_in_incremental_frame_triggers_one_full_redownload(session, asset, monkeypatch):
    stored = _with_actions(_frame(DAYS[DAYS <= "2026-10-09"]), {"2026-09-02": 2.5}, {})
    _seed(session, asset, monkeypatch, stored)
    assert _actions(session) == {(date(2026, 9, 2), "dividend"): 2.5}

    # yfinance re-states the old dividend for the split; only the full frame has it.
    split = {"2026-10-14": 2.0}
    incremental = _with_actions(_frame(), {"2026-10-05": 3.0}, split)
    full = _with_actions(_frame(), {"2026-09-02": 1.25, "2026-10-05": 3.0}, split)
    server = Server(incremental, full=full)
    monkeypatch.setattr(prices_mod, "_download_prices", server)

    res = prices_mod._ingest_daily(session, asset, WEEKEND)

    assert [c["start"] for c in server.calls] == [date(2026, 9, 25), None]
    assert all(c["actions"] is True for c in server.calls)
    assert server.calls[1]["period"] == settings.DAILY_PERIOD
    assert _actions(session) == {
        (date(2026, 9, 2), "dividend"): 1.25,                # re-stated in place
        (date(2026, 10, 5), "dividend"): 3.0,
        (date(2026, 10, 14), "split"): 2.0,
    }
    assert (res.actions.inserted, res.actions.updated) == (2, 1)
    assert (res.incremental_downloads, res.full_downloads) == (1, 1)


def test_known_split_does_not_trigger_a_full_download(session, asset, monkeypatch):
    split = {"2026-10-14": 2.0}
    _seed(session, asset, monkeypatch,
          _with_actions(_frame(DAYS[DAYS <= "2026-10-14"]), {}, split))
    assert (date(2026, 10, 14), "split") in _actions(session)

    server = _serve(monkeypatch, _with_actions(_frame(), {"2026-10-15": 1.0}, split))
    prices_mod.ingest_daily_prices(session, asset, now=WEEKEND)
    assert len(server.calls) == 1 and server.calls[0]["start"] is not None
    assert _actions(session) == {(date(2026, 10, 14), "split"): 2.0,
                                 (date(2026, 10, 15), "dividend"): 1.0}


def test_failed_full_redownload_leaves_the_split_unstored_for_a_retry(session, asset, monkeypatch):
    _seed(session, asset, monkeypatch, _frame(DAYS[DAYS <= "2026-10-09"]))
    incremental = _with_actions(_frame(), {"2026-10-05": 3.0}, {"2026-10-14": 2.0})

    def flaky(ticker, period, interval, actions=False, start=None):
        if start is None:
            raise RuntimeError("rate limited")
        return incremental[incremental.index >= pd.Timestamp(start)].copy()

    monkeypatch.setattr(prices_mod, "_download_prices", flaky)
    prices_mod.ingest_daily_prices(session, asset, now=WEEKEND)
    assert _actions(session) == {(date(2026, 10, 5), "dividend"): 3.0}

    server = Server(incremental)                              # next run: the split is still "new"
    monkeypatch.setattr(prices_mod, "_download_prices", server)
    prices_mod.ingest_daily_prices(session, asset, now=WEEKEND)
    assert len(server.calls) == 2
    assert (date(2026, 10, 14), "split") in _actions(session)


# ──────────────────────────────────────────────────────────
# Intraday
# ──────────────────────────────────────────────────────────

def _intraday(days: list[str], slots: int = 4) -> pd.DataFrame:
    stamps = [pd.Timestamp(f"{d} 09:15", tz="Asia/Kolkata") + pd.Timedelta(minutes=15 * i)
              for d in days for i in range(slots)]
    return pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.5, "Close": 100.5, "Volume": 500},
        index=pd.DatetimeIndex(stamps),
    )


def _store_intraday(session, asset, day: date) -> None:
    session.add(PriceIntraday(asset_id=asset.id, timestamp=datetime.combine(day, datetime.min.time())
                              .replace(hour=9, minute=15), interval="15m",
                              open=100, high=101, low=99, close=100, volume=1))
    session.commit()


class IntradayServer:
    def __init__(self, frame, full=None):
        self.frame, self.full, self.calls = frame, full if full is not None else frame, []

    def __call__(self, ticker, period, interval, actions=False, start=None):
        self.calls.append(dict(period=period, interval=interval, start=start))
        if start is None:
            return self.full.copy()
        return self.frame[self.frame.index >= pd.Timestamp(start, tz="Asia/Kolkata")].copy()


def test_intraday_no_rows_requests_full_period(session, asset, monkeypatch):
    server = IntradayServer(_intraday(["2026-10-15", "2026-10-16"]))
    monkeypatch.setattr(prices_mod, "_download_prices", server)
    prices_mod.ingest_intraday_prices(session, asset, now=WEEKEND)
    assert server.calls == [dict(period="60d", interval="15m", start=None)]


def test_intraday_with_rows_requests_overlap_start(session, asset, monkeypatch):
    _store_intraday(session, asset, date(2026, 10, 15))
    server = IntradayServer(_intraday(["2026-10-13", "2026-10-14", "2026-10-15", "2026-10-16"]))
    monkeypatch.setattr(prices_mod, "_download_prices", server)
    prices_mod.ingest_intraday_prices(session, asset, now=WEEKEND)
    assert server.calls == [dict(period="60d", interval="15m", start=date(2026, 10, 10))]


def test_intraday_overlap_start_beyond_the_60_day_limit_uses_the_period(session, asset, monkeypatch):
    # Last bar 2026-08-23: overlap start 08-18 is 60 days before 10-17 -> rejected by Yahoo.
    _store_intraday(session, asset, date(2026, 8, 23))
    server = IntradayServer(_intraday(["2026-10-15", "2026-10-16"]))
    monkeypatch.setattr(prices_mod, "_download_prices", server)
    prices_mod.ingest_intraday_prices(session, asset, now=WEEKEND)
    assert server.calls == [dict(period="60d", interval="15m", start=None)]


def test_download_start_intraday_boundary(session, asset):
    _store_intraday(session, asset, date(2026, 8, 24))        # start 08-19 = 59 days back: usable
    assert prices_mod._download_start(session, asset, "15m", WEEKEND) == date(2026, 8, 19)
    session.query(PriceIntraday).delete()
    _store_intraday(session, asset, date(2026, 8, 23))        # start 08-18 = 60 days back: not usable
    assert prices_mod._download_start(session, asset, "15m", WEEKEND) is None


def test_download_start_uses_the_table_of_the_interval(session, asset, monkeypatch):
    assert prices_mod._download_start(session, asset, "1d", WEEKEND) is None
    _store_intraday(session, asset, date(2026, 10, 15))
    assert prices_mod._download_start(session, asset, "1d", WEEKEND) is None    # intraday rows only
    session.add(PriceDaily(asset_id=asset.id, date=date(2026, 10, 9), open=1, high=2, low=1,
                           close=1.5, volume=1))
    session.commit()
    assert prices_mod._download_start(session, asset, "1d", WEEKEND) == date(2026, 9, 25)
    assert prices_mod._download_start(session, asset, "15m", WEEKEND) == date(2026, 10, 10)


def test_thin_intraday_window_falls_back_to_full(session, asset, monkeypatch):
    _store_intraday(session, asset, date(2026, 10, 15))
    server = IntradayServer(_intraday(["2026-10-16"]),                  # one trading day only
                            full=_intraday(["2026-10-14", "2026-10-15", "2026-10-16"]))
    monkeypatch.setattr(prices_mod, "_download_prices", server)
    prices_mod.ingest_intraday_prices(session, asset, now=WEEKEND)
    assert [c["start"] for c in server.calls] == [date(2026, 10, 10), None]
    stored_days = {r.date() for r in session.execute(select(PriceIntraday.timestamp)).scalars()}
    assert stored_days == {date(2026, 10, 14), date(2026, 10, 15), date(2026, 10, 16)}


# ──────────────────────────────────────────────────────────
# Stats
# ──────────────────────────────────────────────────────────

def test_ingest_all_prices_stats(session, monkeypatch):
    orig_now = prices_mod._now_ist
    monkeypatch.setattr(prices_mod, "_now_ist", lambda now: orig_now(now or WEEKEND))
    monkeypatch.setattr(prices_mod.time_module, "sleep", lambda s: None)

    a = Asset(exchange_id=1, ticker="A.NS", name="A", asset_type="stock")
    b = Asset(exchange_id=1, ticker="B.NS", name="B", asset_type="stock")   # no history, no data
    session.add_all([a, b])
    session.commit()

    # Existing history: 8 daily sessions up to 10-14 and one intraday day (10-15).
    _seed(session, a, monkeypatch, _frame(DAYS[DAYS <= "2026-10-14"]))
    stamps = _intraday(["2026-10-15"])
    for ts, row in stamps.iterrows():
        session.add(PriceIntraday(
            asset_id=a.id, timestamp=ts.to_pydatetime().astimezone(prices_mod._IST).replace(tzinfo=None),
            interval="15m", open=row.Open, high=row.High, low=row.Low, close=row.Close,
            volume=int(row.Volume)))
    session.commit()

    daily = _with_actions(_frame(revisions={"2026-10-13": 500.0}), {"2026-10-05": 3.0},
                          {"2026-10-12": 2.0})
    daily.loc["2026-10-15", ["Open", "High", "Low", "Close", "Adj Close", "Volume"]] = [50, 50, 50, 50, 50, 0]
    intraday = _intraday(["2026-10-15", "2026-10-16"])
    intraday.loc[intraday.index[0], "Close"] = 100.9          # revision on a stored 10-15 bar
    intraday.loc[intraday.index[-1], "Close"] = np.nan        # rejected: no close

    calls = []

    def fake(ticker, period, interval, actions=False, start=None):
        calls.append((ticker, interval, start))
        if ticker == "B.NS":
            return pd.DataFrame()
        if interval == "1d":
            return daily[daily.index >= pd.Timestamp(start)].copy() if start else daily.copy()
        return intraday[intraday.index >= pd.Timestamp(start, tz="Asia/Kolkata")].copy() if start else intraday.copy()

    monkeypatch.setattr(prices_mod, "_download_prices", fake)
    stats = prices_mod.ingest_all_prices(session)

    # A, daily: 10-16 inserted; 10-13 updated; 10-15 (zero-volume flat) rejected; the dividend and
    # the new split (which costs one full re-download) are inserted.
    # A, intraday: 10-16 x3 inserted; the NaN-close bar rejected; the 10-15 revision updated.
    assert stats == {
        "daily_total": 2, "intraday_total": 4, "errors": [],
        "daily_inserted": 1, "daily_updated": 1, "daily_rejected": 1,
        "intraday_inserted": 3, "intraday_updated": 1, "intraday_rejected": 1,
        "actions_inserted": 2, "actions_updated": 0,
        "full_downloads": 3, "incremental_downloads": 2,       # B x2, A's new split x1
        "no_data": ["B.NS"],                                    # empty download, no stored rows
    }
    assert calls == [
        ("A.NS", "1d", date(2026, 9, 30)), ("A.NS", "1d", None), ("A.NS", "15m", date(2026, 10, 10)),
        ("B.NS", "1d", None), ("B.NS", "15m", None),
    ]


# ──────────────────────────────────────────────────────────
# main.py ingest summary
# ──────────────────────────────────────────────────────────

def test_cmd_ingest_summary_reports_inserted_updated_rejected(session, monkeypatch, caplog):
    import main
    from investbot.data.ingestion import financials

    stats = {
        "daily_total": 7, "intraday_total": 11, "errors": [],
        "daily_inserted": 5, "daily_updated": 2, "daily_rejected": 3,
        "intraday_inserted": 9, "intraday_updated": 2, "intraday_rejected": 4,
        "actions_inserted": 6, "actions_updated": 1,
        "full_downloads": 8, "incremental_downloads": 90,
    }
    monkeypatch.setattr(prices_mod, "ingest_all_prices", lambda *a, **k: stats)
    monkeypatch.setattr(financials, "ingest_all_financials",
                        lambda *a, **k: {"annual": 1, "quarterly": 2, "errors": []})
    monkeypatch.setattr(main, "SessionLocal", lambda: session)
    monkeypatch.setattr(session, "close", lambda: None)

    with caplog.at_level(logging.INFO, logger="investbot"):
        main.cmd_ingest()
    text = "\n".join(r.getMessage() for r in caplog.records)

    assert "Daily prices:    inserted 5, updated 2, rejected 3" in text
    assert "Intraday prices: inserted 9, updated 2, rejected 4" in text
    assert "Corporate actions: inserted 6, updated 1" in text
    assert "8 full, 90 incremental" in text
    assert " rows" not in text
