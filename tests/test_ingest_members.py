"""
`ingest_all_prices(full=, assets=, intervals=)`, `member_assets` and `main.py ingest --full / --members`.

In-memory SQLite plus a stubbed yfinance download; no network, no real DB.
"""
from __future__ import annotations

import logging
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from investbot.config import settings
from investbot.data.ingestion import prices as prices_mod
from investbot.data.models import (
    Asset, Base, CorporateAction, IndexMembership, PriceDaily, PriceIntraday,
)

WEEKEND = datetime(2026, 10, 17, 12, 0)
DAYS = pd.bdate_range("2026-08-24", "2026-10-16")
INTRADAY_DAYS = ["2026-10-14", "2026-10-15", "2026-10-16"]


@pytest.fixture
def session():
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    yield s
    s.close()
    engine.dispose()


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    orig = prices_mod._now_ist
    monkeypatch.setattr(prices_mod, "_now_ist", lambda now: orig(now or WEEKEND))
    monkeypatch.setattr(prices_mod.time_module, "sleep", lambda s: None)


def _asset(session, ticker, active=True):
    a = Asset(exchange_id=1, ticker=ticker, name=ticker, asset_type="stock", is_active=active)
    session.add(a)
    session.commit()
    return a


def _member(session, asset, index_code="NIFTY50"):
    session.add(IndexMembership(index_code=index_code, asset_id=asset.id,
                                effective_from=date(2020, 1, 1), effective_to=date(2022, 1, 1)))
    session.commit()


def _daily(days=DAYS, revisions: dict | None = None, dividends: dict | None = None,
           splits: dict | None = None) -> pd.DataFrame:
    n = len(days)
    close = 100.0 + np.arange(n)
    df = pd.DataFrame({
        "Open": close - 1, "High": close + 2, "Low": close - 2, "Close": close,
        "Adj Close": close, "Volume": 1_000 + np.arange(n),
        "Dividends": 0.0, "Stock Splits": 0.0,
    }, index=pd.DatetimeIndex(days))
    for d, c in (revisions or {}).items():
        df.loc[d, ["High", "Close", "Adj Close"]] = [c + 2, c, c]
    for d, v in (dividends or {}).items():
        df.loc[d, "Dividends"] = v
    for d, v in (splits or {}).items():
        df.loc[d, "Stock Splits"] = v
    return df


def _intraday(days=INTRADAY_DAYS) -> pd.DataFrame:
    stamps = [ts for d in days for ts in pd.date_range(f"{d} 09:15", f"{d} 15:15", freq="15min",
                                                       tz="Asia/Kolkata")]
    close = 100.0 + np.arange(len(stamps))
    return pd.DataFrame({"Open": close - 1, "High": close + 1, "Low": close - 1, "Close": close,
                         "Adj Close": close, "Volume": 500}, index=pd.DatetimeIndex(stamps))


class Server:
    """Fake yfinance: serves per-ticker daily/intraday frames (sliced by `start`), records requests."""

    def __init__(self, daily=None, intraday=None):
        self.daily = daily or {}
        self.intraday = intraday or {}
        self.calls: list[tuple] = []        # (ticker, interval, period, start)

    def __call__(self, ticker, period, interval, actions=False, start=None):
        self.calls.append((ticker, interval, period, start))
        frame = (self.daily if interval == "1d" else self.intraday).get(ticker)
        if frame is None:
            return pd.DataFrame()
        if start is not None:
            cut = pd.Timestamp(start, tz=None if interval == "1d" else "Asia/Kolkata")
            frame = frame[frame.index >= cut]
        return frame.copy()


def _serve(monkeypatch, **kw) -> Server:
    server = Server(**kw)
    monkeypatch.setattr(prices_mod, "_download_prices", server)
    return server


def _rows(session, asset):
    return {r.date: (r.open, r.high, r.low, r.close, r.adj_close, r.volume)
            for r in session.execute(select(PriceDaily).where(PriceDaily.asset_id == asset.id)).scalars()}


def _actions(session, asset):
    return {(r.ex_date, r.action_type): r.value for r in session.execute(
        select(CorporateAction).where(CorporateAction.asset_id == asset.id)).scalars()}


def _seed_history(session, monkeypatch, asset, upto="2026-10-09"):
    """Daily + intraday history up to `upto`, stored through the normal empty-DB path."""
    _serve(monkeypatch, daily={asset.ticker: _daily(DAYS[DAYS <= upto])},
           intraday={asset.ticker: _intraday(INTRADAY_DAYS[:1])})
    prices_mod.ingest_all_prices(session)


# ── defaults reproduce the current behaviour ──────────────

def test_default_call_requests_incremental_with_history_and_full_without(session, monkeypatch):
    a = _asset(session, "A.NS")
    _seed_history(session, monkeypatch, a)
    b = _asset(session, "B.NS")
    server = _serve(monkeypatch,
                    daily={"A.NS": _daily(), "B.NS": _daily()},
                    intraday={"A.NS": _intraday(), "B.NS": _intraday()})

    stats = prices_mod.ingest_all_prices(session)

    assert server.calls == [
        ("A.NS", "1d", settings.DAILY_PERIOD, date(2026, 9, 25)),
        ("A.NS", "15m", settings.INTRADAY_PERIOD, date(2026, 10, 9)),
        ("B.NS", "1d", settings.DAILY_PERIOD, None),
        ("B.NS", "15m", settings.INTRADAY_PERIOD, None),
    ]
    assert stats["errors"] == [] and stats["no_data"] == []
    assert stats["full_downloads"] == 2 and stats["incremental_downloads"] == 2
    assert stats["daily_inserted"] == 5 + len(DAYS)
    assert len(_rows(session, a)) == len(DAYS) and len(_rows(session, b)) == len(DAYS)
    assert stats.keys() >= {
        "daily_total", "intraday_total", "errors", "daily_inserted", "daily_updated",
        "daily_rejected", "intraday_inserted", "intraday_updated", "intraday_rejected",
        "actions_inserted", "actions_updated", "full_downloads", "incremental_downloads", "no_data"}


def test_unknown_interval_is_rejected(session):
    with pytest.raises(ValueError, match="hourly"):
        prices_mod.ingest_all_prices(session, intervals=("hourly",))


# ── full=True ─────────────────────────────────────────────

def test_full_requests_full_period_and_stays_insert_only_outside_the_window(session, monkeypatch):
    a = _asset(session, "A.NS")
    _seed_history(session, monkeypatch, a)
    stored = _rows(session, a)
    gap = date(2026, 9, 2)
    session.query(PriceDaily).filter_by(asset_id=a.id, date=gap).delete()
    session.commit()
    old_day = date(2026, 9, 15)                              # stored, outside the refresh window
    assert old_day in stored

    download = _daily(revisions={"2026-09-15": 999.0}, dividends={"2026-09-10": 2.5})
    server = _serve(monkeypatch, daily={"A.NS": download}, intraday={"A.NS": _intraday()})

    stats = prices_mod.ingest_all_prices(session, full=True)

    # Exactly one request per interval, both for the full period with no start date.
    assert server.calls == [("A.NS", "1d", settings.DAILY_PERIOD, None),
                            ("A.NS", "15m", settings.INTRADAY_PERIOD, None)]
    assert stats["full_downloads"] == 2 and stats["incremental_downloads"] == 0

    rows = _rows(session, a)
    assert gap in rows                                       # missing old bar inserted
    assert rows[old_day] == stored[old_day]                  # older stored bar not overwritten
    assert rows[old_day][3] != 999.0
    assert _actions(session, a) == {(date(2026, 9, 10), "dividend"): 2.5}
    assert stats["actions_inserted"] == 1


def test_full_never_fires_a_second_request(session, monkeypatch):
    a = _asset(session, "A.NS")
    _seed_history(session, monkeypatch, a)

    # A thin download would normally trigger the thin-window fallback.
    server = _serve(monkeypatch, daily={"A.NS": _daily().iloc[-2:]}, intraday={"A.NS": _intraday()[:0]})
    prices_mod._ingest_daily(session, a, WEEKEND, full=True)
    assert server.calls == [("A.NS", "1d", settings.DAILY_PERIOD, None)]

    # A split in the download would normally trigger the "new split" re-download.
    server = _serve(monkeypatch, daily={"A.NS": _daily(splits={"2026-10-14": 2.0})})
    res = prices_mod._ingest_daily(session, a, WEEKEND, full=True)
    assert server.calls == [("A.NS", "1d", settings.DAILY_PERIOD, None)]
    assert res.full_downloads == 1 and res.incremental_downloads == 0
    assert _actions(session, a)[(date(2026, 10, 14), "split")] == 2.0


def test_full_intraday_ignores_stored_bars(session, monkeypatch):
    a = _asset(session, "A.NS")
    _seed_history(session, monkeypatch, a)
    server = _serve(monkeypatch, intraday={"A.NS": _intraday()})
    res = prices_mod._ingest_intraday(session, a, WEEKEND, full=True)
    assert server.calls == [("A.NS", "15m", settings.INTRADAY_PERIOD, None)]
    assert res.counts.inserted > 0


def test_full_still_drops_bars_after_the_16_ist_rule(session, monkeypatch):
    a = _asset(session, "A.NS")
    morning = datetime(2026, 10, 16, 10, 0)                  # today's bar is not final yet
    _serve(monkeypatch, daily={"A.NS": _daily()})
    prices_mod._ingest_daily(session, a, morning, full=True)
    assert max(_rows(session, a)) == date(2026, 10, 15)


# ── member_assets ─────────────────────────────────────────

def test_member_assets_returns_inactive_members_only(session):
    past_b = _asset(session, "B.NS", active=False)
    past_a = _asset(session, "A.NS", active=False)
    active_member = _asset(session, "C.NS", active=True)
    _asset(session, "D.NS", active=False)                    # inactive, never a member
    other_index = _asset(session, "E.NS", active=False)
    for asset in (past_b, past_a, active_member):
        _member(session, asset)
    session.add(IndexMembership(index_code="NIFTY50", asset_id=past_a.id,
                                effective_from=date(2024, 1, 1)))      # second stint: listed once
    _member(session, other_index, index_code="NIFTYBANK")
    session.commit()

    assert [a.ticker for a in prices_mod.member_assets(session)] == ["A.NS", "B.NS"]
    assert [a.ticker for a in prices_mod.member_assets(session, "NIFTYBANK")] == ["E.NS"]


# ── empty downloads ───────────────────────────────────────

def test_empty_download_lands_in_no_data_and_others_are_processed(session, monkeypatch):
    gone = _asset(session, "GONE.NS", active=False)
    ok = _asset(session, "OK.NS", active=False)
    failing = _asset(session, "BOOM.NS", active=False)
    server = _serve(monkeypatch, daily={"OK.NS": _daily()})

    original = server.__call__

    def download(ticker, *a, **kw):
        if ticker == "BOOM.NS":
            raise RuntimeError("network down")
        return original(ticker, *a, **kw)

    monkeypatch.setattr(prices_mod, "_download_prices", download)

    stats = prices_mod.ingest_all_prices(
        session, assets=[gone, failing, ok], intervals=("daily",))

    assert stats["no_data"] == ["GONE.NS", "BOOM.NS"]
    assert stats["errors"] == []
    assert len(_rows(session, ok)) == len(DAYS)
    assert not _rows(session, gone)


def test_stored_history_keeps_a_ticker_out_of_no_data(session, monkeypatch):
    a = _asset(session, "A.NS")
    _seed_history(session, monkeypatch, a)
    _serve(monkeypatch)                                      # every download comes back empty
    assert prices_mod.ingest_all_prices(session, intervals=("daily",))["no_data"] == []


# ── main.py ingest --members / --full ─────────────────────

@pytest.fixture
def main_mod(session, monkeypatch):
    import main
    monkeypatch.setattr(main, "SessionLocal", lambda: session)
    monkeypatch.setattr(session, "close", lambda: None)
    # These tests seed a few ad-hoc stints; the stale-membership preflight has its own tests
    # (test_membership_check.py).
    from investbot.data.membership_check import MembershipSync
    monkeypatch.setattr(main, "check_membership_sync", lambda s, code="NIFTY50": MembershipSync(
        code, True, 0, 0, (), (), (), ()))
    return main


def test_members_ingest_is_daily_only_and_never_changes_is_active(
        session, monkeypatch, main_mod, caplog):
    from investbot.data.ingestion import financials

    past = _asset(session, "PAST.NS", active=False)
    gone = _asset(session, "HDFC.NS", active=False)
    active_member = _asset(session, "LIVE.NS", active=True)
    other = _asset(session, "OTHER.NS", active=False)        # inactive non-member
    for asset in (past, gone, active_member):
        _member(session, asset)
    flags_before = {a.ticker: a.is_active for a in session.query(Asset)}

    server = _serve(monkeypatch,
                    daily={t: _daily(splits={"2026-09-01": 2.0}, dividends={"2026-09-10": 1.0})
                           for t in ("PAST.NS", "LIVE.NS", "OTHER.NS")},
                    intraday={t: _intraday() for t in ("PAST.NS", "LIVE.NS", "OTHER.NS")})
    fin_calls = []
    monkeypatch.setattr(financials, "ingest_all_financials", lambda *a, **k: fin_calls.append(1))

    with caplog.at_level(logging.INFO, logger="investbot"):
        main_mod.cmd_ingest(["--members"])
    text = "\n".join(r.getMessage() for r in caplog.records)

    assert server.calls == [("HDFC.NS", "1d", settings.DAILY_PERIOD, None),
                            ("PAST.NS", "1d", settings.DAILY_PERIOD, None)]
    assert fin_calls == []
    assert {a.ticker: a.is_active for a in session.query(Asset)} == flags_before
    assert len(_rows(session, past)) == len(DAYS)
    assert _actions(session, past) == {(date(2026, 9, 1), "split"): 2.0,
                                       (date(2026, 9, 10), "dividend"): 1.0}
    assert not _rows(session, active_member) and not _rows(session, other)
    assert session.query(PriceIntraday).count() == 0
    assert "No price data for 1 ticker(s): HDFC.NS" in text
    assert "renamed or delisted" in text and "INDUSTOWER" in text
    assert "Intraday" not in text and "financials" not in text.lower()


def test_members_ingest_is_incremental_with_rows_and_full_without(session, monkeypatch, main_mod):
    has = _asset(session, "HAS.NS", active=False)
    new = _asset(session, "NEW.NS", active=False)
    _member(session, has)
    _member(session, new)
    _serve(monkeypatch, daily={"HAS.NS": _daily(DAYS[DAYS <= "2026-10-09"])})
    prices_mod.ingest_all_prices(session, assets=[has], intervals=("daily",))

    server = _serve(monkeypatch, daily={"HAS.NS": _daily(), "NEW.NS": _daily()})
    main_mod.cmd_ingest(["--members"])
    assert server.calls == [("HAS.NS", "1d", settings.DAILY_PERIOD, date(2026, 9, 25)),
                            ("NEW.NS", "1d", settings.DAILY_PERIOD, None)]

    server = _serve(monkeypatch, daily={"HAS.NS": _daily(), "NEW.NS": _daily()})
    main_mod.cmd_ingest(["--members", "NIFTY50", "--full"])
    assert server.calls == [("HAS.NS", "1d", settings.DAILY_PERIOD, None),
                            ("NEW.NS", "1d", settings.DAILY_PERIOD, None)]


def _stub_ingest(monkeypatch, main_mod, members=()):
    calls = []
    stats = {k: 0 for k in (
        "daily_total", "intraday_total", "daily_inserted", "daily_updated", "daily_rejected",
        "intraday_inserted", "intraday_updated", "intraday_rejected", "actions_inserted",
        "actions_updated", "full_downloads", "incremental_downloads")}
    stats.update(errors=[], no_data=[])

    def fake_ingest(session, asset_type="stock", **kw):
        calls.append(("ingest", asset_type, kw))
        return stats

    def fake_members(session, index_code="NIFTY50", asset_type="stock"):
        calls.append(("members", index_code))
        return list(members)

    from investbot.data.ingestion import financials
    monkeypatch.setattr(prices_mod, "ingest_all_prices", fake_ingest)
    monkeypatch.setattr(prices_mod, "member_assets", fake_members)
    monkeypatch.setattr(financials, "ingest_all_financials",
                        lambda *a, **k: calls.append(("financials",)) or
                        {"annual": 0, "quarterly": 0, "errors": []})
    return calls


def test_cmd_ingest_without_flags_is_unchanged(monkeypatch, main_mod):
    calls = _stub_ingest(monkeypatch, main_mod)
    main_mod.cmd_ingest([])
    assert calls == [("ingest", "stock", {"full": False}), ("financials",)]


def test_cmd_ingest_full_flag(monkeypatch, main_mod):
    calls = _stub_ingest(monkeypatch, main_mod)
    main_mod.cmd_ingest(["--full"])
    assert calls == [("ingest", "stock", {"full": True}), ("financials",)]


def test_cmd_ingest_members_flag_parses_default_and_explicit_index(monkeypatch, main_mod):
    calls = _stub_ingest(monkeypatch, main_mod)
    main_mod.cmd_ingest(["--members"])
    main_mod.cmd_ingest(["--members", "NIFTYNEXT50", "--full"])
    main_mod.cmd_ingest(["--full", "--members"])
    assert calls == [
        ("members", "NIFTY50"),
        ("ingest", "stock", {"full": False, "assets": [], "intervals": ("daily",)}),
        ("members", "NIFTYNEXT50"),
        ("ingest", "stock", {"full": True, "assets": [], "intervals": ("daily",)}),
        ("members", "NIFTY50"),
        ("ingest", "stock", {"full": True, "assets": [], "intervals": ("daily",)}),
    ]
