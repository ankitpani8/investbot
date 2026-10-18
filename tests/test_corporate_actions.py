"""
Corporate actions: capture from the daily download, load by ex-date, credit
dividends onto bars, and detect splits whose history was not re-adjusted.

In-memory SQLite plus a stubbed yfinance download; no network, no real DB.
"""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from investbot.data.corporate_actions import dividends_on_bars, load_actions
from investbot.data.ingestion import prices as prices_mod
from investbot.data.models import Asset, Base, CorporateAction, PriceDaily
from investbot.data.quality import SplitIssue, audit_splits, detect_unadjusted_splits

WEEKEND = datetime(2026, 10, 17, 12, 0)             # IST; every fixture bar is in the past


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


def _download(with_actions: bool = True, div2: float = 5.0) -> pd.DataFrame:
    days = pd.bdate_range("2026-10-05", "2026-10-14")            # 8 sessions
    n = len(days)
    df = pd.DataFrame({
        "Open": np.full(n, 100.0), "High": np.full(n, 102.0), "Low": np.full(n, 99.0),
        "Close": np.full(n, 101.0), "Adj Close": np.full(n, 101.0),
        "Volume": np.full(n, 1000),
    }, index=days)
    if with_actions:
        df["Dividends"] = 0.0
        df["Stock Splits"] = 0.0
        df.loc["2026-10-07", "Dividends"] = 2.5
        df.loc["2026-10-12", "Dividends"] = div2
        df.loc["2026-10-09", "Stock Splits"] = 2.0
    return df


def _stub(monkeypatch, frame: pd.DataFrame) -> None:
    monkeypatch.setattr(prices_mod, "_download_prices", lambda *a, **k: frame.copy())


def _actions(session) -> dict:
    return {
        (r.ex_date, r.action_type): (r.value, r.source, r.fetched_at)
        for r in session.execute(select(CorporateAction)).scalars()
    }


def _bars(session) -> list[tuple]:
    return [tuple(r) for r in session.execute(
        select(PriceDaily.date, PriceDaily.open, PriceDaily.high, PriceDaily.low,
               PriceDaily.close, PriceDaily.adj_close, PriceDaily.volume).order_by(PriceDaily.date)
    )]


# ──────────────────────────────────────────────────────────
# Ingestion
# ──────────────────────────────────────────────────────────

def test_daily_download_requests_actions(session, asset, monkeypatch):
    seen = {}

    def fake(ticker, period, interval, actions=False):
        seen["actions"] = actions
        return _download()

    monkeypatch.setattr(prices_mod, "_download_prices", fake)
    prices_mod.ingest_daily_prices(session, asset, now=WEEKEND)
    assert seen["actions"] is True


def test_two_dividends_and_a_split_are_stored(session, asset, monkeypatch):
    _stub(monkeypatch, _download())
    prices_mod.ingest_daily_prices(session, asset, now=WEEKEND)
    stored = _actions(session)
    assert {k: v[0] for k, v in stored.items()} == {
        (date(2026, 10, 7), "dividend"): 2.5,
        (date(2026, 10, 12), "dividend"): 5.0,
        (date(2026, 10, 9), "split"): 2.0,
    }
    assert all(v[1] == "yfinance" and v[2] is not None for v in stored.values())


def test_second_identical_run_inserts_and_updates_nothing(session, asset, monkeypatch):
    frame = _download()
    _stub(monkeypatch, frame)
    prices_mod.ingest_daily_prices(session, asset, now=WEEKEND)
    first = _actions(session)

    counts = prices_mod._upsert_actions(session, asset, frame, datetime(2026, 10, 18, 9, 0))
    assert (counts.inserted, counts.updated, counts.unchanged) == (0, 0, 3)
    assert _actions(session) == first               # fetched_at untouched too


def test_restated_dividend_is_updated_in_place(session, asset, monkeypatch):
    _stub(monkeypatch, _download())
    prices_mod.ingest_daily_prices(session, asset, now=WEEKEND)
    before = _actions(session)

    counts = prices_mod._upsert_actions(
        session, asset, _download(div2=2.5), datetime(2026, 10, 19, 9, 0))
    assert (counts.inserted, counts.updated, counts.unchanged) == (0, 1, 2)
    after = _actions(session)
    assert len(after) == 3
    value, _, fetched = after[(date(2026, 10, 12), "dividend")]
    assert value == 2.5 and fetched == datetime(2026, 10, 19, 9, 0)
    assert after[(date(2026, 10, 7), "dividend")] == before[(date(2026, 10, 7), "dividend")]


def test_actions_outside_refresh_window_are_captured(session, asset, monkeypatch):
    # The 2026-10-07 dividend is older than the 5-session refresh window; it is still stored.
    _stub(monkeypatch, _download())
    prices_mod.ingest_daily_prices(session, asset, now=WEEKEND)
    assert (date(2026, 10, 7), "dividend") in _actions(session)


def test_future_dated_actions_are_skipped(session, asset):
    frame = _download()
    frame.loc[pd.Timestamp("2026-10-14"), "Dividends"] = 3.0
    counts = prices_mod._upsert_actions(session, asset, frame, datetime(2026, 10, 13, 10, 0))   # UTC = 15:30 IST on the 13th
    assert (date(2026, 10, 14), "dividend") not in _actions(session)
    assert counts.inserted == 3


def test_price_rows_identical_with_and_without_action_columns(monkeypatch):
    stored = []
    for with_actions in (True, False):
        engine = create_engine("sqlite://", future=True)
        Base.metadata.create_all(engine)
        s = sessionmaker(bind=engine, expire_on_commit=False)()
        a = Asset(exchange_id=1, ticker="TEST.NS", name="Test Co", asset_type="stock")
        s.add(a)
        s.commit()
        _stub(monkeypatch, _download(with_actions=with_actions))
        written = prices_mod.ingest_daily_prices(s, a, now=WEEKEND)
        stored.append((written, _bars(s), bool(_actions(s))))
        s.close()
        engine.dispose()
    assert stored[0][:2] == stored[1][:2]
    assert stored[0][2] is True and stored[1][2] is False


# ──────────────────────────────────────────────────────────
# load_actions
# ──────────────────────────────────────────────────────────

def test_load_actions_never_returns_ex_dates_after_end(session, asset, monkeypatch):
    _stub(monkeypatch, _download())
    prices_mod.ingest_daily_prices(session, asset, now=WEEKEND)

    full = load_actions(session, "TEST.NS")
    assert list(full.columns) == ["action_type", "value"]
    assert full.index.name == "ex_date" and full.index.is_monotonic_increasing
    assert len(full) == 3

    part = load_actions(session, "TEST.NS", end=date(2026, 10, 9))
    assert len(part) == 2 and part.index.max() <= pd.Timestamp("2026-10-09")
    assert load_actions(session, "TEST.NS", end=date(2026, 10, 1)).empty
    assert load_actions(session, "NOPE.NS").empty


# ──────────────────────────────────────────────────────────
# dividends_on_bars
# ──────────────────────────────────────────────────────────

def _div_frame(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {"action_type": "dividend", "value": [v for _, v in rows]},
        index=pd.DatetimeIndex([pd.Timestamp(d) for d, _ in rows], name="ex_date"),
    )


def test_dividend_on_missing_session_lands_on_next_bar():
    bars = pd.DatetimeIndex(["2026-10-05", "2026-10-06", "2026-10-08", "2026-10-09"])
    out = dividends_on_bars(bars, _div_frame([("2026-10-07", 4.0)]))     # 10-07 has no bar
    assert out.tolist() == [0.0, 0.0, 4.0, 0.0]


def test_same_bar_dividends_sum_and_late_ex_dates_are_dropped():
    bars = pd.DatetimeIndex(["2026-10-05", "2026-10-06", "2026-10-07"])
    actions = _div_frame([("2026-10-06", 1.5), ("2026-10-06", 2.0), ("2026-10-09", 9.0)])
    assert dividends_on_bars(bars, actions).tolist() == [0.0, 3.5, 0.0]


def test_splits_are_not_dividends_and_empty_inputs():
    bars = pd.DatetimeIndex(["2026-10-05", "2026-10-06"])
    split = pd.DataFrame({"action_type": ["split"], "value": [2.0]},
                         index=pd.DatetimeIndex(["2026-10-06"], name="ex_date"))
    assert dividends_on_bars(bars, split).tolist() == [0.0, 0.0]
    assert dividends_on_bars(bars, _div_frame([]).iloc[0:0]).tolist() == [0.0, 0.0]
    assert dividends_on_bars(pd.DatetimeIndex([]), split).shape == (0,)


def test_dividends_on_bars_truncation_invariance():
    bars = pd.bdate_range("2026-09-01", "2026-10-30")
    actions = _div_frame([("2026-09-05", 1.0), ("2026-09-15", 2.0), ("2026-09-15", 0.5),
                          ("2026-10-10", 3.0), ("2026-10-28", 4.0), ("2026-11-20", 7.0)])
    full = dividends_on_bars(bars, actions)
    for cut in range(1, len(bars)):
        part = dividends_on_bars(bars[:cut], actions)
        assert np.array_equal(part, full[:cut])


# ──────────────────────────────────────────────────────────
# detect_unadjusted_splits
# ──────────────────────────────────────────────────────────

def _series(pre_scale: float, gap: float = 1.0, n: int = 10, split_at: int = 5):
    """Prices around 100 (post-split); pre-ex bars are scaled by `pre_scale`. `gap` moves the ex bar."""
    idx = pd.bdate_range("2026-09-01", periods=n)
    scale = np.where(np.arange(n) < split_at, pre_scale, 1.0)
    px = 100.0 * scale
    close = px.copy()
    open_ = px.copy()
    open_[split_at:] *= gap
    close[split_at:] *= gap
    return pd.DataFrame({"open": open_, "close": close}, index=idx), idx[split_at]


def _split(ex, factor=2.0):
    return pd.DataFrame({"action_type": ["split"], "value": [factor]},
                        index=pd.DatetimeIndex([ex], name="ex_date"))


def test_unadjusted_split_is_flagged():
    bars, ex = _series(pre_scale=2.0)                       # pre-split prices still ~200
    issues = detect_unadjusted_splits(bars, _split(ex), ticker="TEST.NS")
    assert len(issues) == 1
    assert issues[0] == SplitIssue("TEST.NS", ex.date(), 2.0, pytest.approx(2.0))


def test_adjusted_split_is_not_flagged():
    bars, ex = _series(pre_scale=1.0)
    assert detect_unadjusted_splits(bars, _split(ex)) == []


def test_ordinary_gap_down_day_is_not_flagged():
    bars, ex = _series(pre_scale=1.0, gap=0.9)              # -10% gap on the ex bar
    assert detect_unadjusted_splits(bars, _split(ex)) == []


def test_splits_outside_the_bars_or_nan_prices_are_ignored():
    bars, ex = _series(pre_scale=2.0)
    assert detect_unadjusted_splits(bars, _split(bars.index[0])) == []        # no previous bar
    assert detect_unadjusted_splits(bars, _split(bars.index[-1] + pd.Timedelta(days=30))) == []
    bars.loc[bars.index[4], "close"] = np.nan
    assert detect_unadjusted_splits(bars, _split(ex)) == []
    assert detect_unadjusted_splits(bars.iloc[0:0], _split(ex)) == []


def test_reverse_split_unadjusted_is_flagged():
    bars, ex = _series(pre_scale=0.5)
    assert len(detect_unadjusted_splits(bars, _split(ex, factor=0.5))) == 1


def test_audit_splits_reads_db_per_asset(session, asset):
    bars, ex = _series(pre_scale=2.0)
    for d, o, c in zip(bars.index, bars["open"], bars["close"]):
        session.add(PriceDaily(asset_id=asset.id, date=d.date(), open=o, high=max(o, c),
                               low=min(o, c), close=c, volume=1))
    session.add(CorporateAction(asset_id=asset.id, ex_date=ex.date(), action_type="split", value=2.0))
    other = Asset(exchange_id=1, ticker="OTHER.NS", name="Other", asset_type="stock")
    session.add(other)
    session.commit()
    issues = audit_splits(session)
    assert [(i.ticker, i.ex_date, i.factor) for i in issues] == [("TEST.NS", ex.date(), 2.0)]


# ──────────────────────────────────────────────────────────
# main.py dq
# ──────────────────────────────────────────────────────────

def test_dq_command_reports_unadjusted_splits_and_exits_nonzero(session, asset, monkeypatch, capsys):
    import sys
    import main

    bars, ex = _series(pre_scale=2.0)
    for d, o, c in zip(bars.index, bars["open"], bars["close"]):
        session.add(PriceDaily(asset_id=asset.id, date=d.date(), open=o, high=max(o, c),
                               low=min(o, c), close=c, volume=1))
    session.add(CorporateAction(asset_id=asset.id, ex_date=ex.date(), action_type="split", value=2.0))
    session.commit()

    monkeypatch.setattr(main, "SessionLocal", lambda: session)
    monkeypatch.setattr(session, "close", lambda: None)
    monkeypatch.setattr(sys, "argv", ["main.py", "dq", "--purge"])
    with pytest.raises(SystemExit) as exc:
        main.cmd_dq()
    out = capsys.readouterr().out
    assert exc.value.code == 1
    assert "1 unadjusted split(s)" in out and "TEST.NS" in out and "human decision" in out
    assert session.query(PriceDaily).count() == len(bars)       # --purge leaves them alone
    assert session.query(CorporateAction).count() == 1
