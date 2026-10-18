"""
Scheduler post-close pipeline: ingest -> audit -> score, with bad data in the scoring window excluding that asset.

In-memory SQLite; ingest_all_prices and scorer.score are monkeypatched. No network.
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from investbot.data.models import (
    Asset, Base, CorporateAction, PriceDaily, PriceIntraday, TechnicalScoreRecord,
)
from investbot.scheduler import jobs
from investbot.scoring.technical import scorer
from investbot.scoring.technical.scorer import TechnicalScore


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


def _add_bar(session, asset, day, close=103.0):
    session.add(PriceDaily(asset_id=asset.id, date=day, open=100.0, high=105.0,
                           low=99.0, close=close, adj_close=close, volume=1_000))
    session.commit()


def _add_asset(session, ticker):
    a = Asset(exchange_id=1, ticker=ticker, name=ticker, asset_type="stock")
    session.add(a)
    session.commit()
    return a


def _add_intraday(session, asset, close=100.0):
    session.add(PriceIntraday(asset_id=asset.id, timestamp=datetime(2026, 10, 16, 9, 15),
                              interval="15m", open=100.0, high=101.0, low=99.0,
                              close=close, volume=5))
    session.commit()


class Calls:
    def __init__(self, monkeypatch, *, ingest_raises=False, statuses=None):
        self.ingest = 0
        self.scored = []                  # (ticker, timeframe, kwargs) per scorer.score call
        statuses = statuses or {}

        def fake_ingest(session, asset_type="stock"):
            self.ingest += 1
            if ingest_raises:
                raise RuntimeError("yfinance down")
            return {"daily_total": 5, "intraday_total": 0, "errors": []}

        def fake_score(ticker, timeframe="1d", *args, **kwargs):
            self.scored.append((ticker, timeframe, kwargs))
            return TechnicalScore(ticker, timeframe, None, status=statuses.get(ticker, "ok"))

        monkeypatch.setattr(jobs, "ingest_all_prices", fake_ingest)
        monkeypatch.setattr(jobs.scorer, "score", fake_score)

    @property
    def tickers(self):
        return [t for t, _, _ in self.scored]


def test_clean_audit_scores_each_asset_once(session, asset, monkeypatch, caplog):
    _add_bar(session, asset, date(2026, 10, 16))
    b = _add_asset(session, "B.NS")
    _add_bar(session, b, date(2026, 10, 16))
    calls = Calls(monkeypatch, statuses={"B.NS": "warmup"})
    with caplog.at_level(logging.INFO, logger=jobs.logger.name):
        result = jobs.run_post_close(session)

    assert calls.ingest == 1
    assert calls.tickers == ["B.NS", "TEST.NS"]
    assert all(tf == "1d" and kw["session"] is session for _, tf, kw in calls.scored)
    assert result.bad_rows == 0 and result.excluded == []
    assert result.report_only_bad_rows == 0 and result.split_issues == 0
    assert result.scored == {"ok": 1, "warmup": 1}
    assert result.skipped_reason is None
    assert result.ingested["daily_total"] == 5
    assert "ok=1 warmup=1 insufficient_data=0 no_data=0 error=0" in caplog.text


# Task 240: a bad row no longer blocks the universe. Replaces test_bad_row_blocks_scoring,
# whose single asset now has its bad row inside the window, so it is excluded (and, as the
# only asset, nothing is scored).
def test_bad_row_in_window_excludes_the_only_asset(session, asset, monkeypatch, caplog):
    _add_bar(session, asset, date(2026, 10, 15))
    _add_bar(session, asset, date(2026, 10, 16), close=None)   # incomplete OHLC
    calls = Calls(monkeypatch)
    with caplog.at_level(logging.INFO, logger=jobs.logger.name):
        result = jobs.run_post_close(session)

    assert calls.ingest == 1 and calls.scored == []
    assert result.scored is None
    assert result.bad_rows == 1 and result.excluded == ["TEST.NS"]
    assert "dq" in result.skipped_reason
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("price_daily" in r.getMessage() and "TEST.NS" in r.getMessage() for r in errors)
    # Nothing was purged.
    assert session.query(PriceDaily).count() == 2


def test_bad_intraday_row_warns_and_scores_everyone(session, asset, monkeypatch, caplog):
    b = _add_asset(session, "B.NS")
    for a in (asset, b):
        _add_bar(session, a, date(2026, 10, 16))
    _add_intraday(session, asset, close=None)
    calls = Calls(monkeypatch)
    with caplog.at_level(logging.INFO, logger=jobs.logger.name):
        result = jobs.run_post_close(session)

    assert sorted(calls.tickers) == ["B.NS", "TEST.NS"]
    assert result.bad_rows == 0 and result.excluded == []
    assert result.report_only_bad_rows == 1
    assert result.skipped_reason is None
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("price_intraday" in m and "TEST.NS" in m for m in warns)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_bad_daily_row_older_than_window_warns_and_scores(session, asset, monkeypatch, caplog):
    monkeypatch.setattr(jobs.scorer, "DAILY_LOOKBACK_BARS", 3)
    _add_bar(session, asset, date(2026, 10, 9), close=None)     # 5th most recent bar: outside
    for d in (12, 13, 14, 15, 16):
        _add_bar(session, asset, date(2026, 10, d))
    calls = Calls(monkeypatch)
    with caplog.at_level(logging.INFO, logger=jobs.logger.name):
        result = jobs.run_post_close(session)

    assert calls.tickers == ["TEST.NS"]
    assert result.bad_rows == 0 and result.excluded == []
    assert result.report_only_bad_rows == 1
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("price_daily" in m and "TEST.NS" in m for m in warns)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_bad_daily_row_in_window_excludes_only_that_asset(session, asset, monkeypatch, caplog):
    monkeypatch.setattr(jobs.scorer, "DAILY_LOOKBACK_BARS", 3)
    b = _add_asset(session, "B.NS")
    for d in (13, 14, 15):
        _add_bar(session, asset, date(2026, 10, d), close=None if d == 14 else 103.0)
        _add_bar(session, b, date(2026, 10, d))
    calls = Calls(monkeypatch)
    with caplog.at_level(logging.INFO, logger=jobs.logger.name):
        result = jobs.run_post_close(session)

    assert calls.tickers == ["B.NS"]
    assert result.excluded == ["TEST.NS"] and result.bad_rows == 1
    assert result.report_only_bad_rows == 0           # already reported by the gate
    assert result.scored == {"ok": 1}
    assert result.skipped_reason is None
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len([m for m in errors if "TEST.NS" in m]) >= 1
    assert not any("B.NS" in m for m in errors)


def test_all_assets_bad_skips_scoring(session, asset, monkeypatch):
    b = _add_asset(session, "B.NS")
    for a in (asset, b):
        _add_bar(session, a, date(2026, 10, 16), close=None)
    calls = Calls(monkeypatch)
    result = jobs.run_post_close(session)

    assert calls.scored == []
    assert result.excluded == ["B.NS", "TEST.NS"] and result.bad_rows == 2
    assert result.scored is None and result.skipped_reason


def test_unadjusted_split_is_logged_and_does_not_block(session, asset, monkeypatch, caplog):
    # Prices step down by half on the ex-date but the stored history was never adjusted.
    _add_bar(session, asset, date(2026, 10, 14))
    session.add(PriceDaily(asset_id=asset.id, date=date(2026, 10, 15), open=50.0, high=52.0,
                           low=49.0, close=51.0, adj_close=51.0, volume=1_000))
    session.add(CorporateAction(asset_id=asset.id, ex_date=date(2026, 10, 15),
                                action_type="split", value=2.0))
    session.commit()
    calls = Calls(monkeypatch)
    with caplog.at_level(logging.INFO, logger=jobs.logger.name):
        result = jobs.run_post_close(session)

    assert calls.tickers == ["TEST.NS"]
    assert result.split_issues == 1 and result.skipped_reason is None
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any("split" in m and "TEST.NS" in m for m in errors)


def test_report_only_audit_failure_does_not_block(session, asset, monkeypatch, caplog):
    _add_bar(session, asset, date(2026, 10, 16))
    calls = Calls(monkeypatch)

    def boom(s):
        raise RuntimeError("split table gone")
    monkeypatch.setattr(jobs.quality, "audit_splits", boom)
    with caplog.at_level(logging.INFO, logger=jobs.logger.name):
        result = jobs.run_post_close(session)

    assert calls.tickers == ["TEST.NS"] and result.skipped_reason is None
    assert "split table gone" in caplog.text


def test_ingest_failure_skips_everything(session, asset, monkeypatch, caplog):
    calls = Calls(monkeypatch, ingest_raises=True)
    audited = []
    monkeypatch.setattr(jobs.quality, "audit_prices", lambda *a, **k: audited.append(a) or [])
    with caplog.at_level(logging.INFO, logger=jobs.logger.name):
        result = jobs.run_post_close(session)   # must not raise

    assert calls.scored == [] and audited == []
    assert result.ingested is None and result.bad_rows is None and result.scored is None
    assert result.skipped_reason
    assert "ingest" in caplog.text and "yfinance down" in caplog.text


def test_audit_failure_skips_scoring(session, asset, monkeypatch, caplog):
    calls = Calls(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("db locked")
    monkeypatch.setattr(jobs.quality, "audit_prices", boom)
    with caplog.at_level(logging.INFO, logger=jobs.logger.name):
        result = jobs.run_post_close(session)

    assert calls.scored == []
    assert result.skipped_reason and result.scored is None
    assert "audit" in caplog.text and "db locked" in caplog.text


def test_score_failure_is_logged_not_raised(session, asset, monkeypatch, caplog):
    _add_bar(session, asset, date(2026, 10, 16))
    Calls(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("talib exploded")
    monkeypatch.setattr(jobs.scorer, "score", boom)
    with caplog.at_level(logging.INFO, logger=jobs.logger.name):
        result = jobs.run_post_close(session)

    assert result.scored is None and result.skipped_reason
    assert "score" in caplog.text and "talib exploded" in caplog.text


def _walk(n, seed):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return close


def test_unaffected_asset_gets_the_same_score_row_as_before(session, asset, monkeypatch):
    """Real scorer: B's technical_scores row is the one score_universe would write."""
    b = _add_asset(session, "B.NS")
    days = pd.bdate_range("2026-04-01", periods=120).date
    for a, seed in ((asset, 1), (b, 2)):
        for d, c in zip(days, _walk(120, seed)):
            session.add(PriceDaily(asset_id=a.id, date=d, open=c, high=c * 1.01, low=c * 0.99,
                                   close=c, adj_close=c, volume=1_000))
    session.commit()
    expected = scorer.score("B.NS", "1d", session=session, persist=False)
    assert expected.status in ("ok", "warmup")

    # corrupt A's latest bar (inside the window)
    last = session.query(PriceDaily).filter_by(asset_id=asset.id).order_by(PriceDaily.date.desc()).first()
    last.close = None
    session.commit()
    monkeypatch.setattr(jobs, "ingest_all_prices", lambda s, asset_type="stock": {})

    result = jobs.run_post_close(session)

    assert result.excluded == ["TEST.NS"] and result.scored == {expected.status: 1}
    rows = session.query(TechnicalScoreRecord).all()
    assert [r.asset_id for r in rows] == [b.id]
    got = rows[0]
    assert got.timestamp == pd.Timestamp(expected.as_of).to_pydatetime()
    if math.isnan(expected.tech_score):
        assert got.tech_score is None
    else:
        assert got.tech_score == pytest.approx(expected.tech_score)
    assert got.regime == expected.regime


def test_job_wrapper_owns_and_closes_session(session, monkeypatch):
    closed = []

    class FakeSession:
        def close(self):
            closed.append(True)

    seen = []
    monkeypatch.setattr(jobs, "SessionLocal", lambda: FakeSession())
    monkeypatch.setattr(jobs, "run_post_close", lambda s: seen.append(s))
    jobs._job_daily_prices()
    assert len(seen) == 1 and closed == [True]

    def boom(s):
        raise RuntimeError("unexpected")
    monkeypatch.setattr(jobs, "run_post_close", boom)
    jobs._job_daily_prices()   # the scheduler thread must survive
    assert closed == [True, True]


def test_create_scheduler_jobs_unchanged():
    sched = jobs.create_scheduler()
    got = {j.id: j for j in sched.get_jobs()}
    assert set(got) == {"intraday_refresh", "daily_prices", "quarterly_financials"}

    def fields(job):
        return {f.name: str(f) for f in job.trigger.fields}

    assert fields(got["daily_prices"])["hour"] == "16"
    assert fields(got["daily_prices"])["minute"] == "0"
    assert fields(got["daily_prices"])["day_of_week"] == "mon-fri"
    assert fields(got["intraday_refresh"])["hour"] == "9-15"
    assert fields(got["intraday_refresh"])["minute"] == "0,30"
    assert fields(got["quarterly_financials"])["day"] == "1"
    assert fields(got["quarterly_financials"])["hour"] == "6"
    assert got["daily_prices"].func is jobs._job_daily_prices
