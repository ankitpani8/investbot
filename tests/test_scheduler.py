"""
Scheduler post-close pipeline: ingest -> audit -> score, with bad data blocking scoring.

In-memory SQLite; ingest_all_prices and score_universe are monkeypatched. No network.
"""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from investbot.data.models import Asset, Base, PriceDaily
from investbot.scheduler import jobs


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


class Calls:
    def __init__(self, monkeypatch, *, ingest_raises=False):
        self.ingest = 0
        self.score = 0
        self.score_kwargs = None

        def fake_ingest(session, asset_type="stock"):
            self.ingest += 1
            if ingest_raises:
                raise RuntimeError("yfinance down")
            return {"daily_total": 5, "intraday_total": 0, "errors": []}

        def fake_score(*args, **kwargs):
            self.score += 1
            self.score_kwargs = (args, kwargs)
            return pd.DataFrame({"status": ["ok", "ok", "warmup", "no_data"]})

        monkeypatch.setattr(jobs, "ingest_all_prices", fake_ingest)
        monkeypatch.setattr(jobs.scorer, "score_universe", fake_score)


def test_clean_audit_scores_once(session, asset, monkeypatch, caplog):
    _add_bar(session, asset, date(2026, 10, 16))
    calls = Calls(monkeypatch)
    with caplog.at_level(logging.INFO, logger=jobs.logger.name):
        result = jobs.run_post_close(session)

    assert calls.ingest == 1 and calls.score == 1
    _, kwargs = calls.score_kwargs
    assert kwargs["asset_type"] == "stock" and kwargs["timeframe"] == "1d"
    assert result.bad_rows == 0
    assert result.scored == {"ok": 2, "warmup": 1, "no_data": 1}
    assert result.skipped_reason is None
    assert result.ingested["daily_total"] == 5
    assert "ok=2 warmup=1 insufficient_data=0 no_data=1 error=0" in caplog.text


def test_bad_row_blocks_scoring(session, asset, monkeypatch, caplog):
    _add_bar(session, asset, date(2026, 10, 15))
    _add_bar(session, asset, date(2026, 10, 16), close=None)   # incomplete OHLC
    calls = Calls(monkeypatch)
    with caplog.at_level(logging.INFO, logger=jobs.logger.name):
        result = jobs.run_post_close(session)

    assert calls.ingest == 1 and calls.score == 0
    assert result.scored is None
    assert result.bad_rows == 1
    assert "dq" in result.skipped_reason
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("price_daily" in r.getMessage() and "TEST.NS" in r.getMessage() for r in errors)
    # Nothing was purged.
    assert session.query(PriceDaily).count() == 2


def test_ingest_failure_skips_everything(session, asset, monkeypatch, caplog):
    calls = Calls(monkeypatch, ingest_raises=True)
    audited = []
    monkeypatch.setattr(jobs.quality, "audit_prices", lambda s: audited.append(s) or [])
    with caplog.at_level(logging.INFO, logger=jobs.logger.name):
        result = jobs.run_post_close(session)   # must not raise

    assert calls.score == 0 and audited == []
    assert result.ingested is None and result.bad_rows is None and result.scored is None
    assert result.skipped_reason
    assert "ingest" in caplog.text and "yfinance down" in caplog.text


def test_audit_failure_skips_scoring(session, asset, monkeypatch, caplog):
    calls = Calls(monkeypatch)

    def boom(s):
        raise RuntimeError("db locked")
    monkeypatch.setattr(jobs.quality, "audit_prices", boom)
    with caplog.at_level(logging.INFO, logger=jobs.logger.name):
        result = jobs.run_post_close(session)

    assert calls.score == 0
    assert result.skipped_reason and result.scored is None
    assert "audit" in caplog.text and "db locked" in caplog.text


def test_score_failure_is_logged_not_raised(session, asset, monkeypatch, caplog):
    Calls(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("talib exploded")
    monkeypatch.setattr(jobs.scorer, "score_universe", boom)
    with caplog.at_level(logging.INFO, logger=jobs.logger.name):
        result = jobs.run_post_close(session)

    assert result.scored is None and result.skipped_reason
    assert "score" in caplog.text and "talib exploded" in caplog.text


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
