"""Asset selection for scripts/smoke_indicators.py (in-memory SQLite, no TA-Lib run)."""
from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from investbot.data.models import Asset, Base, PriceDaily
from scripts import smoke_indicators as smoke


@pytest.fixture
def session():
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    yield s
    s.close()
    engine.dispose()


def _asset(session, ticker, n_rows):
    a = Asset(exchange_id=1, ticker=ticker, name=ticker, asset_type="stock")
    session.add(a)
    session.flush()
    for i in range(n_rows):
        session.add(PriceDaily(asset_id=a.id, date=date(2026, 1, 1) + timedelta(days=i),
                               open=100.0, high=105.0, low=99.0, close=103.0,
                               adj_close=103.0, volume=1_000))
    session.commit()
    return a


def test_default_ticker_used_when_present_even_if_not_the_largest(session):
    _asset(session, smoke.DEFAULT_SMOKE_TICKER, 3)
    _asset(session, "ITC.NS", 10)
    asset, n, warning = smoke.select_smoke_asset(session)
    assert asset.ticker == smoke.DEFAULT_SMOKE_TICKER and n == 3 and warning is None


def test_default_missing_rows_falls_back_with_warning(session):
    _asset(session, smoke.DEFAULT_SMOKE_TICKER, 0)
    _asset(session, "ITC.NS", 10)
    _asset(session, "TCS.NS", 4)
    asset, n, warning = smoke.select_smoke_asset(session)
    assert asset.ticker == "ITC.NS" and n == 10
    assert smoke.DEFAULT_SMOKE_TICKER in warning and "ITC.NS" in warning


def test_default_absent_from_db_falls_back_with_warning(session):
    _asset(session, "ITC.NS", 10)
    asset, _, warning = smoke.select_smoke_asset(session)
    assert asset.ticker == "ITC.NS" and warning


def test_explicit_ticker_is_used(session):
    _asset(session, smoke.DEFAULT_SMOKE_TICKER, 10)
    _asset(session, "ITC.NS", 2)
    asset, n, warning = smoke.select_smoke_asset(session, "ITC.NS")
    assert asset.ticker == "ITC.NS" and n == 2 and warning is None


def test_unknown_explicit_ticker_errors(session):
    _asset(session, smoke.DEFAULT_SMOKE_TICKER, 10)
    with pytest.raises(smoke.SmokeAssetError, match="NOPE.NS"):
        smoke.select_smoke_asset(session, "NOPE.NS")


def test_explicit_ticker_without_rows_errors_instead_of_substituting(session):
    _asset(session, smoke.DEFAULT_SMOKE_TICKER, 10)
    _asset(session, "EMPTY.NS", 0)
    with pytest.raises(smoke.SmokeAssetError):
        smoke.select_smoke_asset(session, "EMPTY.NS")


def test_empty_db_errors(session):
    with pytest.raises(smoke.SmokeAssetError):
        smoke.select_smoke_asset(session)


def test_main_exits_nonzero_for_unknown_ticker(session, monkeypatch):
    monkeypatch.setattr(smoke, "SessionLocal", lambda: session)
    assert smoke.main(["NOPE.NS"]) == 1
