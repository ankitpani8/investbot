"""Seed idempotency and retired-ticker handling (in-memory SQLite, no network)."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from investbot.config.tickers import NIFTY50_TICKERS, RETIRED_TICKERS
from investbot.data.models import Asset, Base, Exchange
from investbot.data.seed import run_seed


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


def _active(session) -> set[str]:
    return {a.ticker for a in session.query(Asset).filter_by(is_active=True)}


def test_retired_not_in_active_list():
    active = {t for t, *_ in NIFTY50_TICKERS}
    assert "TMPV.NS" in active
    assert not active & set(RETIRED_TICKERS)


def test_seed_idempotent(session):
    run_seed(session)
    n = session.query(Asset).count()
    run_seed(session)
    assert session.query(Asset).count() == n


def test_seed_deactivates_retired_and_keeps_row(session):
    # Simulate an existing DB seeded before the demerger
    run_seed(session)
    nse = session.query(Exchange).filter_by(code="NSE").one()
    session.add(Asset(exchange_id=nse.id, ticker="TATAMOTORS.NS", name="Tata Motors",
                      asset_type="stock", is_active=True))
    session.commit()

    run_seed(session)
    old = session.query(Asset).filter_by(ticker="TATAMOTORS.NS").one()
    assert old.is_active is False
    assert "TMPV.NS" in _active(session)
    assert "TATAMOTORS.NS" not in _active(session)
