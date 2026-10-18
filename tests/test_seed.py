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


def _equity_regimes(session):
    from investbot.data.models import TaxRegime

    return {r.regime_name: r for r in session.query(TaxRegime).filter(
        TaxRegime.regime_name.like("IN_EQUITY%"))}


def test_seed_tax_regimes_dated_and_idempotent(session):
    from datetime import date
    from investbot.data.models import TaxRegime

    run_seed(session)
    n = session.query(TaxRegime).count()
    run_seed(session)
    assert session.query(TaxRegime).count() == n
    rows = _equity_regimes(session)
    assert set(rows) == {"IN_EQUITY_LISTED", "IN_EQUITY_LISTED_PRE_2024_07_23"}
    assert rows["IN_EQUITY_LISTED"].effective_from == date(2024, 7, 23)
    old = rows["IN_EQUITY_LISTED_PRE_2024_07_23"]
    assert (old.stcg_rate, old.ltcg_rate, old.ltcg_annual_exemption) == (0.15, 0.10, 100000)
    assert (old.effective_from, old.effective_to) == (date(2018, 4, 1), date(2024, 7, 22))
    assert old.asset_class_id == rows["IN_EQUITY_LISTED"].asset_class_id is not None


def test_seed_tax_regimes_updates_existing_row(session):
    from datetime import date

    run_seed(session)
    row = _equity_regimes(session)["IN_EQUITY_LISTED"]
    row.effective_from = date(2025, 4, 1)  # as seeded before 2024-07-23 was added
    row.stcg_rate = 0.0
    session.commit()

    run_seed(session)
    session.expire_all()
    row = _equity_regimes(session)["IN_EQUITY_LISTED"]
    assert row.effective_from == date(2024, 7, 23)
    assert row.stcg_rate == 0.20
