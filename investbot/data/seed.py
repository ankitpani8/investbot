"""
Seed the database with reference data:
- Asset classes (equity, crypto)
- Exchanges (NSE, BINANCE)
- Assets (NIFTY50 tickers, crypto tickers)
- Tax regimes (Indian equity + crypto)
"""
import logging
from datetime import time, date

from sqlalchemy.orm import Session

from investbot.data.models import AssetClass, Exchange, Asset, TaxRegime
from investbot.config.tickers import NIFTY50_TICKERS, CRYPTO_TICKERS, RETIRED_TICKERS
from investbot.config.tax_regimes import TAX_REGIMES

logger = logging.getLogger(__name__)


def seed_asset_classes(session: Session) -> dict[str, AssetClass]:
    """Seed asset_classes table. Returns {name: AssetClass} mapping."""
    classes = {
        "equity": {
            "name": "equity",
            "description": "Listed equities (stocks, ETFs)",
            "scoring_interval_minutes": 30,
            "market_open": time(9, 15),
            "market_close": time(15, 30),
            "is_24x7": False,
        },
        "crypto": {
            "name": "crypto",
            "description": "Cryptocurrency / Virtual Digital Assets",
            "scoring_interval_minutes": 5,
            "market_open": None,
            "market_close": None,
            "is_24x7": True,
        },
    }

    result = {}
    for key, data in classes.items():
        existing = session.query(AssetClass).filter_by(name=key).first()
        if existing:
            result[key] = existing
            logger.info(f"Asset class '{key}' already exists (id={existing.id})")
        else:
            ac = AssetClass(**data)
            session.add(ac)
            session.flush()  # get the id
            result[key] = ac
            logger.info(f"Created asset class '{key}' (id={ac.id})")
    return result


def seed_exchanges(session: Session, asset_classes: dict[str, AssetClass]) -> dict[str, Exchange]:
    """Seed exchanges table. Returns {code: Exchange} mapping."""
    exchanges_data = [
        {
            "asset_class_id": asset_classes["equity"].id,
            "code": "NSE",
            "name": "National Stock Exchange of India",
            "country": "IN",
            "currency": "INR",
        },
        {
            "asset_class_id": asset_classes["crypto"].id,
            "code": "CRYPTO_GLOBAL",
            "name": "Global Crypto Markets (yfinance proxy)",
            "country": None,
            "currency": "USD",
        },
    ]

    result = {}
    for data in exchanges_data:
        existing = session.query(Exchange).filter_by(code=data["code"]).first()
        if existing:
            result[data["code"]] = existing
            logger.info(f"Exchange '{data['code']}' already exists (id={existing.id})")
        else:
            ex = Exchange(**data)
            session.add(ex)
            session.flush()
            result[data["code"]] = ex
            logger.info(f"Created exchange '{data['code']}' (id={ex.id})")
    return result


def seed_assets(session: Session, exchanges: dict[str, Exchange]):
    """Seed assets table with NIFTY50 + crypto tickers."""
    nse = exchanges["NSE"]
    crypto_ex = exchanges["CRYPTO_GLOBAL"]

    seen_tickers = set()
    added = 0

    # NIFTY50
    for ticker, name, sector, industry in NIFTY50_TICKERS:
        if ticker in seen_tickers:
            continue
        seen_tickers.add(ticker)
        existing = session.query(Asset).filter_by(exchange_id=nse.id, ticker=ticker).first()
        if not existing:
            session.add(Asset(
                exchange_id=nse.id,
                ticker=ticker,
                name=name,
                asset_type="stock",
                sector=sector,
                industry=industry,
                is_active=True,
            ))
            added += 1
    logger.info(f"NIFTY50: added {added} new tickers")

    # Retired symbols: deactivate, keep history
    retired = 0
    for ticker in RETIRED_TICKERS:
        existing = session.query(Asset).filter_by(exchange_id=nse.id, ticker=ticker).first()
        if existing and existing.is_active:
            existing.is_active = False
            retired += 1
    logger.info(f"NIFTY50: deactivated {retired} retired tickers")

    # Crypto
    crypto_added = 0
    for ticker, name, sector, industry in CRYPTO_TICKERS:
        existing = session.query(Asset).filter_by(exchange_id=crypto_ex.id, ticker=ticker).first()
        if not existing:
            session.add(Asset(
                exchange_id=crypto_ex.id,
                ticker=ticker,
                name=name,
                asset_type="crypto",
                sector=sector,
                industry=industry,
                is_active=True,
            ))
            crypto_added += 1
    logger.info(f"Crypto: added {crypto_added} new tickers")


def seed_tax_regimes(session: Session, asset_classes: dict[str, AssetClass]):
    """Seed tax_regimes; an existing row with the same regime_name is updated in place."""
    added = updated = 0
    for regime_data in TAX_REGIMES:
        data = {**regime_data}  # copy: don't mutate the module-level TAX_REGIMES
        ac = asset_classes.get(data.pop("asset_class_name"))
        data["asset_class_id"] = ac.id if ac else None

        # Convert date strings
        for key in ("effective_from", "effective_to"):
            if data.get(key):
                data[key] = date.fromisoformat(data[key])

        existing = session.query(TaxRegime).filter_by(regime_name=data["regime_name"]).first()
        if existing:
            for key, value in data.items():
                setattr(existing, key, value)
            updated += 1
            continue
        session.add(TaxRegime(**data))
        added += 1
    logger.info(f"Tax regimes: added {added} new entries, refreshed {updated}")


def run_seed(session: Session):
    """Run all seed functions in order."""
    logger.info("=== Starting database seed ===")
    asset_classes = seed_asset_classes(session)
    exchanges = seed_exchanges(session, asset_classes)
    seed_assets(session, exchanges)
    seed_tax_regimes(session, asset_classes)
    session.commit()
    logger.info("=== Database seed complete ===")
