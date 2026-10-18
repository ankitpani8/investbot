"""
The evaluation harness's only DB access: the stock universe (active stocks, or
every past and present member of an index from ``index_membership``), dated
membership, and one asset's daily OHLCV history at a time (8 GB RAM: never the
whole universe).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import exists, or_, select
from sqlalchemy.orm import Session

from investbot.data.index_membership import membership_matrix
from investbot.data.models import Asset, IndexMembership, PriceDaily

OHLCV: tuple[str, ...] = ("open", "high", "low", "close", "volume")


def active_tickers(session: Session, asset_type: str = "stock") -> list[str]:
    return list(session.execute(
        select(Asset.ticker)
        .where(Asset.is_active.is_(True), Asset.asset_type == asset_type)
        .order_by(Asset.ticker)
    ).scalars().all())


def universe_tickers(session: Session, index_code: str, start: date | None = None,
                     end: date | None = None) -> list[str]:
    """Every asset that was a member of ``index_code`` at any point in [start, end]
    (open-ended where None), active or not, with or without price rows."""
    stmt = (select(Asset.ticker).distinct()
            .join(IndexMembership, IndexMembership.asset_id == Asset.id)
            .where(IndexMembership.index_code == index_code))
    if end is not None:
        stmt = stmt.where(IndexMembership.effective_from <= end)
    if start is not None:
        stmt = stmt.where(or_(IndexMembership.effective_to.is_(None),
                              IndexMembership.effective_to >= start))
    return sorted(session.execute(stmt).scalars().all())


def has_index_membership(session: Session, index_code: str) -> bool:
    return bool(session.execute(
        select(exists().where(IndexMembership.index_code == index_code))).scalar())


def membership_intervals(session: Session, index_code: str,
                         tickers: list[str] | None = None) -> pd.DataFrame:
    """``index_membership`` stints as (ticker, effective_from, effective_to) rows."""
    stmt = (select(Asset.ticker, IndexMembership.effective_from, IndexMembership.effective_to)
            .join(Asset, Asset.id == IndexMembership.asset_id)
            .where(IndexMembership.index_code == index_code))
    if tickers is not None:
        stmt = stmt.where(Asset.ticker.in_(list(tickers)))
    rows = session.execute(stmt).all()
    return pd.DataFrame([tuple(r) for r in rows],
                        columns=["ticker", "effective_from", "effective_to"])


def membership_for(session: Session, index_code: str, tickers: list[str], dates) -> pd.DataFrame:
    """Boolean (dates × tickers): member of ``index_code`` on that date.

    A stint counts from its ``effective_from`` (the first session the change
    applies to), so the value on t depends only on changes effective <= t.
    Tickers without rows are False everywhere. ``COMPLETE_FROM`` is not applied
    here (see ``panel.build_dataset``).
    """
    return membership_matrix(membership_intervals(session, index_code, tickers), dates,
                             list(tickers))


def tickers_with_prices(session: Session, tickers: list[str]) -> set[str]:
    """Subset of ``tickers`` with at least one daily price row."""
    return set(session.execute(
        select(Asset.ticker).where(Asset.ticker.in_(list(tickers)),
                                   exists().where(PriceDaily.asset_id == Asset.id))
    ).scalars().all())


def load_daily_history(session: Session, ticker: str, end: date | None = None) -> pd.DataFrame:
    """Every stored daily bar of ``ticker`` dated <= ``end`` (all if None), oldest first."""
    stmt = (select(PriceDaily.date, PriceDaily.open, PriceDaily.high, PriceDaily.low,
                   PriceDaily.close, PriceDaily.volume)
            .join(Asset, Asset.id == PriceDaily.asset_id)
            .where(Asset.ticker == ticker))
    if end is not None:
        stmt = stmt.where(PriceDaily.date <= end)
    rows = session.execute(stmt.order_by(PriceDaily.date)).all()
    idx = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows], name="date")
    data = np.array([r[1:] for r in rows], dtype=np.float64).reshape(len(rows), len(OHLCV))
    return pd.DataFrame(data, index=idx, columns=list(OHLCV))
