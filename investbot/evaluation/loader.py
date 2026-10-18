"""
The evaluation harness's only DB access: the active stock universe and one
asset's daily OHLCV history at a time (8 GB RAM: never the whole universe).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from investbot.data.models import Asset, PriceDaily

OHLCV: tuple[str, ...] = ("open", "high", "low", "close", "volume")


def active_tickers(session: Session, asset_type: str = "stock") -> list[str]:
    return list(session.execute(
        select(Asset.ticker)
        .where(Asset.is_active.is_(True), Asset.asset_type == asset_type)
        .order_by(Asset.ticker)
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
