"""
Price-row data quality: one set of rules shared by ingestion and `main.py dq`.

A row that fails validation is never stored. Ingestion skips it and logs the
reason; because the date is not stored, the next ingest run fetches it again.
Indicator code relies on this: a single NaN close poisons every running-state
TA-Lib indicator (SMA, EMA, MACD, Bollinger) for the rest of the series and
turns others (RSI, HT_TRENDLINE) into plausible-looking wrong numbers.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from investbot.data.models import Asset, PriceDaily, PriceIntraday

logger = logging.getLogger(__name__)

# Rejection reasons (stable strings: they are printed by `dq` and asserted in tests).
MISSING_OHLC = "missing_ohlc"
NON_POSITIVE_PRICE = "non_positive_price"
INCONSISTENT_OHLC = "inconsistent_ohlc"
MISSING_VOLUME = "missing_volume"
NEGATIVE_VOLUME = "negative_volume"
ZERO_VOLUME_FLAT = "zero_volume_flat"

# Relative slack for the high/low envelope check; prices come from float32 sources.
_ENVELOPE_TOL = 1e-6


def _is_missing(x) -> bool:
    """True for None, NaN, +/-inf, pd.NA and anything non-numeric."""
    if x is None:
        return True
    try:
        return not math.isfinite(float(x))
    except (TypeError, ValueError):
        return True


def validate_price_row(o, h, l, c, volume, *, daily: bool) -> str | None:
    """
    Return the rejection reason for one bar, or None if the bar is storable.

    `daily=True` additionally rejects zero-volume bars with O=H=L=C. yfinance
    emits those as carry-forward placeholders for sessions with no real data
    (holidays, and days it failed to populate); they are not trades. Intraday
    bars do not get this rule, a quiet 15m bar can legitimately be flat.
    """
    prices = (o, h, l, c)
    if any(_is_missing(p) for p in prices):
        return MISSING_OHLC
    o, h, l, c = (float(p) for p in prices)
    if min(o, h, l, c) <= 0:
        return NON_POSITIVE_PRICE
    if h < max(o, l, c) * (1 - _ENVELOPE_TOL) or l > min(o, h, c) * (1 + _ENVELOPE_TOL):
        return INCONSISTENT_OHLC

    if _is_missing(volume):
        return MISSING_VOLUME
    if volume < 0:
        return NEGATIVE_VOLUME
    if daily and volume == 0 and o == h == l == c:
        return ZERO_VOLUME_FLAT
    return None


# ──────────────────────────────────────────────────────────
# Database audit
# ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BadRow:
    table: str                       # 'price_daily' | 'price_intraday'
    row_id: int
    asset_id: int
    ticker: str
    when: date | datetime
    reason: str


def audit_prices(session: Session) -> list[BadRow]:
    """Run the ingestion rules over every stored price row. Read-only."""
    tickers = dict(session.execute(select(Asset.id, Asset.ticker)).all())
    bad: list[BadRow] = []

    daily = select(
        PriceDaily.id, PriceDaily.asset_id, PriceDaily.date,
        PriceDaily.open, PriceDaily.high, PriceDaily.low, PriceDaily.close,
        PriceDaily.volume,
    ).order_by(PriceDaily.asset_id, PriceDaily.date)
    for rid, aid, when, o, h, l, c, v in session.execute(daily).yield_per(5000):
        reason = validate_price_row(o, h, l, c, v, daily=True)
        if reason:
            bad.append(BadRow("price_daily", rid, aid, tickers.get(aid, str(aid)), when, reason))

    intraday = select(
        PriceIntraday.id, PriceIntraday.asset_id, PriceIntraday.timestamp,
        PriceIntraday.open, PriceIntraday.high, PriceIntraday.low, PriceIntraday.close,
        PriceIntraday.volume,
    ).order_by(PriceIntraday.asset_id, PriceIntraday.timestamp)
    for rid, aid, when, o, h, l, c, v in session.execute(intraday).yield_per(5000):
        reason = validate_price_row(o, h, l, c, v, daily=False)
        if reason:
            bad.append(BadRow("price_intraday", rid, aid, tickers.get(aid, str(aid)), when, reason))

    return bad


def purge_bad_rows(session: Session, bad: list[BadRow]) -> int:
    """Delete the given audited rows. Returns the number of rows deleted."""
    ids: dict[str, list[int]] = defaultdict(list)
    for b in bad:
        ids[b.table].append(b.row_id)

    deleted = 0
    for model in (PriceDaily, PriceIntraday):
        row_ids = ids.get(model.__tablename__, [])
        for i in range(0, len(row_ids), 500):          # stay under SQLite's variable limit
            chunk = row_ids[i:i + 500]
            deleted += session.execute(delete(model).where(model.id.in_(chunk))).rowcount
    session.commit()
    return deleted
