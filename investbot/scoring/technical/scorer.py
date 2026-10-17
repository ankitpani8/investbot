"""
Technical scorer: ``score(ticker)`` orchestrator, the only technical module with DB I/O.

Loads one asset's bars from the DB, runs the pure compute modules
(``indicators``, ``candle_patterns``, ``support_resistance``, ``regime``,
``multi_timeframe``, ``confluence``) on one frame, reads the last bar and
persists it to ``technical_scores``.

Timeframes
    "1d"    daily bars: indicators + candles + S/R + regime → confluence.
            The MTF family is absent (NaN) and drops out of the weights.
    "15m"   15m bars (DB open-time labels): indicators + candles + S/R (intraday
            volume-profile window) + ``mtf_alignment`` against daily bars.
            REGIME / REGIME_CONFIDENCE are the *daily* regime as seen by the
            15m bar (MTF_REGIME_1D), because regime.py is calibrated on daily
            bars only.

No lookahead (``as_of``)
    Only bars known at ``as_of`` are loaded, so ``score(t, as_of=x)`` equals
    scoring a DB that ends at x. This is the backtest entry point.
        date        daily bars dated <= as_of; all 15m bars of that date.
        datetime    a daily bar of date D counts from D 16:00 IST
                    (``DAILY_AVAILABLE_AT``); a 15m bar opened at t counts from
                    t + 15 min (its close). Naive datetimes are IST, like the DB.
    At most ``DAILY_LOOKBACK_BARS`` / ``INTRADAY_LOOKBACK_BARS`` bars before
    the cutoff are loaded, so a score depends on a fixed-length window, not
    on how far back the DB happens to go (memory stays bounded too).

Cache
    A stored row for (asset, timeframe, last bar) is reused when its
    ``input_hash`` (a fingerprint of every OHLCV value in the loaded window)
    and ``scorer_version`` match. A revised bar anywhere in the window (the
    ingestion refresh window rewrites recent bars) changes the hash and forces
    a recompute. Bump ``SCORER_VERSION`` when compute logic changes.
    Staleness is always re-evaluated against the current reference time.

Failure modes never raise: an unknown ticker, no bars, too few bars, warmup
or a compute error comes back as a ``TechnicalScore`` with ``status`` set and
``tech_score`` NaN. Only ``status in {"ok", "warmup", "insufficient_data"}``
rows are persisted; errors are logged and not cached.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from investbot.config.scoring_weights import FAMILIES, REGIME_NAMES
from investbot.config.settings import TIMEZONE
from investbot.data.models import Asset, PriceDaily, PriceIntraday, TechnicalScoreRecord
from investbot.scoring.technical.candle_patterns import compute_candle_patterns
from investbot.scoring.technical.confluence import compute_confluence
from investbot.scoring.technical.indicators import add_all_indicators
from investbot.scoring.technical.multi_timeframe import (
    BASE_BAR,
    DAILY_AVAILABLE_AT,
    mtf_alignment,
)
from investbot.scoring.technical.regime import compute_regime
from investbot.scoring.technical.support_resistance import (
    VP_WINDOW_INTRADAY,
    compute_support_resistance_columns,
)

logger = logging.getLogger(__name__)

SUPPORTED_TIMEFRAMES: tuple[str, ...] = ("1d", "15m")
#: Daily bars loaded per score (~3 years). On RELIANCE.NS daily, 750 bars give
#: the same TECH_SCORE as the full 5-year history to ~1e-5 at half the cost.
DAILY_LOOKBACK_BARS: int = 750
#: 15m bars loaded per intraday score (~64 sessions, all yfinance keeps).
INTRADAY_LOOKBACK_BARS: int = 1600
#: Below this many bars the compute modules are not run at all.
MIN_BARS: int = 30
#: A last bar more than this many business days before the reference time is stale.
STALE_AFTER_BUSINESS_DAYS: int = 3
#: Part of the cache key. Bump whenever a compute module's output changes.
SCORER_VERSION: str = "1"

OHLCV: tuple[str, ...] = ("open", "high", "low", "close", "volume")
_IST = ZoneInfo(TIMEZONE)
_PERSISTED_STATUSES = ("ok", "warmup", "insufficient_data")


@dataclass
class TechnicalScore:
    """Technical score of one asset at one bar, plus data-quality flags."""
    ticker: str
    timeframe: str
    as_of: pd.Timestamp | None                  # timestamp of the scored (last) bar
    tech_score: float = math.nan                # [-1, 1]; NaN unless status == "ok"
    family_scores: dict[str, float] = field(default_factory=dict)
    regime: str | None = None                   # 'trend' | 'range' | 'chaotic'
    regime_confidence: float = math.nan
    reasons: list[str] = field(default_factory=list)
    # data quality
    status: str = "ok"      # ok | warmup | insufficient_data | no_data | unknown_asset | error
    bars_available: int = 0
    warmup_satisfied: bool = False
    staleness_days: int | None = None           # business days from the bar to the reference time
    is_stale: bool = False
    cached: bool = False
    message: str = ""

    def to_row(self) -> dict:
        """Flat dict for tables (score_universe, CLI)."""
        return {
            "ticker": self.ticker,
            "timeframe": self.timeframe,
            "as_of": self.as_of,
            "tech_score": self.tech_score,
            **{f"score_{f}": self.family_scores.get(f, math.nan) for f in FAMILIES},
            "regime": self.regime,
            "regime_confidence": self.regime_confidence,
            "status": self.status,
            "bars_available": self.bars_available,
            "warmup_satisfied": self.warmup_satisfied,
            "staleness_days": self.staleness_days,
            "is_stale": self.is_stale,
            "cached": self.cached,
            "top_reason": self.reasons[0] if self.reasons else "",
        }


# ──────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────

def score(
    ticker: str,
    timeframe: str = "1d",
    as_of: date | datetime | None = None,
    *,
    session: Session | None = None,
    persist: bool = True,
    use_cache: bool = True,
) -> TechnicalScore:
    """
    Technical score of ``ticker`` at the last bar known at ``as_of`` (default: now).

    Parameters
    ----------
    timeframe : {"1d", "15m"}
    as_of : date or datetime, optional
        Only bars known at this time are loaded (see the module docstring).
    session : Session, optional
        Defaults to a new ``SessionLocal()``, closed on return.
    persist : bool
        Upsert the result into ``technical_scores``.
    use_cache : bool
        Reuse a stored row for the same bar and input window.
    """
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"timeframe must be one of {SUPPORTED_TIMEFRAMES}, got {timeframe!r}")
    own = session is None
    if own:
        from investbot.data.db import SessionLocal
        session = SessionLocal()
    try:
        return _score(session, ticker, timeframe, as_of, persist=persist, use_cache=use_cache)
    finally:
        if own:
            session.close()


def score_universe(
    asset_type: str = "stock",
    timeframe: str = "1d",
    as_of: date | datetime | None = None,
    *,
    session: Session | None = None,
    persist: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Score every active asset of ``asset_type``, one asset at a time.

    Returns one row per asset (``TechnicalScore.to_row()`` columns plus
    ``rank``), best ``tech_score`` first; flagged assets (NaN score) last.
    """
    own = session is None
    if own:
        from investbot.data.db import SessionLocal
        session = SessionLocal()
    try:
        tickers = session.execute(
            select(Asset.ticker)
            .where(Asset.is_active.is_(True), Asset.asset_type == asset_type)
            .order_by(Asset.ticker)
        ).scalars().all()
        rows = []
        for t in tickers:
            rows.append(score(t, timeframe, as_of, session=session,
                              persist=persist, use_cache=use_cache).to_row())
    finally:
        if own:
            session.close()

    columns = list(TechnicalScore("", timeframe, None).to_row())
    df = pd.DataFrame(rows, columns=columns)
    df["tech_score"] = df["tech_score"].astype(float)
    df = df.sort_values(["tech_score", "ticker"], ascending=[False, True],
                        na_position="last", kind="stable").reset_index(drop=True)
    df.insert(0, "rank", np.where(df["tech_score"].notna(), np.arange(1, len(df) + 1), 0))
    return df


def ensure_schema(session: Session) -> None:
    """Create ``technical_scores`` if missing (same create-if-absent path as ``init_db``)."""
    TechnicalScoreRecord.__table__.create(bind=session.get_bind(), checkfirst=True)


# ──────────────────────────────────────────────────────────
# Pure compute (no DB)
# ──────────────────────────────────────────────────────────

def compute_daily_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily OHLCV → every upstream column + confluence, on the daily index."""
    ind = add_all_indicators(daily)
    upstream = pd.concat([
        ind,
        pd.DataFrame(compute_candle_patterns(daily), index=daily.index),
        pd.DataFrame(compute_support_resistance_columns(daily), index=daily.index),
        pd.DataFrame(compute_regime(daily), index=daily.index),
    ], axis=1)
    return pd.concat([upstream, pd.DataFrame(compute_confluence(upstream), index=daily.index)],
                     axis=1)


def compute_intraday_frame(bars_15m: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """15m OHLCV (open-time labels) + daily OHLCV → upstream + MTF + confluence on the 15m index."""
    mtf = mtf_alignment(bars_15m, daily, input_label="open")
    mtf.index = mtf.index - BASE_BAR              # close-time labels → DB open-time labels
    mtf = mtf.reindex(bars_15m.index)
    ind = add_all_indicators(bars_15m)
    upstream = pd.concat([
        ind,
        pd.DataFrame(compute_candle_patterns(bars_15m), index=bars_15m.index),
        pd.DataFrame(compute_support_resistance_columns(bars_15m, vp_window=VP_WINDOW_INTRADAY),
                     index=bars_15m.index),
        mtf,
        pd.DataFrame({"REGIME": mtf["MTF_REGIME_1D"].to_numpy(),
                      "REGIME_CONFIDENCE": mtf["MTF_REGIME_CONFIDENCE_1D"].to_numpy()},
                     index=bars_15m.index),
    ], axis=1)
    return pd.concat([upstream, pd.DataFrame(compute_confluence(upstream), index=bars_15m.index)],
                     axis=1)


# ──────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────

def _score(session, ticker, timeframe, as_of, *, persist, use_cache) -> TechnicalScore:
    asset = session.execute(select(Asset).where(Asset.ticker == ticker)).scalars().first()
    if asset is None:
        return TechnicalScore(ticker, timeframe, None, status="unknown_asset",
                              message=f"no asset with ticker {ticker!r}")

    daily_cut, intraday_cut = _cutoffs(as_of)
    if timeframe == "1d":
        bars = _load_daily(session, asset.id, daily_cut, DAILY_LOOKBACK_BARS)
        context = None
    else:
        bars = _load_intraday(session, asset.id, "15m", intraday_cut, INTRADAY_LOOKBACK_BARS)
        context = _load_daily(session, asset.id, daily_cut, DAILY_LOOKBACK_BARS)

    if bars.empty:
        return TechnicalScore(ticker, timeframe, None, status="no_data",
                              message="no bars at or before as_of")

    last = bars.index[-1]
    staleness = _staleness(last, as_of)
    flags = dict(bars_available=len(bars), staleness_days=staleness,
                 is_stale=staleness > STALE_AFTER_BUSINESS_DAYS)
    digest = _fingerprint(bars, context)

    if use_cache:
        ensure_schema(session)
        rec = _find_record(session, asset.id, timeframe, last)
        if rec is not None and rec.input_hash == digest and rec.scorer_version == SCORER_VERSION:
            return _from_record(rec, ticker, **flags)

    if len(bars) < MIN_BARS:
        result = TechnicalScore(ticker, timeframe, last, status="insufficient_data",
                                message=f"{len(bars)} bars < MIN_BARS={MIN_BARS}", **flags)
    else:
        try:
            if timeframe == "1d":
                frame = compute_daily_frame(bars)
            else:
                frame = compute_intraday_frame(bars, context)
        except Exception as exc:        # one bad asset must not stop a universe run
            logger.warning("scoring %s %s failed: %s", ticker, timeframe, exc, exc_info=True)
            return TechnicalScore(ticker, timeframe, last, status="error",
                                  message=f"{type(exc).__name__}: {exc}", **flags)
        result = _from_frame(frame.iloc[-1], ticker, timeframe, last, **flags)

    if persist:
        ensure_schema(session)
        _upsert(session, asset.id, result, digest)
    return result


def _cutoffs(as_of) -> tuple[date | None, datetime | None]:
    """(last visible daily date, latest allowed 15m bar *open* time, exclusive end)."""
    if as_of is None:
        return None, None
    if isinstance(as_of, datetime):
        ts = pd.Timestamp(as_of)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(_IST).tz_localize(None)
        d = ts.date()
        daily = d if ts.time() >= DAILY_AVAILABLE_AT else d - timedelta(days=1)
        return daily, (ts - BASE_BAR).to_pydatetime()       # open <= as_of - 15m ⇔ closed
    # a plain date: the whole day is known
    return as_of, datetime.combine(as_of, datetime.max.time()) - BASE_BAR


def _load_daily(session, asset_id: int, cutoff: date | None, limit: int) -> pd.DataFrame:
    stmt = select(PriceDaily.date, PriceDaily.open, PriceDaily.high, PriceDaily.low,
                  PriceDaily.close, PriceDaily.volume).where(PriceDaily.asset_id == asset_id)
    if cutoff is not None:
        stmt = stmt.where(PriceDaily.date <= cutoff)
    rows = session.execute(stmt.order_by(PriceDaily.date.desc()).limit(limit)).all()
    return _frame(rows[::-1])


def _load_intraday(session, asset_id: int, interval: str, cutoff: datetime | None,
                   limit: int) -> pd.DataFrame:
    stmt = select(PriceIntraday.timestamp, PriceIntraday.open, PriceIntraday.high,
                  PriceIntraday.low, PriceIntraday.close, PriceIntraday.volume).where(
        PriceIntraday.asset_id == asset_id, PriceIntraday.interval == interval)
    if cutoff is not None:
        stmt = stmt.where(PriceIntraday.timestamp <= cutoff)
    rows = session.execute(stmt.order_by(PriceIntraday.timestamp.desc()).limit(limit)).all()
    return _frame(rows[::-1])


def _frame(rows) -> pd.DataFrame:
    idx = pd.DatetimeIndex([r[0] for r in rows], name="timestamp")
    data = np.array([r[1:] for r in rows], dtype=np.float64).reshape(len(rows), len(OHLCV))
    return pd.DataFrame(data, index=idx, columns=list(OHLCV))


def _fingerprint(bars: pd.DataFrame, context: pd.DataFrame | None) -> str:
    h = hashlib.sha1()
    for df in (bars, context):
        if df is None:
            continue
        h.update(df.index.as_unit("ns").asi8.tobytes())
        h.update(np.ascontiguousarray(df[list(OHLCV)].to_numpy(np.float64)).tobytes())
    return h.hexdigest()


def _staleness(last_bar: pd.Timestamp, as_of) -> int:
    if as_of is None:
        ref = datetime.now(_IST).date()
    elif isinstance(as_of, datetime):
        ts = pd.Timestamp(as_of)
        ref = (ts.tz_convert(_IST) if ts.tzinfo is not None else ts).date()
    else:
        ref = as_of
    return int(np.busday_count(last_bar.date(), ref)) if ref > last_bar.date() else 0


def _nan_or(x) -> float:
    return math.nan if x is None or pd.isna(x) else float(x)


def _from_frame(row: pd.Series, ticker, timeframe, last, **flags) -> TechnicalScore:
    tech = _nan_or(row.get("TECH_SCORE"))
    code = row.get("REGIME")
    regime = None if code is None or pd.isna(code) else REGIME_NAMES[int(code)]
    reasons = row.get("TECH_REASONS")
    ok = not math.isnan(tech)
    return TechnicalScore(
        ticker, timeframe, last,
        tech_score=tech,
        family_scores={f: _nan_or(row.get(f"TECH_SCORE_{f.upper()}")) for f in FAMILIES},
        regime=regime,
        regime_confidence=_nan_or(row.get("REGIME_CONFIDENCE")),
        reasons=list(reasons) if isinstance(reasons, (list, tuple, np.ndarray)) else [],
        status="ok" if ok else "warmup",
        warmup_satisfied=ok,
        **flags,
    )


def _find_record(session, asset_id, timeframe, last) -> TechnicalScoreRecord | None:
    return session.execute(select(TechnicalScoreRecord).where(
        TechnicalScoreRecord.asset_id == asset_id,
        TechnicalScoreRecord.timeframe == timeframe,
        TechnicalScoreRecord.timestamp == last.to_pydatetime(),
    )).scalars().first()


def _from_record(rec: TechnicalScoreRecord, ticker, **flags) -> TechnicalScore:
    return TechnicalScore(
        ticker, rec.timeframe, pd.Timestamp(rec.timestamp),
        tech_score=_nan_or(rec.tech_score),
        family_scores={f: _nan_or(getattr(rec, f"score_{f}")) for f in FAMILIES},
        regime=rec.regime,
        regime_confidence=_nan_or(rec.regime_confidence),
        reasons=json.loads(rec.signal_reasons) if rec.signal_reasons else [],
        status=rec.status,
        warmup_satisfied=bool(rec.warmup_satisfied),
        cached=True,
        **flags,
    )


def _db_float(x: float) -> float | None:
    return None if x is None or math.isnan(x) else float(x)


def _upsert(session, asset_id: int, s: TechnicalScore, digest: str) -> None:
    if s.status not in _PERSISTED_STATUSES or s.as_of is None:
        return
    rec = _find_record(session, asset_id, s.timeframe, s.as_of)
    if rec is None:
        rec = TechnicalScoreRecord(asset_id=asset_id, timeframe=s.timeframe,
                                   timestamp=s.as_of.to_pydatetime())
        session.add(rec)
    rec.tech_score = _db_float(s.tech_score)
    for f in FAMILIES:
        setattr(rec, f"score_{f}", _db_float(s.family_scores.get(f, math.nan)))
    rec.regime = s.regime
    rec.regime_confidence = _db_float(s.regime_confidence)
    rec.signal_reasons = json.dumps(s.reasons)
    rec.status = s.status
    rec.bars_available = s.bars_available
    rec.warmup_satisfied = s.warmup_satisfied
    rec.staleness_days = s.staleness_days
    rec.input_hash = digest
    rec.scorer_version = SCORER_VERSION
    rec.computed_at = datetime.now(_IST).replace(tzinfo=None)
    session.commit()
