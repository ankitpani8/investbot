"""
Tests for investbot.scoring.technical.scorer.

A throwaway SQLite file per test with synthetic bars: no network, and never
the real or scratch DB. The compute modules run for real.
"""
from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, delete, func, select, update
from sqlalchemy.orm import sessionmaker

pytest.importorskip("talib")

from investbot.data.models import (  # noqa: E402
    Asset, AssetClass, Base, Exchange, PriceDaily, PriceIntraday, TechnicalScoreRecord,
)
from investbot.scoring.technical import scorer as sc  # noqa: E402

N_DAILY = 900            # > DAILY_LOOKBACK_BARS, so the window cut is exercised
N_SESSIONS = 30          # 15m sessions at the end of the daily history
BARS_PER_SESSION = 25    # 09:15 … 15:15 bar opens


def _ohlcv(n: int, seed: int, vol: float = 0.012) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, vol, n)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, vol / 4, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    volume = rng.integers(100_000, 1_000_000, n)
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def _make_db(path, daily_bars: dict[str, int], intraday: tuple[str, ...] = (),
             inactive: tuple[str, ...] = ()):
    engine = create_engine(f"sqlite:///{path}", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    s = Session()
    ac = AssetClass(name="equity")
    ex = Exchange(asset_class=ac, code="NSE", name="NSE", currency="INR")
    s.add_all([ac, ex])
    s.flush()
    for k, (ticker, n) in enumerate(daily_bars.items()):
        a = Asset(exchange_id=ex.id, ticker=ticker, name=ticker, asset_type="stock",
                  is_active=ticker not in inactive)
        s.add(a)
        s.flush()
        if n == 0:
            continue
        days = pd.bdate_range(end="2026-10-15", periods=n)
        d = _ohlcv(n, seed=k + 1)
        s.add_all([PriceDaily(asset_id=a.id, date=day.date(), open=d["open"][i],
                              high=d["high"][i], low=d["low"][i], close=d["close"][i],
                              adj_close=d["close"][i], volume=int(d["volume"][i]))
                   for i, day in enumerate(days)])
        if ticker in intraday:
            sessions = days[-N_SESSIONS:]
            stamps = [datetime.combine(day.date(), time(9, 15)) + timedelta(minutes=15 * j)
                      for day in sessions for j in range(BARS_PER_SESSION)]
            m = _ohlcv(len(stamps), seed=100 + k, vol=0.002)
            s.add_all([PriceIntraday(asset_id=a.id, timestamp=t, interval="15m",
                                     open=m["open"][i], high=m["high"][i], low=m["low"][i],
                                     close=m["close"][i], volume=int(m["volume"][i]))
                       for i, t in enumerate(stamps)])
    s.commit()
    return s


@pytest.fixture
def db(tmp_path):
    s = _make_db(tmp_path / "a.db",
                 {"SYN.NS": N_DAILY, "WARM.NS": 120, "SHORT.NS": 10, "EMPTY.NS": 0,
                  "OLD.NS": 50},
                 intraday=("SYN.NS",), inactive=("OLD.NS",))
    yield s
    s.close()


def _same(a: sc.TechnicalScore, b: sc.TechnicalScore) -> None:
    ra, rb = a.to_row(), b.to_row()
    for key in ("cached", "staleness_days", "is_stale"):     # relative to the reference time
        ra.pop(key), rb.pop(key)
    for k in ra:
        if isinstance(ra[k], float) and isinstance(rb[k], float) and math.isnan(ra[k]):
            assert math.isnan(rb[k]), k
        elif isinstance(ra[k], float):
            assert ra[k] == pytest.approx(rb[k], abs=1e-12), k
        else:
            assert ra[k] == rb[k], k
    assert a.reasons == b.reasons


# ──────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────

def test_daily_score_fields(db):
    s = sc.score("SYN.NS", session=db, as_of=date(2026, 10, 15))
    assert s.status == "ok" and s.warmup_satisfied and not s.cached
    assert -1.0 <= s.tech_score <= 1.0
    assert s.as_of == pd.Timestamp("2026-10-15")
    assert s.bars_available == sc.DAILY_LOOKBACK_BARS
    assert set(s.family_scores) == set(sc.FAMILIES)
    assert math.isnan(s.family_scores["mtf"])            # daily-only frame
    assert s.regime in ("trend", "range", "chaotic") and 0 <= s.regime_confidence <= 1
    assert s.reasons and all(isinstance(r, str) for r in s.reasons)
    assert s.staleness_days == 0 and not s.is_stale


def test_daily_score_matches_pure_compute(db):
    """The DB path is just the compute modules on the last DAILY_LOOKBACK_BARS bars."""
    s = sc.score("SYN.NS", session=db, persist=False, use_cache=False)
    days = pd.bdate_range(end="2026-10-15", periods=N_DAILY)
    d = _ohlcv(N_DAILY, seed=1)
    df = pd.DataFrame(d, index=days).astype(float).iloc[-sc.DAILY_LOOKBACK_BARS:]
    frame = sc.compute_daily_frame(df)
    assert s.tech_score == pytest.approx(frame["TECH_SCORE"].iloc[-1], abs=1e-12)


def test_unsupported_timeframe_raises(db):
    with pytest.raises(ValueError):
        sc.score("SYN.NS", "1w", session=db)


# ──────────────────────────────────────────────────────────
# No lookahead: as_of == a DB truncated at as_of
# ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("cut", ["2026-03-02", "2026-09-30"])
def test_as_of_equals_truncated_db_daily(db, tmp_path, cut):
    t = date.fromisoformat(cut)
    full = sc.score("SYN.NS", as_of=t, session=db, persist=False, use_cache=False)

    trunc = _make_db(tmp_path / "trunc.db", {"SYN.NS": N_DAILY})
    trunc.execute(delete(PriceDaily).where(PriceDaily.date > t))
    trunc.commit()
    cut_score = sc.score("SYN.NS", session=trunc, persist=False, use_cache=False)
    trunc.close()

    assert full.as_of == pd.Timestamp(t)
    _same(full, cut_score)


def test_as_of_datetime_hides_todays_daily_bar_until_16h(db):
    before = sc.score("SYN.NS", as_of=datetime(2026, 10, 15, 15, 59), session=db, persist=False)
    after = sc.score("SYN.NS", as_of=datetime(2026, 10, 15, 16, 0), session=db, persist=False)
    assert before.as_of == pd.Timestamp("2026-10-14")
    assert after.as_of == pd.Timestamp("2026-10-15")


def test_as_of_equals_truncated_db_intraday(db, tmp_path):
    t = datetime(2026, 10, 13, 10, 59)         # the 10:30 bar has closed, the 10:45 bar has not
    full = sc.score("SYN.NS", "15m", as_of=t, session=db, persist=False, use_cache=False)
    assert full.as_of == pd.Timestamp("2026-10-13 10:30")

    trunc = _make_db(tmp_path / "trunc.db", {"SYN.NS": N_DAILY}, intraday=("SYN.NS",))
    trunc.execute(delete(PriceIntraday).where(PriceIntraday.timestamp > datetime(2026, 10, 13, 10, 30)))
    trunc.execute(delete(PriceDaily).where(PriceDaily.date >= date(2026, 10, 13)))
    trunc.commit()
    cut_score = sc.score("SYN.NS", "15m", session=trunc, persist=False, use_cache=False)
    trunc.close()

    assert full.status == "ok"
    assert not math.isnan(full.family_scores["mtf"])
    _same(full, cut_score)


# ──────────────────────────────────────────────────────────
# Cache and persistence
# ──────────────────────────────────────────────────────────

def test_cache_hit_skips_recompute(db, monkeypatch):
    first = sc.score("SYN.NS", session=db)
    assert not first.cached
    assert db.execute(select(func.count()).select_from(TechnicalScoreRecord)).scalar() == 1

    def boom(*a, **k):
        raise AssertionError("recomputed on a cache hit")
    monkeypatch.setattr(sc, "compute_daily_frame", boom)
    second = sc.score("SYN.NS", session=db)
    assert second.cached
    _same(first, second)
    assert db.execute(select(func.count()).select_from(TechnicalScoreRecord)).scalar() == 1


def test_revised_bar_invalidates_cache(db):
    first = sc.score("SYN.NS", session=db)
    asset_id = db.execute(select(Asset.id).where(Asset.ticker == "SYN.NS")).scalar()
    db.execute(update(PriceDaily).where(PriceDaily.asset_id == asset_id,
                                        PriceDaily.date == date(2026, 10, 13))
               .values(close=PriceDaily.close * 1.05))
    db.commit()
    second = sc.score("SYN.NS", session=db)
    assert not second.cached
    assert second.as_of == first.as_of
    assert db.execute(select(func.count()).select_from(TechnicalScoreRecord)).scalar() == 1


def test_scorer_version_bump_invalidates_cache(db, monkeypatch):
    sc.score("SYN.NS", session=db)
    monkeypatch.setattr(sc, "SCORER_VERSION", "test-bump")
    assert not sc.score("SYN.NS", session=db).cached


def test_cache_hit_refreshes_staleness(db):
    sc.score("SYN.NS", session=db, as_of=date(2026, 10, 15))
    later = sc.score("SYN.NS", session=db, as_of=date(2026, 10, 30))
    assert later.cached and later.as_of == pd.Timestamp("2026-10-15")
    assert later.staleness_days == 11 and later.is_stale


def test_persisted_row_contents(db):
    s = sc.score("SYN.NS", session=db)
    rec = db.execute(select(TechnicalScoreRecord)).scalars().one()
    assert rec.timeframe == "1d" and rec.timestamp == datetime(2026, 10, 15)
    assert rec.tech_score == pytest.approx(s.tech_score)
    assert rec.score_mtf is None and rec.status == "ok" and rec.scorer_version == sc.SCORER_VERSION


def test_ensure_schema_creates_missing_table(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}", future=True)
    Base.metadata.create_all(engine, tables=[t for t in Base.metadata.sorted_tables
                                             if t.name != "technical_scores"])
    s = sessionmaker(bind=engine)()
    sc.ensure_schema(s)
    assert engine.dialect.has_table(engine.connect(), "technical_scores")
    s.close()


# ──────────────────────────────────────────────────────────
# Flagged results, never exceptions
# ──────────────────────────────────────────────────────────

def test_unknown_ticker_is_flagged(db):
    s = sc.score("NOPE.NS", session=db)
    assert s.status == "unknown_asset" and math.isnan(s.tech_score) and s.as_of is None


def test_no_bars_is_flagged(db):
    s = sc.score("EMPTY.NS", session=db)
    assert s.status == "no_data" and s.bars_available == 0 and math.isnan(s.tech_score)


def test_as_of_before_history_is_flagged(db):
    s = sc.score("SYN.NS", as_of=date(2000, 1, 1), session=db)
    assert s.status == "no_data"


def test_too_few_bars_is_flagged(db):
    s = sc.score("SHORT.NS", session=db)
    assert s.status == "insufficient_data" and s.bars_available == 10
    assert not s.warmup_satisfied and math.isnan(s.tech_score)


def test_warmup_is_flagged(db):
    s = sc.score("WARM.NS", session=db)
    assert s.status == "warmup" and s.bars_available == 120
    assert not s.warmup_satisfied and math.isnan(s.tech_score)


def test_no_intraday_bars_is_flagged(db):
    assert sc.score("WARM.NS", "15m", session=db).status == "no_data"


def test_compute_error_is_flagged_and_not_cached(db, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("bad bars")
    monkeypatch.setattr(sc, "compute_daily_frame", boom)
    s = sc.score("SYN.NS", session=db)
    assert s.status == "error" and "bad bars" in s.message
    assert db.execute(select(func.count()).select_from(TechnicalScoreRecord)).scalar() == 0


# ──────────────────────────────────────────────────────────
# Universe
# ──────────────────────────────────────────────────────────

def test_score_universe_ranks_active_assets(db):
    df = sc.score_universe("stock", session=db)
    assert set(df["ticker"]) == {"SYN.NS", "WARM.NS", "SHORT.NS", "EMPTY.NS"}   # OLD.NS inactive
    assert df.iloc[0]["ticker"] == "SYN.NS" and df.iloc[0]["rank"] == 1
    assert (df.iloc[1:]["rank"] == 0).all() and df.iloc[1:]["tech_score"].isna().all()
    statuses = dict(zip(df["ticker"], df["status"]))
    assert statuses == {"SYN.NS": "ok", "WARM.NS": "warmup",
                        "SHORT.NS": "insufficient_data", "EMPTY.NS": "no_data"}
    again = sc.score_universe("stock", session=db)
    assert again.set_index("ticker").loc[["SYN.NS", "WARM.NS", "SHORT.NS"], "cached"].all()
