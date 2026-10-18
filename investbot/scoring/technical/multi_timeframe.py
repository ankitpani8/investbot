"""
Multi-timeframe (15m / 1h / daily) features without lookahead.

Pure compute module: takes DataFrames and returns DataFrames or dicts of
float64 arrays. No DB I/O. ``scorer.py`` composes the technical modules.

Timestamp conventions
    The DB stores 15m bars as naive IST timestamps labelled by the bar's
    **open** time (09:15 .. 15:15, 25 bars per full NSE session). Every frame
    this module returns is labelled by the bar's **close** time instead, so
    "the bar labelled t" always means "the bar whose data is complete at t".
    Daily bars come indexed by date and are relabelled by the time their
    values become available (``DAILY_AVAILABLE_AT``, 16:00 IST, the same
    cutoff ingestion uses before storing today's daily bar).

Resampling (``resample_ohlcv``)
    Bins are anchored at the session open each day, so 1h bars cover
    09:15-10:15, 10:15-11:15, ..., 14:15-15:15 and a partial 15:15-15:30.
    A bin is labelled by its scheduled close, clipped to the session close:
    10:15, 11:15, ..., 15:15, 15:30. The label never depends on which 15m
    bars happen to be present, so a bin that is still forming (only some of
    its 15m bars exist yet) carries a label in the future of the last 15m
    bar and stays invisible to it. ``resample_ohlcv(df, "15min")`` just
    relabels 15m bars by close time and is how the base frame is built.

Alignment (``align_higher_timeframe``)
    A base bar labelled t gets the newest HTF bar whose label is <= t. Both
    labels are close times, so the base bar sees only HTF bars that had
    closed by t. A 1h bar closing at 10:15 is visible to the 15m bar closing
    at 10:15 (they end on the same tick and share the same final close), not
    to the 15m bar closing at 10:00. Daily bars are labelled 16:00 on their
    date, after every intraday bar of that date, so every intraday bar on
    day D sees day D-1's (previous session's) daily values, never day D's.

MTF alignment (``mtf_alignment``)
    Per timeframe (15m, 1h, 1d):
        trend direction    = mean(sign(EMA_20 - EMA_50), SUPERTd_10_3.0),
                             one of -1, -0.5, 0, 0.5, 1
        momentum regime    = +1 if RSI_14 > RSI_BULL, -1 if < RSI_BEAR, else 0
    MTF_TREND_ALIGN    = TIMEFRAME_WEIGHTS-weighted mean of the trend
                         directions, in [-1, 1]. +1 means every timeframe is
                         in an uptrend by both measures, -1 every one in a
                         downtrend, values near 0 mean disagreement or flat.
    MTF_MOMENTUM_ALIGN = the same weighted mean over the momentum regimes.
    MTF_REGIME_1D      = REGIME (0 range / 1 trend / 2 chaotic) from
                         ``regime.py`` on daily bars, previous session.
    MTF_REGIME_CONFIDENCE_1D = REGIME_CONFIDENCE alongside it (the smoothed
                         daily value, not REGIME_CONFIDENCE_RAW).
    The per-timeframe components are emitted as well (MTF_TREND_15M, ...),
    so ``scorer.py`` can explain a score. Regime is only taken from daily
    bars: its thresholds are calibrated for daily data (see BACKLOG).

Warmup
    An aligned value is NaN until the HTF bar it comes from exists and is
    past its own warmup. MTF_* are NaN until every timeframe they combine is
    available (no partial averages). Nothing is dropped.
"""
from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd
import talib

from investbot.scoring.technical._extensions import supertrend
from investbot.scoring.technical.regime import compute_regime

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

#: NSE cash session, IST.
SESSION_OPEN: time = time(9, 15)
SESSION_CLOSE: time = time(15, 30)
#: Bar length of the stored intraday data.
BASE_BAR: pd.Timedelta = pd.Timedelta("15min")
#: Time on date D at which the daily bar of D counts as known. Ingestion never
#: stores today's daily bar before 16:00 IST, and it is after every intraday
#: bar of D, so intraday bars on D see D-1's daily values.
DAILY_AVAILABLE_AT: time = time(16, 0)

H1_PREFIX: str = "H1_"
D1_PREFIX: str = "D1_"

# Features computed on every timeframe
EMA_FAST: int = 20
EMA_SLOW: int = 50
RSI_PERIOD: int = 14
SUPERTREND_LENGTH: int = 10
SUPERTREND_MULT: float = 3.0
#: RSI regime bands: above RSI_BULL is bullish momentum, below RSI_BEAR
#: bearish, in between neutral.
RSI_BULL: float = 55.0
RSI_BEAR: float = 45.0
#: Weights of 15m, 1h and 1d in the MTF_*_ALIGN averages.
TIMEFRAME_WEIGHTS: tuple[float, float, float] = (1.0, 1.0, 1.0)

_ST_TAG = f"{SUPERTREND_LENGTH}_{SUPERTREND_MULT}"
FEATURE_COLUMNS: tuple[str, ...] = (
    f"EMA_{EMA_FAST}", f"EMA_{EMA_SLOW}", f"SUPERTd_{_ST_TAG}", f"RSI_{RSI_PERIOD}",
)
_TIMEFRAMES: tuple[str, ...] = ("15M", "1H", "1D")


# ──────────────────────────────────────────────────────────
# Resampling and labelling
# ──────────────────────────────────────────────────────────

def _check_ohlcv(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"expected a DatetimeIndex, got {type(df.index).__name__}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("index must be monotonically increasing")
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")


def _offset(t: time) -> pd.Timedelta:
    return pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)


def resample_ohlcv(
    df_15m: pd.DataFrame,
    rule: str | pd.Timedelta = "1h",
    *,
    input_label: str = "open",
    session_open: time = SESSION_OPEN,
    session_close: time = SESSION_CLOSE,
    base_bar: pd.Timedelta = BASE_BAR,
) -> pd.DataFrame:
    """
    Aggregate session bars into ``rule`` bars labelled by close time.

    Parameters
    ----------
    df_15m : DataFrame
        OHLCV bars of length ``base_bar``, oldest first. Not mutated.
    rule : str or Timedelta
        Target bar length, a multiple of ``base_bar`` (e.g. "15min", "1h").
    input_label : {"open", "close"}
        Whether ``df_15m`` is labelled by bar open (the DB convention) or close.
    session_open, session_close : time
        Session bounds. Bins start at ``session_open`` each day and the last
        bin ends at ``session_close`` (it may be shorter than ``rule``).

    Returns
    -------
    DataFrame
        open/high/low/close/volume plus BAR_COUNT (number of base bars in the
        bin), indexed by the bin's close time. Bins with no base bars are
        absent.
    """
    _check_ohlcv(df_15m)
    rule = pd.Timedelta(rule)
    if rule <= pd.Timedelta(0) or rule % base_bar != pd.Timedelta(0):
        raise ValueError(f"rule {rule} must be a positive multiple of {base_bar}")
    if input_label not in ("open", "close"):
        raise ValueError("input_label must be 'open' or 'close'")

    bar_open = df_15m.index if input_label == "open" else df_15m.index - base_bar
    day = bar_open.normalize()
    since_open = (bar_open - day) - _offset(session_open)
    close_off = _offset(session_close) - _offset(session_open)
    outside = (since_open < pd.Timedelta(0)) | (since_open + base_bar > close_off)
    if outside.any():
        raise ValueError(f"{int(outside.sum())} bars fall outside the session, "
                         f"first at {bar_open[outside][0]}")

    k = np.asarray(since_open // rule, dtype=np.int64)
    bin_close_ns = np.minimum((k + 1) * rule.value, close_off.value)
    label = day + _offset(session_open) + pd.to_timedelta(bin_close_ns, unit="ns")

    grouped = df_15m[list(OHLCV_COLUMNS)].groupby(np.asarray(label), sort=True)
    out = grouped.agg({"open": "first", "high": "max", "low": "min",
                       "close": "last", "volume": "sum"})
    out["BAR_COUNT"] = grouped.size().astype(np.int64)
    out.index = pd.DatetimeIndex(out.index, name=df_15m.index.name)
    if df_15m.index.tz is not None and out.index.tz is None:
        out.index = out.index.tz_localize(df_15m.index.tz)
    return out


def label_daily_bars(
    daily_df: pd.DataFrame,
    available_at: time = DAILY_AVAILABLE_AT,
) -> pd.DataFrame:
    """Return a copy of ``daily_df`` relabelled date + ``available_at``."""
    if not isinstance(daily_df.index, pd.DatetimeIndex):
        raise TypeError(f"expected a DatetimeIndex, got {type(daily_df.index).__name__}")
    out = daily_df.copy()
    out.index = daily_df.index.normalize() + _offset(available_at)
    return out


def align_higher_timeframe(
    base_df: pd.DataFrame,
    htf_df: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """
    Attach HTF columns (renamed ``prefix + name``) to every base bar.

    Both frames must be labelled by close (or availability) time. Base bar t
    gets the newest HTF row with label <= t, i.e. the last HTF bar that had
    closed at t; NaN before the first one. Returns a new DataFrame: base_df
    plus the prefixed columns, built with one concat.
    """
    for name, frame in (("base_df", base_df), ("htf_df", htf_df)):
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError(f"{name} needs a DatetimeIndex")
        if not frame.index.is_monotonic_increasing:
            raise ValueError(f"{name} index must be monotonically increasing")

    # Compare in one explicit unit: pandas 3 indexes can carry different resolutions
    # (us vs ns), and raw .asi8 from mixed units silently mis-aligns (= lookahead).
    pos = np.searchsorted(
        htf_df.index.as_unit("ns").asi8, base_df.index.as_unit("ns").asi8, side="right"
    ) - 1
    seen = pos >= 0
    take = np.where(seen, pos, 0)
    cols: dict[str, np.ndarray] = {}
    for col in htf_df.columns:
        # leading NaN row so an empty htf_df still indexes safely
        src = np.r_[np.nan, htf_df[col].to_numpy(dtype=np.float64)]
        cols[f"{prefix}{col}"] = np.where(seen, src[take + 1], np.nan)
    return pd.concat([base_df, pd.DataFrame(cols, index=base_df.index)], axis=1)


# ──────────────────────────────────────────────────────────
# Per-timeframe features and alignment scores
# ──────────────────────────────────────────────────────────

def timeframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """EMA_20, EMA_50, SUPERTd_10_3.0 and RSI_14 on one timeframe (float64, warmup NaN)."""
    c = df["close"].to_numpy(dtype=np.float64)
    st = supertrend(df, length=SUPERTREND_LENGTH, multiplier=SUPERTREND_MULT)
    st_dir = st[f"SUPERTd_{_ST_TAG}"].astype(np.float64)
    st_dir[np.isnan(st[f"SUPERT_{_ST_TAG}"])] = np.nan   # 0 during warmup → NaN
    cols = {
        f"EMA_{EMA_FAST}": talib.EMA(c, timeperiod=EMA_FAST),
        f"EMA_{EMA_SLOW}": talib.EMA(c, timeperiod=EMA_SLOW),
        f"SUPERTd_{_ST_TAG}": st_dir,
        f"RSI_{RSI_PERIOD}": talib.RSI(c, timeperiod=RSI_PERIOD),
    }
    return pd.DataFrame(cols, index=df.index)


def trend_direction(ema_fast: np.ndarray, ema_slow: np.ndarray, st_dir: np.ndarray) -> np.ndarray:
    """mean(sign(EMA fast - slow), SuperTrend direction); NaN if either is missing."""
    return 0.5 * (np.sign(ema_fast - ema_slow) + st_dir)


def momentum_regime(rsi: np.ndarray) -> np.ndarray:
    """+1 above RSI_BULL, -1 below RSI_BEAR, 0 between; NaN where RSI is NaN."""
    out = np.where(rsi > RSI_BULL, 1.0, np.where(rsi < RSI_BEAR, -1.0, 0.0))
    out[np.isnan(rsi)] = np.nan
    return out


def _weighted_mean(parts: list[np.ndarray], weights: tuple[float, ...]) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    return np.column_stack(parts) @ (w / w.sum())   # NaN in any part → NaN


def list_mtf_columns() -> list[str]:
    """All column names produced by mtf_alignment, in output order."""
    return (
        ["MTF_TREND_ALIGN", "MTF_MOMENTUM_ALIGN", "MTF_REGIME_1D", "MTF_REGIME_CONFIDENCE_1D"]
        + [f"MTF_TREND_{tf}" for tf in _TIMEFRAMES]
        + [f"MTF_MOMENTUM_{tf}" for tf in _TIMEFRAMES]
    )


def mtf_alignment(
    df_15m: pd.DataFrame,
    df_daily: pd.DataFrame,
    *,
    input_label: str = "open",
    htf_rule: str = "1h",
    daily_available_at: time = DAILY_AVAILABLE_AT,
    timeframe_weights: tuple[float, float, float] = TIMEFRAME_WEIGHTS,
    regime_kwargs: dict | None = None,
) -> pd.DataFrame:
    """
    15m / 1h / 1d trend and momentum agreement and the daily regime.

    Parameters
    ----------
    df_15m : DataFrame
        15m OHLCV for one asset, labelled as ``input_label`` says (the DB
        stores bar open times). Not mutated.
    df_daily : DataFrame
        Daily OHLCV for the same asset, indexed by date. Rows for dates after
        the last 15m bar are harmless: they are never visible.
    htf_rule : str
        Bar length of the middle timeframe.
    regime_kwargs : dict, optional
        Passed to ``regime.compute_regime`` on the daily bars.

    Returns
    -------
    DataFrame
        Indexed by 15m close time, columns ``list_mtf_columns()``, float64.
    """
    _check_ohlcv(df_daily)
    base = resample_ohlcv(df_15m, BASE_BAR, input_label=input_label)
    h1 = resample_ohlcv(df_15m, htf_rule, input_label=input_label)

    daily_feat = timeframe_features(df_daily)
    reg = compute_regime(df_daily, **(regime_kwargs or {}))
    daily_feat = pd.concat(
        [daily_feat, pd.DataFrame({"REGIME": reg["REGIME"],
                                   "REGIME_CONFIDENCE": reg["REGIME_CONFIDENCE"]},
                                  index=df_daily.index)], axis=1)
    daily_feat = label_daily_bars(daily_feat, daily_available_at)
    if base.index.tz is not None and daily_feat.index.tz is None:
        daily_feat.index = daily_feat.index.tz_localize(base.index.tz)

    frame = align_higher_timeframe(timeframe_features(base), timeframe_features(h1), H1_PREFIX)
    frame = align_higher_timeframe(frame, daily_feat, D1_PREFIX)

    ema_f, ema_s = f"EMA_{EMA_FAST}", f"EMA_{EMA_SLOW}"
    st, rsi = f"SUPERTd_{_ST_TAG}", f"RSI_{RSI_PERIOD}"
    out: dict[str, np.ndarray] = {}
    trends, moms = [], []
    for tf, p in zip(_TIMEFRAMES, ("", H1_PREFIX, D1_PREFIX)):
        trends.append(trend_direction(frame[p + ema_f].to_numpy(), frame[p + ema_s].to_numpy(),
                                      frame[p + st].to_numpy()))
        moms.append(momentum_regime(frame[p + rsi].to_numpy()))
    out["MTF_TREND_ALIGN"] = _weighted_mean(trends, timeframe_weights)
    out["MTF_MOMENTUM_ALIGN"] = _weighted_mean(moms, timeframe_weights)
    out["MTF_REGIME_1D"] = frame[D1_PREFIX + "REGIME"].to_numpy(dtype=np.float64)
    out["MTF_REGIME_CONFIDENCE_1D"] = frame[D1_PREFIX + "REGIME_CONFIDENCE"].to_numpy(
        dtype=np.float64)
    for tf, arr in zip(_TIMEFRAMES, trends):
        out[f"MTF_TREND_{tf}"] = arr
    for tf, arr in zip(_TIMEFRAMES, moms):
        out[f"MTF_MOMENTUM_{tf}"] = arr
    return pd.DataFrame(out, index=base.index)
