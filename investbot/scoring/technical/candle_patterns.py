"""
Candlestick pattern recognition: every TA-Lib ``CDL*`` function.

Pure compute module: takes an OHLCV DataFrame and returns pattern columns.
No DB I/O. Not wired into ``indicators.add_all_indicators``; ``scorer.py``
composes the technical modules, so the indicator column contract is untouched.

Output columns (all float64, NaN during warmup)
    CDL_<NAME>       raw TA-Lib value, one of {-200, -100, -80, 0, 80, 100, 200}.
                     ±100 is a plain signal and ±200 a confirmed one (TA-Lib
                     emits ±200 for confirmation bars, e.g. HIKKAKE).
                     ±80 is TA-Lib's weaker variant of ENGULFING, HARAMI and
                     HARAMICROSS (e.g. engulfing that only ties an open or close).
    CDLN_<NAME>      sign of the raw value: -1 bearish, 0 none, +1 bullish.
    CDL_BULL_COUNT   number of patterns with a bullish signal on the bar.
    CDL_BEAR_COUNT   number of patterns with a bearish signal on the bar.
    CDL_NET_SCORE    sum over patterns of sign × weight, where weight is
                     1.0 for confirmed (±200) and 0.5 for plain (±100 and
                     the weaker ±80) signals, divided by ``NET_SCORE_SCALE`` and clipped
                     to [-1, 1].

<NAME> is the TA-Lib function name without its ``CDL`` prefix, so
``talib.CDLENGULFING`` gives ``CDL_ENGULFING`` and ``CDLN_ENGULFING``.

No lookahead
    TA-Lib patterns only read the current bar and bars before it. A pattern
    is reported on the bar that completes it, never on an earlier bar of the
    formation. Body and shadow thresholds ("long body", "doji" and so on)
    are trailing averages over a fixed number of past bars (TA-Lib candle
    settings, at most 10 bars), so the value at bar t does not depend on
    where the series starts or ends once the warmup is over. This is tested
    by truncation invariance in ``tests/test_candle_patterns.py``.

Lookback (bars before t needed for a value at t; from TA-Lib's own lookback)
    14  BREAKAWAY, LADDERBOTTOM, MATHOLD, RISEFALL3METHODS
    13  3BLACKCROWS, CONCEALBABYSWALL
    12  2CROWS, 3INSIDE, 3STARSINSOUTH, 3WHITESOLDIERS, ABANDONEDBABY,
        ADVANCEBLOCK, and most other three-bar patterns
    Single-bar patterns need about 10 (the candle-settings average period),
    and the shortest need 2. The first ``lookback`` rows of each column are
    NaN rather than TA-Lib's 0, because "not enough history" is not the same
    as "no pattern". The aggregates are NaN until every pattern is out of
    warmup (``max_lookback()`` rows), so they always cover the same set.

NaN input
    Incomplete OHLC rows are rejected at ingestion (``investbot/data/quality.py``).
    If one reaches this layer anyway, the bar itself is NaN in every column,
    but TA-Lib's running averages carry the NaN forward, so later bars read
    as 0 ("no pattern") for every pattern that uses them. Same failure mode
    as the stateful indicators in ``indicators.py``.

Volume is not used: TA-Lib patterns are price-only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import talib
from talib import abstract

PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")

#: Every TA-Lib pattern function, in TA-Lib's own order (61 with TA-Lib 0.8.1).
PATTERN_FUNCTIONS: tuple[str, ...] = tuple(talib.get_function_groups()["Pattern Recognition"])

#: Weight of a confirmed (±200) signal in CDL_NET_SCORE.
CONFIRMED_WEIGHT: float = 1.0
#: Weight of a plain (±100, and TA-Lib's weaker ±80) signal in CDL_NET_SCORE.
PLAIN_WEIGHT: float = 0.5
#: Divisor for the weighted sum before clipping to [-1, 1]. With 3.0, six
#: agreeing plain signals (or three confirmed ones) saturate the score and one
#: lone plain pattern gives ±0.17. Measured on 20 NIFTY50 daily histories
#: (~24k bars, 2026-10-17): median 1 pattern per bar, 90th percentile 5,
#: max 11; |CDL_NET_SCORE| median 0.17, 90th percentile 0.67, and 2.2% of
#: bars clip at ±1. Many patterns overlap (DOJI, LONGLEGGEDDOJI, RICKSHAWMAN,
#: HIGHWAVE, SPINNINGTOP fire together), which confluence.py should handle.
NET_SCORE_SCALE: float = 3.0


def _suffix(func_name: str) -> str:
    return func_name[len("CDL"):]


def pattern_lookbacks() -> dict[str, int]:
    """TA-Lib lookback (bars before t needed for a value at t) for each pattern."""
    return {name: abstract.Function(name).lookback for name in PATTERN_FUNCTIONS}


def max_lookback() -> int:
    """Rows at the start of the series for which the aggregates are NaN."""
    return max(pattern_lookbacks().values())


def list_candle_columns() -> list[str]:
    """All column names produced by compute_candle_patterns, in output order."""
    raw = [f"CDL_{_suffix(n)}" for n in PATTERN_FUNCTIONS]
    norm = [f"CDLN_{_suffix(n)}" for n in PATTERN_FUNCTIONS]
    return raw + norm + ["CDL_BULL_COUNT", "CDL_BEAR_COUNT", "CDL_NET_SCORE"]


def compute_candle_patterns(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Run every TA-Lib CDL* function on an OHLCV DataFrame.

    Parameters
    ----------
    df : DataFrame
        Lowercase 'open', 'high', 'low', 'close' columns, oldest bar first.
        Not mutated.

    Returns
    -------
    dict[str, np.ndarray]
        float64 arrays of len(df), keyed as in ``list_candle_columns()``.
    """
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing OHLC columns: {missing}")

    o, h, l, c = (df[col].to_numpy(dtype=np.float64, copy=True) for col in PRICE_COLUMNS)
    n = len(df)
    bad_bar = np.isnan(o) | np.isnan(h) | np.isnan(l) | np.isnan(c)
    lookbacks = pattern_lookbacks()

    raw_cols: dict[str, np.ndarray] = {}
    norm_cols: dict[str, np.ndarray] = {}
    bull = np.zeros(n, dtype=np.float64)
    bear = np.zeros(n, dtype=np.float64)
    weighted = np.zeros(n, dtype=np.float64)

    for name in PATTERN_FUNCTIONS:
        raw = getattr(talib, name)(o, h, l, c).astype(np.float64)
        sign = np.sign(raw)
        weight = np.where(np.abs(raw) >= 200, CONFIRMED_WEIGHT, PLAIN_WEIGHT)
        bull += sign > 0
        bear += sign < 0
        weighted += sign * weight

        raw[: lookbacks[name]] = np.nan
        raw[bad_bar] = np.nan
        suffix = _suffix(name)
        raw_cols[f"CDL_{suffix}"] = raw
        norm_cols[f"CDLN_{suffix}"] = np.sign(raw)

    net = np.clip(weighted / NET_SCORE_SCALE, -1.0, 1.0)
    agg_invalid = bad_bar.copy()
    agg_invalid[: max_lookback()] = True
    for arr in (bull, bear, net):
        arr[agg_invalid] = np.nan

    return {
        **raw_cols,
        **norm_cols,
        "CDL_BULL_COUNT": bull,
        "CDL_BEAR_COUNT": bear,
        "CDL_NET_SCORE": net,
    }


def add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame: ``df`` plus every candle pattern column (one pd.concat)."""
    cols = compute_candle_patterns(df)
    return pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
