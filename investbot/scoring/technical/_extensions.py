"""
Custom indicator extensions beyond TA-Lib's coverage.

Each of these is either (a) modern enough that TA-Lib's frozen indicator
set doesn't include it, or (b) a derived metric we want computed once at
this layer rather than in every downstream consumer.

Indicators here:
    Ichimoku Cloud       — Tenkan, Kijun, Senkou A, Senkou B
                           (Chikou INTENTIONALLY OMITTED — lookahead bias)
    SuperTrend           — ATR-based trailing-stop overlay
    Keltner Channel      — EMA middle ± ATR bands (needed for Squeeze)
    Squeeze              — BB-inside-KC flag + continuous intensity
    Heikin-Ashi          — Smoothed candles (separate from raw OHLC)
    Vortex Indicator     — VI+ / VI- directional movement
    Relative Volume      — Current volume vs rolling average
    Returns              — 1-period log and percent returns, cumulative

All functions take a DataFrame and return a dict of {column_name: array
or Series}. The orchestrator (indicators.py) assembles everything via a
single pd.concat at the end — this avoids pandas' DataFrame fragmentation
penalty from repeated single-column assignment.

Squeeze() reads BBU/BBL from the partial frame (populated by TA-Lib
core), and KCU/KCL from the keltner output it receives. compute_extension_columns()
handles that wiring.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import talib

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────

def compute_extension_columns(df: pd.DataFrame) -> dict[str, Any]:
    """Compute all extensions against a partial frame containing OHLCV + TA-Lib columns.

    The partial frame includes everything TA-Lib computed, so squeeze()
    can read BBU/BBL directly. Returns a single merged dict — the
    orchestrator concats this into the final frame in one shot.
    """
    out: dict[str, Any] = {}

    _merge(out, _safely(ichimoku,    df))
    _merge(out, _safely(supertrend,  df))
    _merge(out, _safely(keltner,     df))   # must run before squeeze

    # squeeze() needs KCU/KCL; pass a view that includes what we just built.
    # Building a temporary frame of just the KC columns is cheap.
    kc_only = pd.DataFrame({k: v for k, v in out.items() if k.startswith("KC")},
                           index=df.index)
    partial_for_squeeze = pd.concat([df, kc_only], axis=1, copy=False)
    _merge(out, _safely(squeeze,     partial_for_squeeze))

    _merge(out, _safely(heikin_ashi, df))
    _merge(out, _safely(vortex,      df))
    _merge(out, _safely(rvol,        df))
    _merge(out, _safely(returns,     df))

    return out


def _safely(fn, df: pd.DataFrame) -> dict[str, Any]:
    """Run an extension; return its dict, or empty dict on failure."""
    try:
        return fn(df)
    except Exception as exc:  # noqa: BLE001
        logger.debug("extension %s failed: %s", fn.__name__, exc)
        return {}


def _merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Merge source dict into target. No-op if source is empty."""
    target.update(source)


# ──────────────────────────────────────────────────────────
# Ichimoku Cloud (no-lookahead variant)
# ──────────────────────────────────────────────────────────

def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26,
             senkou: int = 52) -> dict[str, Any]:
    """Ichimoku Kinkō Hyō, safe for backtesting.

    Components computed:
        Tenkan-sen (Conversion Line)  — (Nh + Nl) / 2 over `tenkan` bars
        Kijun-sen  (Base Line)        — (Nh + Nl) / 2 over `kijun` bars
        Senkou Span A (Leading Span A) — (Tenkan + Kijun) / 2, displayed
            `kijun` bars in the future. Implemented as .shift(kijun), which
            at time t gives the value computed at t-kijun. NOT lookahead.
        Senkou Span B (Leading Span B) — (Nh + Nl) / 2 over `senkou` bars,
            displayed `kijun` bars in the future. Same no-lookahead shift.

    Chikou Span (Lagging Span) is INTENTIONALLY OMITTED. It is defined as
    close.shift(-kijun) — literally the close price 26 bars in the future.
    Including it in a feature matrix used for backtesting or ML produces
    spectacular in-sample results that are entirely artifacts of future
    data leakage. This omission is the whole point of using our own
    implementation instead of pandas-ta's default.
    """
    high, low = df["high"], df["low"]

    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen  = (high.rolling(kijun).max()  + low.rolling(kijun).min())  / 2
    span_a     = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    span_b     = ((high.rolling(senkou).max() + low.rolling(senkou).min()) / 2).shift(kijun)

    return {
        f"ICHI_TENKAN_{tenkan}": tenkan_sen,
        f"ICHI_KIJUN_{kijun}":   kijun_sen,
        f"ICHI_SPAN_A_{kijun}":  span_a,
        f"ICHI_SPAN_B_{kijun}":  span_b,
    }


# ──────────────────────────────────────────────────────────
# SuperTrend (ATR-based trailing band)
# ──────────────────────────────────────────────────────────

def supertrend(df: pd.DataFrame, length: int = 10,
               multiplier: float = 3.0) -> dict[str, Any]:
    """SuperTrend: ATR-based trailing band with directional bias.

    Inherently iterative — each bar's band and direction depend on the
    previous bar's values. Implemented as a numpy loop; ~1ms for 5000 bars,
    not worth numba-izing until it matters.

    Outputs:
        SUPERT_10_3.0   — the trailing band value at each bar
        SUPERTd_10_3.0  — direction: +1 (uptrend) or -1 (downtrend)
    """
    H = df["high"].to_numpy(dtype=np.float64)
    L = df["low"].to_numpy(dtype=np.float64)
    C = df["close"].to_numpy(dtype=np.float64)
    n = len(df)

    hl2 = (H + L) / 2.0
    atr = talib.ATR(H, L, C, timeperiod=length)

    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    trend       = np.full(n, np.nan)
    direction   = np.zeros(n, dtype=np.int64)

    for i in range(n):
        if np.isnan(basic_upper[i]):
            continue
        if i == 0 or np.isnan(final_upper[i - 1]):
            final_upper[i] = basic_upper[i]
            final_lower[i] = basic_lower[i]
            direction[i]   = 1
        else:
            # "Final" bands: tighten only in the favorable direction
            final_upper[i] = (
                basic_upper[i]
                if basic_upper[i] < final_upper[i - 1] or C[i - 1] > final_upper[i - 1]
                else final_upper[i - 1]
            )
            final_lower[i] = (
                basic_lower[i]
                if basic_lower[i] > final_lower[i - 1] or C[i - 1] < final_lower[i - 1]
                else final_lower[i - 1]
            )
            if C[i] > final_upper[i - 1]:
                direction[i] = 1
            elif C[i] < final_lower[i - 1]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]

        trend[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    tag = f"{length}_{multiplier}"
    return {
        f"SUPERT_{tag}":  trend,
        f"SUPERTd_{tag}": direction,
    }


# ──────────────────────────────────────────────────────────
# Keltner Channel (EMA ± ATR × multiplier)
# ──────────────────────────────────────────────────────────

def keltner(df: pd.DataFrame, length: int = 20, atr_length: int = 10,
            multiplier: float = 2.0) -> dict[str, Any]:
    """Keltner Channel. Required upstream of squeeze().

    Middle line is an EMA of close; bands are ATR-based. Standard Chester
    Keltner / Linda Raschke formulation.
    """
    H = df["high"].to_numpy(dtype=np.float64)
    L = df["low"].to_numpy(dtype=np.float64)
    C = df["close"].to_numpy(dtype=np.float64)

    ema = talib.EMA(C, timeperiod=length)
    atr = talib.ATR(H, L, C, timeperiod=atr_length)

    tag = f"{length}_{multiplier}"
    return {
        f"KCM_{tag}": ema,
        f"KCU_{tag}": ema + multiplier * atr,
        f"KCL_{tag}": ema - multiplier * atr,
    }


# ──────────────────────────────────────────────────────────
# Squeeze — flag + continuous intensity
# ──────────────────────────────────────────────────────────

def squeeze(df: pd.DataFrame) -> dict[str, Any]:
    """BB-inside-KC squeeze detector.

    Reads BBU_20_2.0 / BBL_20_2.0 (from TA-Lib core) and KCU_20_2.0 /
    KCL_20_2.0 (from keltner() above). Returns empty dict if either set
    is missing.

    Outputs:
        SQUEEZE_ON         — 1 when Bollinger Bands are fully inside the
                             Keltner Channel (compression regime), else 0.
                             Int64 dtype, supports <NA>.
        SQUEEZE_INTENSITY  — Continuous compression metric, normalized by
                             KC width:
                                (kc_width - bb_width) / kc_width × 100
                             Positive → BB narrower than KC (squeezed).
                             100     → BB fully collapsed to middle.
                             Negative → BB wider than KC (expanding).
                             Zero    → BB walls touching KC walls.

    Downstream (confluence.py) will likely also care about:
        - SQUEEZE_INTENSITY.diff()   — rate of compression (tightening?)
        - SQUEEZE_ON transitions     — release = potential breakout
    """
    required = ("BBU_20_2.0", "BBL_20_2.0", "KCU_20_2.0", "KCL_20_2.0")
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.debug("squeeze skipped — missing columns: %s", missing)
        return {}

    bbu = df["BBU_20_2.0"]
    bbl = df["BBL_20_2.0"]
    kcu = df["KCU_20_2.0"]
    kcl = df["KCL_20_2.0"]

    bb_width = bbu - bbl
    kc_width = kcu - kcl

    squeeze_on = ((bbu < kcu) & (bbl > kcl)).astype("Int64")
    intensity = (kc_width - bb_width) / kc_width.replace(0, np.nan) * 100.0

    return {
        "SQUEEZE_ON":        squeeze_on,
        "SQUEEZE_INTENSITY": intensity,
    }


# ──────────────────────────────────────────────────────────
# Heikin-Ashi (smoothed candles)
# ──────────────────────────────────────────────────────────

def heikin_ashi(df: pd.DataFrame) -> dict[str, Any]:
    """Heikin-Ashi transformed OHLC.

        HA_Close = (O + H + L + C) / 4
        HA_Open[0] = (O[0] + C[0]) / 2
        HA_Open[i] = (HA_Open[i-1] + HA_Close[i-1]) / 2
        HA_High  = max(H, HA_Open, HA_Close)
        HA_Low   = min(L, HA_Open, HA_Close)

    The HA_Open recurrence is a linear dependency, implemented as a
    numpy loop. For a 5000-bar series this is <1 ms.

    NOTE: Heikin-Ashi is NOT the same as the raw OHLC — downstream
    code that wants actual market prices should continue using O/H/L/C,
    not HA_*. HA is a visualization/smoothing aid.
    """
    O = df["open"].to_numpy(dtype=np.float64)
    H = df["high"].to_numpy(dtype=np.float64)
    L = df["low"].to_numpy(dtype=np.float64)
    C = df["close"].to_numpy(dtype=np.float64)
    n = len(df)

    ha_close = (O + H + L + C) / 4.0
    ha_open  = np.empty(n, dtype=np.float64)
    ha_open[0] = (O[0] + C[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    ha_high = np.maximum.reduce([H, ha_open, ha_close])
    ha_low  = np.minimum.reduce([L, ha_open, ha_close])

    return {
        "HA_OPEN":  ha_open,
        "HA_HIGH":  ha_high,
        "HA_LOW":   ha_low,
        "HA_CLOSE": ha_close,
    }


# ──────────────────────────────────────────────────────────
# Vortex Indicator (VI+ / VI-)
# ──────────────────────────────────────────────────────────

def vortex(df: pd.DataFrame, length: int = 14) -> dict[str, Any]:
    """Vortex Indicator — directional trend strength.

        VM+ = |High_t - Low_{t-1}|
        VM- = |Low_t  - High_{t-1}|
        VI+ = Σ VM+ / Σ TR   (over `length` bars)
        VI- = Σ VM- / Σ TR

    Crossovers of VI+ and VI- signal trend changes. TA-Lib lacks this
    (despite having nearly everything else) — it's a Botes/Siepman 2010
    indicator, post-dating TA-Lib's frozen list.
    """
    H = df["high"]
    L = df["low"]
    C = df["close"]

    tr_arr = talib.TRANGE(H.to_numpy(dtype=np.float64),
                          L.to_numpy(dtype=np.float64),
                          C.to_numpy(dtype=np.float64))
    tr = pd.Series(tr_arr, index=df.index)

    vm_plus  = (H - L.shift(1)).abs()
    vm_minus = (L - H.shift(1)).abs()

    sum_tr = tr.rolling(length).sum()
    return {
        f"VTXP_{length}": vm_plus.rolling(length).sum()  / sum_tr.replace(0, np.nan),
        f"VTXM_{length}": vm_minus.rolling(length).sum() / sum_tr.replace(0, np.nan),
    }


# ──────────────────────────────────────────────────────────
# Relative Volume
# ──────────────────────────────────────────────────────────

def rvol(df: pd.DataFrame, periods: tuple[int, ...] = (20, 50)) -> dict[str, Any]:
    """Relative Volume: current bar's volume as a ratio to rolling average.

        RVOL_N = Volume_t / mean(Volume, N bars)

    Values > 1 indicate above-average activity. RVOL ≥ 2 is commonly
    used as an unusual-activity threshold (breakouts, news events,
    institutional flow). ML studies consistently rank volume-based
    features as high-importance predictors, so this is worth including
    even though pandas-ta doesn't expose it.
    """
    V = df["volume"]
    return {
        f"RVOL_{p}": V / V.rolling(p).mean().replace(0, np.nan)
        for p in periods
    }


# ──────────────────────────────────────────────────────────
# Returns (log and percent, single-period and cumulative)
# ──────────────────────────────────────────────────────────

def returns(df: pd.DataFrame) -> dict[str, Any]:
    """Log and percent returns — single-period and cumulative.

    Why include these if they're trivial? Two reasons:
      1. ML literature (XGBoost / random forest studies) consistently
         finds raw return-based features outperforming many technical
         indicators in feature importance. Including them as first-class
         columns ensures they're available to the PCA+XGBoost pipeline.
      2. Performance dashboards need cumulative return as a baseline.
    """
    close = df["close"]
    pct   = close.pct_change()
    logr  = np.log(close / close.shift(1))

    return {
        "LOGRET_1":    logr,
        "PCTRET_1":    pct,
        "CUMLOGRET_1": logr.cumsum(),
        "CUMPCTRET_1": (1.0 + pct).cumprod() - 1.0,
    }
