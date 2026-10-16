"""
TA-Lib core indicator wrapper.

Computes ~90 TA-Lib indicators and returns them as a dict of {column_name:
1D numpy array or pandas Series}. The public orchestrator (indicators.py)
assembles these into the final DataFrame via a single pd.concat — this
avoids pandas' O(N²) DataFrame fragmentation penalty from repeated
single-column assignment.

Column naming matches pandas-ta conventions where possible so downstream
code that referenced those names continues to work unchanged.

TA-Lib requires float64 numpy arrays. We cast once at the top of
compute_talib_columns() and reuse. Volume comes from the DB as int64,
so it needs explicit float casting — TA-Lib crashes with obscure
errors on int arrays.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd
import talib

logger = logging.getLogger(__name__)

# Moving average periods: Fibonacci-flavoured for EMAs, round for SMAs.
EMA_PERIODS: tuple[int, ...] = (5, 8, 13, 20, 50, 100, 200)
SMA_PERIODS: tuple[int, ...] = (5, 10, 20, 50, 100, 200)


# ──────────────────────────────────────────────────────────
# Safe dispatch helper
# ──────────────────────────────────────────────────────────

def _safe(
    out: dict[str, Any],
    col_names: str | tuple[str, ...],
    func: Callable,
    *args,
    **kwargs,
) -> None:
    """Call a TA-Lib function and stash its output(s) into the output dict.

    Parameters
    ----------
    out : dict
        Accumulator — populated in-place.
    col_names : str | tuple[str, ...]
        Single name for single-output functions, tuple for multi-output.
    func : callable
        TA-Lib function (e.g. talib.RSI).

    Failures are tolerated — the column simply won't appear in the output
    dict, so it'll be absent downstream. Logged at DEBUG level to avoid
    log spam during development.
    """
    try:
        result = func(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 — intentionally broad
        logger.debug("talib.%s(%s) failed: %s", func.__name__, col_names, exc)
        return

    if isinstance(col_names, str):
        out[col_names] = result
    else:
        for name, arr in zip(col_names, result):
            out[name] = arr


# ──────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────

def compute_talib_columns(df: pd.DataFrame) -> dict[str, Any]:
    """Compute all TA-Lib indicators, return as a dict of column arrays.

    Parameters
    ----------
    df : DataFrame
        OHLCV DataFrame — must already be validated by the caller.

    Returns
    -------
    dict
        {column_name: 1D array or Series} sized to match len(df).
        Caller (indicators.py) assembles into the final DataFrame.
    """
    O = df["open"].to_numpy(dtype=np.float64)
    H = df["high"].to_numpy(dtype=np.float64)
    L = df["low"].to_numpy(dtype=np.float64)
    C = df["close"].to_numpy(dtype=np.float64)
    V = df["volume"].to_numpy(dtype=np.float64)

    out: dict[str, Any] = {}
    _momentum(out, O, H, L, C, V)
    _overlap(out, O, H, L, C, V, df)     # df passed for BB derivation access
    _trend(out, O, H, L, C, V)
    _volatility(out, H, L, C)
    _volume(out, H, L, C, V)
    _cycle(out, C)
    _price_transform(out, O, H, L, C)
    _statistic(out, C)
    return out


# ──────────────────────────────────────────────────────────
# Momentum (~30 indicators)
# ──────────────────────────────────────────────────────────

def _momentum(out, O, H, L, C, V):
    _safe(out, "ADX_14",        talib.ADX,       H, L, C, timeperiod=14)
    _safe(out, "ADXR_14",       talib.ADXR,      H, L, C, timeperiod=14)
    _safe(out, "APO_12_26",     talib.APO,       C, fastperiod=12, slowperiod=26, matype=0)

    # Aroon: returns (down, up). pandas-ta naming: AROOND_14, AROONU_14
    _safe(out, ("AROOND_14", "AROONU_14"), talib.AROON, H, L, timeperiod=14)
    _safe(out, "AROONOSC_14",   talib.AROONOSC,  H, L, timeperiod=14)

    _safe(out, "BOP",           talib.BOP,       O, H, L, C)
    _safe(out, "CCI_20",        talib.CCI,       H, L, C, timeperiod=20)
    _safe(out, "CMO_14",        talib.CMO,       C, timeperiod=14)
    _safe(out, "DX_14",         talib.DX,        H, L, C, timeperiod=14)

    # MACD: returns (macd, signal, hist). pandas-ta naming.
    _safe(out, ("MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9"),
          talib.MACD, C, fastperiod=12, slowperiod=26, signalperiod=9)

    _safe(out, "MFI_14",        talib.MFI,       H, L, C, V, timeperiod=14)
    _safe(out, "MINUS_DI_14",   talib.MINUS_DI,  H, L, C, timeperiod=14)
    _safe(out, "PLUS_DI_14",    talib.PLUS_DI,   H, L, C, timeperiod=14)
    _safe(out, "MINUS_DM_14",   talib.MINUS_DM,  H, L, timeperiod=14)
    _safe(out, "PLUS_DM_14",    talib.PLUS_DM,   H, L, timeperiod=14)
    _safe(out, "MOM_10",        talib.MOM,       C, timeperiod=10)
    _safe(out, "PPO_12_26",     talib.PPO,       C, fastperiod=12, slowperiod=26, matype=0)
    _safe(out, "ROC_10",        talib.ROC,       C, timeperiod=10)
    _safe(out, "ROCP_10",       talib.ROCP,      C, timeperiod=10)
    _safe(out, "ROCR_10",       talib.ROCR,      C, timeperiod=10)
    _safe(out, "ROCR100_10",    talib.ROCR100,   C, timeperiod=10)
    _safe(out, "RSI_14",        talib.RSI,       C, timeperiod=14)

    # Stochastic: returns (slowk, slowd). pandas-ta uses STOCHk/STOCHd.
    _safe(out, ("STOCHk_14_3_3", "STOCHd_14_3_3"),
          talib.STOCH, H, L, C,
          fastk_period=14, slowk_period=3, slowk_matype=0,
          slowd_period=3, slowd_matype=0)

    # Stochastic Fast: returns (fastk, fastd).
    _safe(out, ("STOCHFk_14_3", "STOCHFd_14_3"),
          talib.STOCHF, H, L, C, fastk_period=14, fastd_period=3, fastd_matype=0)

    # Stochastic RSI: returns (fastk, fastd).
    _safe(out, ("STOCHRSIk_14", "STOCHRSId_14"),
          talib.STOCHRSI, C, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)

    _safe(out, "TRIX_30",       talib.TRIX,      C, timeperiod=30)
    _safe(out, "ULTOSC_7_14_28", talib.ULTOSC,   H, L, C,
          timeperiod1=7, timeperiod2=14, timeperiod3=28)
    _safe(out, "WILLR_14",      talib.WILLR,     H, L, C, timeperiod=14)


# ──────────────────────────────────────────────────────────
# Overlap / Moving Averages (~25 indicators)
# ──────────────────────────────────────────────────────────

def _overlap(out, O, H, L, C, V, df):
    # Bollinger Bands: returns (upper, middle, lower).
    _safe(out, ("BBU_20_2.0", "BBM_20_2.0", "BBL_20_2.0"),
          talib.BBANDS, C, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    # Derived BB metrics — %B (position within band) and Bandwidth
    # (normalized width). Canonical pandas-ta outputs.
    if all(k in out for k in ("BBU_20_2.0", "BBM_20_2.0", "BBL_20_2.0")):
        bbu = out["BBU_20_2.0"]
        bbm = out["BBM_20_2.0"]
        bbl = out["BBL_20_2.0"]
        width = bbu - bbl
        # Use np.where to avoid /0 warnings; NaN where width is 0
        safe_width = np.where(width == 0, np.nan, width)
        safe_mid   = np.where(bbm == 0, np.nan, bbm)
        out["BBP_20_2.0"] = (C - bbl) / safe_width
        out["BBB_20_2.0"] = width / safe_mid

    # Standard moving averages at multiple periods.
    for p in SMA_PERIODS:
        _safe(out, f"SMA_{p}",  talib.SMA,  C, timeperiod=p)
    for p in EMA_PERIODS:
        _safe(out, f"EMA_{p}",  talib.EMA,  C, timeperiod=p)

    # Other moving averages (single canonical period each).
    _safe(out, "DEMA_30",       talib.DEMA,      C, timeperiod=30)
    _safe(out, "TEMA_30",       talib.TEMA,      C, timeperiod=30)
    _safe(out, "TRIMA_30",      talib.TRIMA,     C, timeperiod=30)
    _safe(out, "WMA_30",        talib.WMA,       C, timeperiod=30)
    _safe(out, "KAMA_30",       talib.KAMA,      C, timeperiod=30)
    _safe(out, "T3_5",          talib.T3,        C, timeperiod=5, vfactor=0.7)

    # Hilbert Transform Trendline — adaptive trendline, no period.
    _safe(out, "HT_TRENDLINE",  talib.HT_TRENDLINE, C)

    # MAMA / FAMA: MESA Adaptive MA (returns both).
    _safe(out, ("MAMA", "FAMA"), talib.MAMA, C, fastlimit=0.5, slowlimit=0.05)

    # MIDPOINT / MIDPRICE.
    _safe(out, "MIDPOINT_14",   talib.MIDPOINT,  C, timeperiod=14)
    _safe(out, "MIDPRICE_14",   talib.MIDPRICE,  H, L, timeperiod=14)


# ──────────────────────────────────────────────────────────
# Trend (Parabolic SAR family)
# ──────────────────────────────────────────────────────────

def _trend(out, O, H, L, C, V):
    _safe(out, "PSAR_0.02_0.2",  talib.SAR,     H, L, acceleration=0.02, maximum=0.2)
    _safe(out, "PSAREXT",        talib.SAREXT,  H, L)


# ──────────────────────────────────────────────────────────
# Volatility
# ──────────────────────────────────────────────────────────

def _volatility(out, H, L, C):
    _safe(out, "ATR_14",    talib.ATR,    H, L, C, timeperiod=14)
    _safe(out, "NATR_14",   talib.NATR,   H, L, C, timeperiod=14)  # Normalized ATR (%)
    _safe(out, "TRANGE",    talib.TRANGE, H, L, C)                 # True Range (single-bar)


# ──────────────────────────────────────────────────────────
# Volume
# ──────────────────────────────────────────────────────────

def _volume(out, H, L, C, V):
    _safe(out, "AD",             talib.AD,     H, L, C, V)          # Chaikin A/D Line
    _safe(out, "ADOSC_3_10",     talib.ADOSC,  H, L, C, V,
          fastperiod=3, slowperiod=10)                              # Chaikin Oscillator
    _safe(out, "OBV",            talib.OBV,    C, V)


# ──────────────────────────────────────────────────────────
# Cycle (Hilbert Transform family)
# ──────────────────────────────────────────────────────────

def _cycle(out, C):
    _safe(out, "HT_DCPERIOD",    talib.HT_DCPERIOD,  C)    # Dominant Cycle Period
    _safe(out, "HT_DCPHASE",     talib.HT_DCPHASE,   C)    # Dominant Cycle Phase
    _safe(out, ("HT_PHASOR_I", "HT_PHASOR_Q"),
          talib.HT_PHASOR,       C)                        # In-phase / Quadrature
    _safe(out, ("HT_SINE", "HT_LEADSINE"),
          talib.HT_SINE,         C)                        # Sine Wave (lead/lag)
    _safe(out, "HT_TRENDMODE",   talib.HT_TRENDMODE, C)    # 0 = cycle, 1 = trend


# ──────────────────────────────────────────────────────────
# Price Transform (4 single-bar features)
# ──────────────────────────────────────────────────────────

def _price_transform(out, O, H, L, C):
    _safe(out, "AVGPRICE",   talib.AVGPRICE,  O, H, L, C)
    _safe(out, "MEDPRICE",   talib.MEDPRICE,  H, L)
    _safe(out, "TYPPRICE",   talib.TYPPRICE,  H, L, C)
    _safe(out, "WCLPRICE",   talib.WCLPRICE,  H, L, C)


# ──────────────────────────────────────────────────────────
# Statistic Functions
# ──────────────────────────────────────────────────────────

def _statistic(out, C):
    _safe(out, "LINEARREG_14",         talib.LINEARREG,           C, timeperiod=14)
    _safe(out, "LINEARREG_ANGLE_14",   talib.LINEARREG_ANGLE,     C, timeperiod=14)
    _safe(out, "LINEARREG_INTERCEPT_14", talib.LINEARREG_INTERCEPT, C, timeperiod=14)
    _safe(out, "LINEARREG_SLOPE_14",   talib.LINEARREG_SLOPE,     C, timeperiod=14)
    _safe(out, "STDDEV_20",            talib.STDDEV,              C, timeperiod=20, nbdev=1)
    _safe(out, "VAR_20",               talib.VAR,                 C, timeperiod=20, nbdev=1)
    _safe(out, "TSF_14",               talib.TSF,                 C, timeperiod=14)
