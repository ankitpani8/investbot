"""
Confluence: group-normalised, regime-weighted, rule-based technical score.

Pure compute module: takes one DataFrame holding OHLCV plus whatever upstream
technical columns are available (``indicators.py``, ``candle_patterns.py``,
``support_resistance.py``, ``regime.py``, ``multi_timeframe.py``, all on the
same bar index) and returns a dict of arrays. No DB I/O, no ``dropna()``,
input not mutated. ``scorer.py`` composes the upstream modules.

Pipeline
    1. Signals. Every rule in ``SIGNALS`` turns one indicator (or a small,
       documented combination) into a directional signal in [-1, +1]:
       +1 bullish, -1 bearish, 0 neutral, NaN when an input is NaN (warmup).
       A signal whose input columns are absent from the frame is skipped.
    2. Families. Signals are averaged **within** their family first
       (trend, momentum, volatility, volume, pattern, sr, mtf). This is the
       group normalisation: ten correlated moving-average signals count as
       one trend opinion, not ten votes. ``TECH_SCORE_<FAMILY>`` is that mean.
       It is NaN on a bar where any of the family's signals is NaN (no
       partial averages during warmup), and NaN throughout when none of the
       family's inputs are in the frame (e.g. MTF on daily-only scoring).
    3. Weights. Family weights come from ``investbot/config/scoring_weights.py``:
       ``w = c * FAMILY_WEIGHTS[REGIME] + (1 - c) * FAMILY_WEIGHTS["neutral"]``
       with c = REGIME_CONFIDENCE, the EMA-smoothed one (neutral weights while the regime is NaN),
       renormalised over the families whose inputs are in the frame.
    4. Score. ``TECH_SCORE = multiplier * sum_f w_f * TECH_SCORE_<f>``, where
       the multiplier is the same blend of ``REGIME_SCORE_MULTIPLIER`` (<= 1).
       Every term is in [-1, 1] and the weights sum to 1, so the score is too.
       NaN until every family in the frame has a value (daily: ~200 bars,
       set by EMA_200).
    5. Reasons. Each signal's contribution is ``multiplier * w_f * s / n_f``
       (n_f = signals in its family), so contributions add up to
       TECH_SCORE exactly. ``TECH_REASONS`` lists the ``REASONS_TOP_N``
       largest by absolute value, e.g. ``"price vs EMA 200: bullish (+0.08)"``.

Signal rules (constants below; ATR = ATR_14)
    trend
        EMA_<n>_POS      tanh((close - EMA_n) / (EMA_POS_ATR_SCALE * ATR)), n = 20, 50, 200
        EMA_20_50_CROSS  tanh((EMA_20 - EMA_50) / (EMA_POS_ATR_SCALE * ATR)), same for 50/200
        SUPERTREND       SUPERTd_10_3.0 (+1/-1); NaN while SUPERT_10_3.0 is NaN
        PSAR             sign(close - PSAR_0.02_0.2)
        DI_SPREAD        (PLUS_DI_14 - MINUS_DI_14) / (PLUS_DI_14 + MINUS_DI_14)
        AROON            AROONOSC_14 / 100
        ICHI_CLOUD       +1 above both spans, -1 below both, 0 inside the cloud
        ICHI_TK          sign(ICHI_TENKAN_9 - ICHI_KIJUN_26)
        LINREG_SLOPE     tanh(LINEARREG_SLOPE_14 / (LINREG_SLOPE_ATR_SCALE * ATR))
        VORTEX           clip((VTXP_14 - VTXM_14) / VORTEX_SCALE, -1, 1)
        FIB_SWING        FIB_SWING_DIR, direction of the last confirmed swing
    momentum
        RSI              oscillator rule on RSI_14, bands 30/50/70 on [0, 100]
        STOCH            oscillator rule on STOCHk_14_3_3, bands 20/50/80
        WILLR            oscillator rule on WILLR_14 + 100, bands 20/50/80
        CCI              oscillator rule on CCI_20, bands -100/0/100, extremes ±250
        ULTOSC           oscillator rule on ULTOSC_7_14_28, bands 30/50/70
        MACD_HIST        0.5 * sign(MACDh) + 0.5 * sign(MACDh[t] - MACDh[t-1])
        ROC              tanh(ROC_10 / (NATR_14 * sqrt(10))): the 10-bar move in
                         units of its expected size under ATR-sized daily moves
    volatility
        BB_POS           oscillator rule on BBP_20_2.0, bands 0/0.5/1, extremes -0.5/1.5
        KC_POS           oscillator rule on (close - KCM) / (KCU - KCM), bands ±1, extremes ±2
    volume
        OBV_FLOW         clip(FLOW_GAIN * (OBV[t] - OBV[t-k]) / sum(volume[t-k+1..t]))
        AD_FLOW          the same on the Chaikin A/D line (CMF-like), k = FLOW_WINDOW
        MFI              oscillator rule on MFI_14, bands 20/50/80
        RVOL_CONFIRM     sign(LOGRET_1) * clip(RVOL_20 - 1, 0, 1): a heavy-volume bar
                         confirms its own direction
    pattern
        CANDLES          sum over CDL_* of sign * weight (``candle_patterns``
                         CONFIRMED_WEIGHT / PLAIN_WEIGHT) / NET_SCORE_SCALE, clipped,
                         **excluding** ``INDECISION_PATTERNS``. TA-Lib emits those
                         as +100 regardless of direction (DOJI and friends) or signs
                         them by candle colour (HIGHWAVE, SPINNINGTOP), and they fire
                         together on one candle shape, so they would add a fake
                         bullish bias and overcount one candle.
    sr
        SR_POSITION      (d_res - d_sup) / (d_res + d_sup) with d = SR_DIST_*_PCT:
                         +1 sitting on support, -1 at resistance. No resistance
                         above (support only) = +1, no support below = -1.
        VP_POS           VP_POS (position against the 70% value area, clipped ±1)
    mtf
        MTF_TREND        MTF_TREND_ALIGN
        MTF_MOMENTUM     MTF_MOMENTUM_ALIGN

Oscillator rule
    Piecewise linear through (lo, -1), (mid, 0), (hi, +1): above the midline is
    bullish momentum. Beyond a band the signal fades linearly towards
    ``±OSC_EXTREME_VALUE`` at the extreme (floor/ceiling): overbought is
    exhaustion, so a pinned oscillator stops adding conviction. With the
    default 0.0, RSI 85 reads +0.5 and RSI 100 reads 0. See
    docs/DECISIONS_PENDING.md for the alternatives.

No lookahead
    Every rule is a function of the same bar's columns, plus the previous bar
    (MACD slope) or the previous ``FLOW_WINDOW`` bars (flows). Upstream columns
    are lookahead-free, so the score is too; tested by truncation invariance in
    ``tests/test_confluence.py``.

Column contract
    ``COLUMN_FAMILIES`` maps every column the upstream modules produce to a
    family or to ``None`` (not scored: context, price levels, redundant
    duplicates, regime inputs). A test fails when an upstream module adds a
    column that is not listed here, so a new indicator has to be placed
    deliberately.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from investbot.config.scoring_weights import (
    FAMILIES,
    FAMILY_WEIGHTS,
    REGIME_NAMES,
    REGIME_SCORE_MULTIPLIER,
)
from investbot.scoring.technical import candle_patterns as cp

# ──────────────────────────────────────────────────────────
# Signal rule parameters (rules, not weights: weights live in config)
# ──────────────────────────────────────────────────────────

#: Distance (in ATRs) at which price-vs-average reads tanh(1) = 0.76.
EMA_POS_ATR_SCALE: float = 2.0
#: LINEARREG_SLOPE_14 of this many ATRs per bar reads tanh(1).
LINREG_SLOPE_ATR_SCALE: float = 0.1
#: VTXP - VTXM spread that saturates the vortex signal.
VORTEX_SCALE: float = 0.3
#: Oscillator signal at the extreme of its range (0 = fades out, 1 = no fade,
#: -1 = full mean reversion at the extreme).
OSC_EXTREME_VALUE: float = 0.0
#: Bars over which OBV / A/D flow is measured.
FLOW_WINDOW: int = 10
#: Gain on the volume flow fraction before clipping to [-1, 1].
FLOW_GAIN: float = 1.5
#: Number of entries in TECH_REASONS.
REASONS_TOP_N: int = 3

#: Patterns with no directional meaning (see module docstring).
INDECISION_PATTERNS: frozenset[str] = frozenset({
    "CDLDOJI", "CDLDRAGONFLYDOJI", "CDLGRAVESTONEDOJI", "CDLLONGLEGGEDDOJI",
    "CDLRICKSHAWMAN", "CDLHIGHWAVE", "CDLSPINNINGTOP",
})
DIRECTIONAL_PATTERNS: tuple[str, ...] = tuple(
    n for n in cp.PATTERN_FUNCTIONS if n not in INDECISION_PATTERNS
)

TREND, MOMENTUM, VOLATILITY, VOLUME, PATTERN, SR, MTF = FAMILIES

# ──────────────────────────────────────────────────────────
# Column → family contract
# ──────────────────────────────────────────────────────────

#: Every upstream column → family, or None when it is not scored.
COLUMN_FAMILIES: dict[str, str | None] = {
    # indicators.py — momentum
    "ADX_14": None,             # strength, not direction; feeds regime.py
    "ADXR_14": None,
    "APO_12_26": None,          # duplicate of the MACD line
    "AROOND_14": None,          # folded into AROONOSC_14
    "AROONU_14": None,
    "AROONOSC_14": TREND,
    "BOP": None,                # single-bar, noisy
    "CCI_20": MOMENTUM,
    "CMO_14": None,             # CMO = 2 * RSI - 100, duplicate of RSI_14
    "DX_14": None,
    "MACD_12_26_9": None,       # MACDh carries the signal
    "MACDs_12_26_9": None,
    "MACDh_12_26_9": MOMENTUM,
    "MFI_14": VOLUME,
    "MINUS_DI_14": TREND,
    "PLUS_DI_14": TREND,
    "MINUS_DM_14": None,        # raw inputs of the DIs
    "PLUS_DM_14": None,
    "MOM_10": None,             # unnormalised ROC_10
    "PPO_12_26": None,          # duplicate of MACD
    "ROC_10": MOMENTUM,
    "ROCP_10": None,            # ROC_10 / 100
    "ROCR_10": None,
    "ROCR100_10": None,
    "RSI_14": MOMENTUM,
    "STOCHk_14_3_3": MOMENTUM,
    "STOCHd_14_3_3": None,      # smoothed STOCHk
    "STOCHFk_14_3": None,       # unsmoothed STOCHk, noisy
    "STOCHFd_14_3": None,
    "STOCHRSIk_14": None,       # oscillator of RSI_14, noisy duplicate
    "STOCHRSId_14": None,
    "TRIX_30": None,
    "ULTOSC_7_14_28": MOMENTUM,
    "WILLR_14": MOMENTUM,
    # indicators.py — overlap / moving averages
    "BBU_20_2.0": None,
    "BBM_20_2.0": None,
    "BBL_20_2.0": None,
    "BBP_20_2.0": VOLATILITY,
    "BBB_20_2.0": None,         # bandwidth: non-directional
    "SMA_5": None,              # SMAs duplicate the EMAs
    "SMA_10": None,
    "SMA_20": None,
    "SMA_50": None,
    "SMA_100": None,
    "SMA_200": None,
    "EMA_5": None,
    "EMA_8": None,
    "EMA_13": None,
    "EMA_20": TREND,
    "EMA_50": TREND,
    "EMA_100": None,
    "EMA_200": TREND,
    "DEMA_30": None,
    "TEMA_30": None,
    "TRIMA_30": None,
    "WMA_30": None,
    "KAMA_30": None,
    "T3_5": None,
    "HT_TRENDLINE": None,
    "MAMA": None,
    "FAMA": None,
    "MIDPOINT_14": None,
    "MIDPRICE_14": None,
    "PSAR_0.02_0.2": TREND,
    "PSAREXT": None,            # duplicate of PSAR
    # volatility
    "ATR_14": None,             # normaliser for the trend signals
    "NATR_14": None,            # normaliser for ROC
    "TRANGE": None,
    # volume
    "AD": VOLUME,
    "ADOSC_3_10": None,         # derivative of AD, covered by AD_FLOW
    "OBV": VOLUME,
    # cycle (Hilbert): phase information, not direction
    "HT_DCPERIOD": None,
    "HT_DCPHASE": None,
    "HT_PHASOR_I": None,
    "HT_PHASOR_Q": None,
    "HT_SINE": None,
    "HT_LEADSINE": None,
    "HT_TRENDMODE": None,
    # price transforms: price levels
    "AVGPRICE": None,
    "MEDPRICE": None,
    "TYPPRICE": None,
    "WCLPRICE": None,
    # statistics
    "LINEARREG_14": None,
    "LINEARREG_ANGLE_14": None,  # scale-dependent angle of the slope
    "LINEARREG_INTERCEPT_14": None,
    "LINEARREG_SLOPE_14": TREND,
    "STDDEV_20": None,
    "VAR_20": None,
    "TSF_14": None,
    # _extensions.py
    "ICHI_TENKAN_9": TREND,
    "ICHI_KIJUN_26": TREND,
    "ICHI_SPAN_A_26": TREND,
    "ICHI_SPAN_B_26": TREND,
    "SUPERT_10_3.0": TREND,
    "SUPERTd_10_3.0": TREND,
    "KCM_20_2.0": VOLATILITY,
    "KCU_20_2.0": VOLATILITY,
    "KCL_20_2.0": None,
    "SQUEEZE_ON": None,         # compression flag, no direction
    "SQUEEZE_INTENSITY": None,
    "HA_OPEN": None,            # smoothed candles duplicate the trend family
    "HA_HIGH": None,
    "HA_LOW": None,
    "HA_CLOSE": None,
    "VTXP_14": TREND,
    "VTXM_14": TREND,
    "RVOL_20": VOLUME,
    "RVOL_50": None,
    "LOGRET_1": VOLUME,         # direction of the RVOL_CONFIRM bar
    "PCTRET_1": None,
    "CUMLOGRET_1": None,
    "CUMPCTRET_1": None,
    # candle_patterns.py: raw CDL_* feed CANDLES (indecision patterns are
    # mapped but excluded inside the rule); CDLN_* and the aggregates are
    # duplicates of the raw values.
    **{f"CDL_{n[3:]}": PATTERN for n in cp.PATTERN_FUNCTIONS},
    **{f"CDLN_{n[3:]}": None for n in cp.PATTERN_FUNCTIONS},
    "CDL_BULL_COUNT": None,
    "CDL_BEAR_COUNT": None,
    "CDL_NET_SCORE": None,      # recomputed without indecision patterns
    # support_resistance.py
    "PIVOT_HIGH": None,
    "PIVOT_LOW": None,
    "SR_SUPPORT_1": None,       # levels: summarised by the distances
    "SR_SUPPORT_2": None,
    "SR_SUPPORT_3": None,
    "SR_RESIST_1": None,
    "SR_RESIST_2": None,
    "SR_RESIST_3": None,
    "SR_DIST_SUPPORT_PCT": SR,
    "SR_DIST_RESIST_PCT": SR,
    "SR_STRENGTH_SUPPORT": None,
    "SR_STRENGTH_RESIST": None,
    "VP_POC": None,
    "VP_VAH": None,
    "VP_VAL": None,
    "VP_POS": SR,
    "FIB_236": None,
    "FIB_382": None,
    "FIB_500": None,
    "FIB_618": None,
    "FIB_786": None,
    "FIB_NEAREST_DIST_PCT": None,
    "FIB_SWING_HIGH": None,
    "FIB_SWING_LOW": None,
    "FIB_SWING_DIR": TREND,     # swing structure is trend evidence
    # regime.py: sets the family weights, never scored itself
    "HURST_100": None,
    "HURST_250": None,
    "LYAP_100": None,
    "LYAP_250": None,
    "REALVOL_20": None,
    "REALVOL_250": None,
    "VOL_RATIO": None,
    "REGIME_SCORE_RANGE": None,
    "REGIME_SCORE_TREND": None,
    "REGIME_SCORE_CHAOS": None,
    "REGIME": None,
    "REGIME_CONFIDENCE_RAW": None,   # unsmoothed gap; weights use the smoothed one
    "REGIME_CONFIDENCE": None,
    # multi_timeframe.py
    "MTF_TREND_ALIGN": MTF,
    "MTF_MOMENTUM_ALIGN": MTF,
    "MTF_REGIME_1D": None,      # daily regime context for intraday frames
    "MTF_REGIME_CONFIDENCE_1D": None,
    "MTF_TREND_15M": None,      # components of the align columns
    "MTF_TREND_1H": None,
    "MTF_TREND_1D": None,
    "MTF_MOMENTUM_15M": None,
    "MTF_MOMENTUM_1H": None,
    "MTF_MOMENTUM_1D": None,
}

# ──────────────────────────────────────────────────────────
# Signal rules
# ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Signal:
    """One rule: ``fn(*arrays of inputs)`` → directional signal in [-1, 1]."""

    name: str
    family: str
    label: str
    inputs: tuple[str, ...]
    fn: Callable[..., np.ndarray]


def _lag(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    out[k:] = x[:-k]
    return out


def _nan_where(out: np.ndarray, *inputs: np.ndarray) -> np.ndarray:
    bad = np.zeros(out.shape, dtype=bool)
    for x in inputs:
        bad |= np.isnan(x)
    out = out.astype(np.float64)
    out[bad] = np.nan
    return out


def _sign(x: np.ndarray) -> np.ndarray:
    return np.sign(x)        # NaN stays NaN


def oscillator(
    x: np.ndarray, floor: float, lo: float, mid: float, hi: float, ceil: float,
    extreme: float | None = None,
) -> np.ndarray:
    """Piecewise oscillator rule (see module docstring). NaN in → NaN out."""
    e = OSC_EXTREME_VALUE if extreme is None else extreme
    out = np.interp(x, [floor, lo, mid, hi, ceil], [-e, -1.0, 0.0, 1.0, e])
    return _nan_where(out, x)


def _ema_pos(close, ema, atr):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.tanh((close - ema) / (EMA_POS_ATR_SCALE * atr))


def _di_spread(plus, minus):
    total = plus + minus
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, (plus - minus) / total, np.where(np.isnan(total), np.nan, 0.0))


def _cloud(close, span_a, span_b):
    top, bottom = np.maximum(span_a, span_b), np.minimum(span_a, span_b)
    out = np.where(close > top, 1.0, np.where(close < bottom, -1.0, 0.0))
    return _nan_where(out, close, span_a, span_b)


def _supertrend(direction, band):
    return _nan_where(direction.astype(np.float64), band)


def _macd_hist(hist):
    return 0.5 * np.sign(hist) + 0.5 * np.sign(hist - _lag(hist, 1))


def _roc(roc, natr):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.tanh(roc / (natr * np.sqrt(10.0)))


def _kc_pos(close, mid, upper):
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = (close - mid) / (upper - mid)
    return oscillator(rel, -2.0, -1.0, 0.0, 1.0, 2.0)


def _flow(line, volume):
    vol_sum = pd.Series(volume).rolling(FLOW_WINDOW, min_periods=FLOW_WINDOW).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = (line - _lag(line, FLOW_WINDOW)) / vol_sum
    frac = np.where(vol_sum > 0, frac, np.where(np.isnan(vol_sum), np.nan, 0.0))
    return np.clip(FLOW_GAIN * frac, -1.0, 1.0)


def _rvol_confirm(rvol, logret):
    return np.sign(logret) * np.clip(rvol - 1.0, 0.0, 1.0)


def _candles(*raw):
    weighted = np.zeros_like(raw[0], dtype=np.float64)
    for r in raw:
        w = np.where(np.abs(r) >= 200, cp.CONFIRMED_WEIGHT, cp.PLAIN_WEIGHT)
        weighted += np.sign(r) * w           # NaN (warmup) propagates
    return np.clip(weighted / cp.NET_SCORE_SCALE, -1.0, 1.0)


def _sr_position(d_sup, d_res):
    has_s, has_r = ~np.isnan(d_sup), ~np.isnan(d_res)
    with np.errstate(divide="ignore", invalid="ignore"):
        total = d_sup + d_res
        both = np.where(total > 0, (d_res - d_sup) / total, 0.0)
    return np.where(has_s & has_r, both,
                    np.where(has_s, 1.0, np.where(has_r, -1.0, np.nan)))


def _clip_unit(x):
    return np.clip(x, -1.0, 1.0)


_ST = "10_3.0"
SIGNALS: tuple[Signal, ...] = (
    # trend
    *(Signal(f"EMA_{n}_POS", TREND, f"price vs EMA {n}", ("close", f"EMA_{n}", "ATR_14"), _ema_pos)
      for n in (20, 50, 200)),
    Signal("EMA_20_50_CROSS", TREND, "EMA 20 vs EMA 50", ("EMA_20", "EMA_50", "ATR_14"), _ema_pos),
    Signal("EMA_50_200_CROSS", TREND, "EMA 50 vs EMA 200", ("EMA_50", "EMA_200", "ATR_14"), _ema_pos),
    Signal("SUPERTREND", TREND, "SuperTrend", (f"SUPERTd_{_ST}", f"SUPERT_{_ST}"), _supertrend),
    Signal("PSAR", TREND, "Parabolic SAR", ("close", "PSAR_0.02_0.2"), lambda c, p: _sign(c - p)),
    Signal("DI_SPREAD", TREND, "+DI vs -DI", ("PLUS_DI_14", "MINUS_DI_14"), _di_spread),
    Signal("AROON", TREND, "Aroon oscillator", ("AROONOSC_14",), lambda a: a / 100.0),
    Signal("ICHI_CLOUD", TREND, "price vs Ichimoku cloud",
           ("close", "ICHI_SPAN_A_26", "ICHI_SPAN_B_26"), _cloud),
    Signal("ICHI_TK", TREND, "Tenkan vs Kijun", ("ICHI_TENKAN_9", "ICHI_KIJUN_26"),
           lambda t, k: _sign(t - k)),
    Signal("LINREG_SLOPE", TREND, "14-bar regression slope", ("LINEARREG_SLOPE_14", "ATR_14"),
           lambda s, a: np.tanh(s / (LINREG_SLOPE_ATR_SCALE * a))),
    Signal("VORTEX", TREND, "Vortex VI+ vs VI-", ("VTXP_14", "VTXM_14"),
           lambda p, m: _clip_unit((p - m) / VORTEX_SCALE)),
    Signal("FIB_SWING", TREND, "last confirmed swing", ("FIB_SWING_DIR",), lambda d: d),
    # momentum
    Signal("RSI", MOMENTUM, "RSI 14", ("RSI_14",), lambda x: oscillator(x, 0, 30, 50, 70, 100)),
    Signal("STOCH", MOMENTUM, "Stochastic %K", ("STOCHk_14_3_3",),
           lambda x: oscillator(x, 0, 20, 50, 80, 100)),
    Signal("WILLR", MOMENTUM, "Williams %R", ("WILLR_14",),
           lambda x: oscillator(x + 100.0, 0, 20, 50, 80, 100)),
    Signal("CCI", MOMENTUM, "CCI 20", ("CCI_20",), lambda x: oscillator(x, -250, -100, 0, 100, 250)),
    Signal("ULTOSC", MOMENTUM, "Ultimate oscillator", ("ULTOSC_7_14_28",),
           lambda x: oscillator(x, 0, 30, 50, 70, 100)),
    Signal("MACD_HIST", MOMENTUM, "MACD histogram", ("MACDh_12_26_9",), _macd_hist),
    Signal("ROC", MOMENTUM, "10-bar rate of change", ("ROC_10", "NATR_14"), _roc),
    # volatility
    Signal("BB_POS", VOLATILITY, "position in Bollinger bands", ("BBP_20_2.0",),
           lambda x: oscillator(x, -0.5, 0.0, 0.5, 1.0, 1.5)),
    Signal("KC_POS", VOLATILITY, "position in Keltner channel",
           ("close", "KCM_20_2.0", "KCU_20_2.0"), _kc_pos),
    # volume
    Signal("OBV_FLOW", VOLUME, "OBV flow", ("OBV", "volume"), _flow),
    Signal("AD_FLOW", VOLUME, "accumulation/distribution flow", ("AD", "volume"), _flow),
    Signal("MFI", VOLUME, "money flow index", ("MFI_14",), lambda x: oscillator(x, 0, 20, 50, 80, 100)),
    Signal("RVOL_CONFIRM", VOLUME, "relative volume on the bar", ("RVOL_20", "LOGRET_1"), _rvol_confirm),
    # pattern
    Signal("CANDLES", PATTERN, "candle patterns",
           tuple(f"CDL_{n[3:]}" for n in DIRECTIONAL_PATTERNS), _candles),
    # support / resistance
    Signal("SR_POSITION", SR, "position between support and resistance",
           ("SR_DIST_SUPPORT_PCT", "SR_DIST_RESIST_PCT"), _sr_position),
    Signal("VP_POS", SR, "price vs volume-profile value area", ("VP_POS",), lambda x: x),
    # multi-timeframe
    Signal("MTF_TREND", MTF, "15m/1h/1d trend alignment", ("MTF_TREND_ALIGN",), lambda x: x),
    Signal("MTF_MOMENTUM", MTF, "15m/1h/1d momentum alignment", ("MTF_MOMENTUM_ALIGN",),
           lambda x: x),
)

# ──────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────


def list_confluence_columns() -> list[str]:
    """All keys returned by compute_confluence, in output order."""
    return ["TECH_SCORE", *(f"TECH_SCORE_{f.upper()}" for f in FAMILIES), "TECH_REASONS"]


def compute_signals(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Every signal in ``SIGNALS`` whose inputs are all present, as float64 in [-1, 1].

    Each rule propagates NaN itself (warmup in → NaN out), so a rule may also
    give a missing input a meaning (SR_POSITION with no resistance above).
    """
    out: dict[str, np.ndarray] = {}
    for sig in SIGNALS:
        if not all(c in df.columns for c in sig.inputs):
            continue
        arrays = [df[c].to_numpy(dtype=np.float64) for c in sig.inputs]
        with np.errstate(invalid="ignore"):
            values = np.asarray(sig.fn(*arrays), dtype=np.float64)
        out[sig.name] = np.clip(values, -1.0, 1.0)     # NaN stays NaN
    return out


def family_weights(df: pd.DataFrame) -> np.ndarray:
    """Regime-blended family weights (before renormalisation) and score multiplier.

    Returns an (n_bars, len(FAMILIES) + 1) array: one column per family in
    FAMILIES order, then the blended ``REGIME_SCORE_MULTIPLIER``.
    """
    n = len(df)
    names = ["neutral", *REGIME_NAMES.values()]
    table = np.array([[FAMILY_WEIGHTS[r][f] for f in FAMILIES] + [REGIME_SCORE_MULTIPLIER[r]]
                      for r in names])
    regime = df["REGIME"].to_numpy(dtype=np.float64) if "REGIME" in df else np.full(n, np.nan)
    conf = (df["REGIME_CONFIDENCE"].to_numpy(dtype=np.float64)
            if "REGIME_CONFIDENCE" in df else np.full(n, np.nan))
    known = ~np.isnan(regime) & ~np.isnan(conf)
    row = np.zeros(n, dtype=np.int64)
    codes = list(REGIME_NAMES)
    row[known] = 1 + np.searchsorted(codes, regime[known].astype(np.int64))
    c = np.where(known, np.clip(conf, 0.0, 1.0), 0.0)[:, None]
    return c * table[row] + (1.0 - c) * table[0]


def compute_confluence(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Group-normalised, regime-weighted technical score.

    Parameters
    ----------
    df : DataFrame
        OHLCV plus any upstream technical columns on one bar index, oldest bar
        first. Missing columns just drop the signals that need them. Not mutated.

    Returns
    -------
    dict[str, np.ndarray]
        Keyed as ``list_confluence_columns()``. Scores are float64; TECH_SCORE is
        NaN until every family in the frame has a value. TECH_REASONS is an object array of lists of str.
    """
    signals, fam_scores, contrib, has_score = _evaluate(df)
    score = np.where(has_score, np.clip(contrib.sum(axis=1), -1.0, 1.0), np.nan)

    out: dict[str, np.ndarray] = {"TECH_SCORE": score}
    for j, fam in enumerate(FAMILIES):
        out[f"TECH_SCORE_{fam.upper()}"] = fam_scores[:, j]
    out["TECH_REASONS"] = _reasons(list(signals), contrib, has_score)
    return out


def add_confluence(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame: ``df`` plus the confluence columns (one pd.concat)."""
    cols = compute_confluence(df)
    return pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)


def signal_contributions(df: pd.DataFrame) -> pd.DataFrame:
    """Per-signal contribution to TECH_SCORE; each row sums to TECH_SCORE (0 if NaN)."""
    signals, _, contrib, _ = _evaluate(df)
    return pd.DataFrame(contrib, index=df.index, columns=list(signals))


# ──────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────

_FAMILY_OF = {s.name: s.family for s in SIGNALS}
_LABEL_OF = {s.name: s.label for s in SIGNALS}


def _evaluate(df: pd.DataFrame):
    """Signals, family means (n, n_families), per-signal contributions (n, n_signals)
    and the mask of bars where TECH_SCORE is defined."""
    n = len(df)
    signals = compute_signals(df)
    names = list(signals)
    fam_idx = np.array([FAMILIES.index(_FAMILY_OF[s]) for s in names], dtype=np.int64)
    stack = np.column_stack([signals[s] for s in names]) if names else np.empty((n, 0))

    # A family is scored only on bars where every one of its signals has a
    # value: no partial averages while some members are still warming up.
    in_frame = np.array([(fam_idx == j).any() for j in range(len(FAMILIES))])
    fam_scores = np.full((n, len(FAMILIES)), np.nan)
    counts = np.zeros(len(FAMILIES))
    for j in np.flatnonzero(in_frame):
        cols = fam_idx == j
        counts[j] = cols.sum()
        fam_scores[:, j] = stack[:, cols].mean(axis=1)      # NaN in any member → NaN

    # The score needs every family that is in the frame.
    complete = ~np.isnan(fam_scores[:, in_frame]).any(axis=1) if in_frame.any() else np.zeros(n, bool)
    wm = family_weights(df)
    w = np.where(in_frame, wm[:, :-1], 0.0)
    w = w / w.sum(axis=1, keepdims=True) if in_frame.any() else w
    with np.errstate(invalid="ignore", divide="ignore"):
        # weight of one signal = multiplier * family weight / signals in the family
        share = np.where(in_frame, wm[:, -1:] * w / np.maximum(counts, 1), 0.0)
    contrib = np.where(complete[:, None], np.nan_to_num(stack) * share[:, fam_idx], 0.0)
    return signals, fam_scores, contrib, complete


def _reasons(names: list[str], contrib: np.ndarray, has_score: np.ndarray) -> np.ndarray:
    n = len(has_score)
    out = np.empty(n, dtype=object)
    top = np.argsort(-np.abs(contrib), axis=1, kind="stable")[:, :REASONS_TOP_N]
    for i in range(n):
        items = []
        if has_score[i]:
            for k in top[i]:
                c = contrib[i, k]
                if abs(c) < 0.005:      # rounds to ±0.00: not a reason
                    continue
                side = "bullish" if c > 0 else "bearish"
                items.append(f"{_LABEL_OF[names[k]]}: {side} ({c:+.2f})")
        out[i] = items
    return out
