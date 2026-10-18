"""
Support / resistance features: swing pivots, pivot-clustered S/R levels,
rolling volume profile and Fibonacci retracements.

Pure compute module: takes an OHLCV DataFrame and returns a dict of
``{column: np.ndarray}`` (``compute_support_resistance_columns``) or the
input plus those columns assembled with one ``pd.concat``
(``add_support_resistance``). No DB I/O, no ``dropna()``, input not mutated.
Warmup rows are NaN; consumers pick their own cutoff.

No-lookahead contract — read this before touching the pivot code
-----------------------------------------------------------------
A swing pivot high at bar ``j`` needs ``high[j]`` to beat the
``PIVOT_LEFT_BARS`` bars before it *and* the ``PIVOT_RIGHT_BARS`` (k) bars
after it. Those right-hand bars are the future as seen from ``j``, so the
pivot is only **known** at bar ``t = j + k``. Every pivot is therefore
emitted at its confirmation bar ``t``, never back-filled to ``j``:

    PIVOT_HIGH[t] = high[t - k]   if bar t - k is a confirmed pivot high
                    NaN           otherwise

Marking ``PIVOT_HIGH[j]`` instead is the classic lookahead trap: a backtest
would "see" the top on the day it printed. All downstream features here (S/R
levels, Fibonacci swing) only consume pivots whose confirmation bar is
``<= t``. The truncation-invariance tests in
``tests/test_support_resistance.py`` cover every output column.

Pivot rule: ``high[j] >  max(high[j-L : j])`` (strict on the left) and
``high[j] >= max(high[j+1 : j+k+1])`` (ties allowed on the right), so the
first bar of a flat top is the pivot. Lows mirror this. Any NaN in the
window means no pivot.

S/R levels (``SR_*``)
---------------------
At each bar, take the last ``SR_MAX_PIVOTS`` confirmed pivots (highs and lows
together, as plain price levels). Sort them by price and cluster greedily:
a pivot joins the current cluster while it is within the tolerance of the
cluster's lowest price. The tolerance at bar t is
``SR_CLUSTER_ATR_MULT * ATR_14[t]`` in price units (0.25 ATR), so it scales
with the asset's own volatility: a calm stock gets fine levels, a wild one
broad zones. ATR_14 is the module's own causal TA-Lib ATR (Wilder), or the
``ATR_14`` column if the input frame already has one. ``SR_CLUSTER_PCT`` (a
percent of the cluster's lowest price) is a **fallback only**: it applies on
bars where ATR is still NaN (the first ``SR_ATR_PERIOD`` bars, or after a
missing OHLC value, which TA-Lib propagates), and nowhere else.
Because the tolerance moves every bar, levels are re-clustered every bar
(the sort is done once per new pivot).

Each pivot has recency weight ``0.5 ** ((t - j) / SR_RECENCY_HALFLIFE)``. A
cluster's level is the weighted mean of its prices; its "recency-weighted
touches" is the sum of its weights. ``SR_STRENGTH_*`` is that sum divided by
the total weight of **all** pivots in the lookback at bar t, so it is a share
in [0, 1] comparable across assets and timeframes (the shares of all
clusters sum to 1; the nearest support and resistance sum to <= 1). The
common decay factor cancels in the ratio, so a share only changes when a
pivot enters or leaves the lookback or the clusters re-form.

Clusters at or below the close are supports, those above are resistances,
each ranked nearest-first: ``SR_SUPPORT_1`` is the nearest support.
``SR_DIST_SUPPORT_PCT = (close - SR_SUPPORT_1) / close * 100`` (>= 0) and
``SR_DIST_RESIST_PCT = (SR_RESIST_1 - close) / close * 100`` (> 0).
``SR_STRENGTH_*`` is the share of the nearest level (see above).

Volume profile (``VP_*``)
-------------------------
Rolling over the last ``vp_window`` bars **including bar t** (data up to t
is allowed; nothing after it). The window's [min low, max high] range is
split into ``VP_BINS`` bins and each bar's volume is spread uniformly over
its own [low, high] range (zero-range bars go into one bin).
``VP_POC`` is the centre of the heaviest bin. The value area grows from the
POC towards the heavier neighbour until it holds ``VP_VALUE_AREA`` (70%) of
volume; ``VP_VAH`` / ``VP_VAL`` are its outer bin edges.
``VP_POS = (close - mid) / half_width`` clipped to [-1, +1], where mid and
half_width describe [VAL, VAH]. A window with any NaN input is NaN.
``VP_DIST`` is the unclipped companion: the signed distance of the close
from the value area in ATR_14 units, 0 inside [VAL, VAH], ``(close - VAH) /
ATR`` above it and ``(close - VAL) / ATR`` below it. NaN where the profile or
ATR is NaN (or ATR is 0). ``VP_POS`` is unchanged.

Fibonacci (``FIB_*``)
---------------------
The most recent confirmed swing, zig-zag style: consecutive pivots of the
same type are merged into a run keeping the extreme. When a high-run follows
a low-run the swing is up (low → high) and retracements hang down from the
high: ``FIB_x = high - x * (high - low)``. A down swing mirrors this:
``FIB_x = low + x * (high - low)``. If one bar is both a pivot high and a
pivot low (outside bar), the low is processed first on a bullish bar
(close >= open) and the high first otherwise. ``FIB_SWING_DIR`` is +1/-1.
``FIB_NEAREST_DIST_PCT = (close - nearest level) / close * 100`` (signed;
nearest by absolute distance among the five levels).

Performance
-----------
Benchmarked 2026-10-17 on synthetic random-walk data, 1500 daily rows × 50
assets processed one asset at a time: ~16 s total (~0.3 s per asset) on the
dev WSL box (8 GB RAM, timings noisy under load). Real RELIANCE.NS daily
(1234 rows) takes ~0.27 s; 1495 intraday bars with ``vp_window=500`` ~0.5 s.
Memory is bounded per asset: the volume profile works in batches of at most
``_VP_BATCH_ELEMS`` (2M) window×bar×edge elements, i.e. ~50 MB of temporaries
regardless of history length. About 60% of the time is the volume profile,
the rest is S/R levels, which are re-clustered only when a new pivot confirms.
"""
from __future__ import annotations

import bisect
import logging

import numpy as np
import pandas as pd
import talib
from numpy.lib.stride_tricks import sliding_window_view

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Parameters (defaults are listed in docs/DECISIONS_PENDING.md)
# ──────────────────────────────────────────────────────────

PIVOT_LEFT_BARS = 3
PIVOT_RIGHT_BARS = 3          # k: confirmation delay in bars

SR_MAX_PIVOTS = 20            # last N confirmed pivots used for levels
SR_CLUSTER_ATR_MULT = 0.25    # cluster tolerance = this × ATR_14[t] (price units)
SR_ATR_PERIOD = 14
SR_CLUSTER_PCT = 0.5          # FALLBACK only: percent tolerance while ATR is NaN (warmup)
SR_RECENCY_HALFLIFE = 60.0    # bars; pivot weight halves every this many bars
SR_LEVELS = 3                 # SR_SUPPORT_1..3, SR_RESIST_1..3

VP_WINDOW_DAILY = 120
VP_WINDOW_INTRADAY = 500
VP_BINS = 50
VP_VALUE_AREA = 0.70

FIB_RATIOS: dict[str, float] = {
    "FIB_236": 0.236,
    "FIB_382": 0.382,
    "FIB_500": 0.500,
    "FIB_618": 0.618,
    "FIB_786": 0.786,
}

REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

SR_COLUMNS: tuple[str, ...] = (
    "PIVOT_HIGH",
    "PIVOT_LOW",
    *(f"SR_SUPPORT_{i}" for i in range(1, SR_LEVELS + 1)),
    *(f"SR_RESIST_{i}" for i in range(1, SR_LEVELS + 1)),
    "SR_DIST_SUPPORT_PCT",
    "SR_DIST_RESIST_PCT",
    "SR_STRENGTH_SUPPORT",
    "SR_STRENGTH_RESIST",
    "VP_POC",
    "VP_VAH",
    "VP_VAL",
    "VP_POS",
    "VP_DIST",
    *FIB_RATIOS,
    "FIB_NEAREST_DIST_PCT",
    "FIB_SWING_HIGH",
    "FIB_SWING_LOW",
    "FIB_SWING_DIR",
)


# ──────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────

def add_support_resistance(
    df: pd.DataFrame, *, vp_window: int = VP_WINDOW_DAILY
) -> pd.DataFrame:
    """Return a new DataFrame: ``df`` plus every column in ``SR_COLUMNS``."""
    cols = compute_support_resistance_columns(df, vp_window=vp_window)
    return pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1, copy=False)


def compute_support_resistance_columns(
    df: pd.DataFrame,
    *,
    vp_window: int = VP_WINDOW_DAILY,
    pivot_left: int = PIVOT_LEFT_BARS,
    pivot_right: int = PIVOT_RIGHT_BARS,
    sr_max_pivots: int = SR_MAX_PIVOTS,
    sr_cluster_atr_mult: float = SR_CLUSTER_ATR_MULT,
    sr_cluster_pct: float = SR_CLUSTER_PCT,
    sr_halflife: float = SR_RECENCY_HALFLIFE,
    vp_bins: int = VP_BINS,
    vp_value_area: float = VP_VALUE_AREA,
) -> dict[str, np.ndarray]:
    """Compute all S/R columns. Use ``vp_window=VP_WINDOW_INTRADAY`` for 15m bars.

    ATR_14 is read from ``df["ATR_14"]`` if present, else computed here.
    """
    _validate(df)
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    atr = _atr(df, high, low, close)

    out: dict[str, np.ndarray] = {}
    piv = swing_pivots(high, low, left=pivot_left, right=pivot_right)
    out.update(piv)
    out.update(sr_levels(
        piv["PIVOT_HIGH"], piv["PIVOT_LOW"], close,
        right=pivot_right, max_pivots=sr_max_pivots,
        atr=atr, cluster_atr_mult=sr_cluster_atr_mult,
        cluster_pct=sr_cluster_pct, halflife=sr_halflife,
    ))
    out.update(volume_profile(
        high, low, close, volume, atr,
        window=vp_window, bins=vp_bins, value_area=vp_value_area,
    ))
    out.update(fibonacci(piv["PIVOT_HIGH"], piv["PIVOT_LOW"], open_, close, right=pivot_right))
    return {c: out[c] for c in SR_COLUMNS}


# ──────────────────────────────────────────────────────────
# Swing pivots
# ──────────────────────────────────────────────────────────

def swing_pivots(
    high: np.ndarray, low: np.ndarray, *, left: int, right: int
) -> dict[str, np.ndarray]:
    """Pivot price emitted at its confirmation bar ``j + right``; NaN elsewhere."""
    n = len(high)
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    size = left + 1 + right
    if n < size:
        return {"PIVOT_HIGH": ph, "PIVOT_LOW": pl}

    with np.errstate(invalid="ignore"):
        wh = sliding_window_view(high, size)        # row r covers bars r .. r+size-1
        wl = sliding_window_view(low, size)
        ch, cl = wh[:, left], wl[:, left]
        # np.max propagates NaN and any comparison with NaN is False,
        # so a NaN anywhere in the window yields no pivot.
        is_ph = (ch > wh[:, :left].max(axis=1)) & (ch >= wh[:, left + 1:].max(axis=1))
        is_pl = (cl < wl[:, :left].min(axis=1)) & (cl <= wl[:, left + 1:].min(axis=1))

    # Window row r ends at bar r + size - 1 = confirmation bar t.
    t = np.arange(size - 1, n)
    ph[t[is_ph]] = ch[is_ph]
    pl[t[is_pl]] = cl[is_pl]
    return {"PIVOT_HIGH": ph, "PIVOT_LOW": pl}


# ──────────────────────────────────────────────────────────
# Clustered S/R levels
# ──────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Causal ATR_14: the frame's own column if supplied, else TA-Lib's Wilder ATR."""
    if f"ATR_{SR_ATR_PERIOD}" in df.columns:
        return df[f"ATR_{SR_ATR_PERIOD}"].to_numpy(dtype=float)
    return talib.ATR(high, low, close, timeperiod=SR_ATR_PERIOD)


def _cluster(p: list[float], w: list[float], abs_tol: float, rel_tol: float = 0.0):
    """Greedy clustering of price-sorted pivots.

    ``p`` ascending prices, ``w`` their weights. A pivot joins the current
    cluster while ``p <= lowest * (1 + rel_tol) + abs_tol``, where lowest is
    the cluster's lowest price (ATR mode: ``abs_tol`` only; the percent
    fallback: ``rel_tol`` only). Returns
    (levels, cluster weights), both ascending by level; a level is the
    weighted mean price of its cluster.
    """
    levels: list[float] = []
    weights: list[float] = []
    start = 0
    n = len(p)
    for i in range(1, n + 1):
        if i == n or p[i] > p[start] * (1.0 + rel_tol) + abs_tol:
            sw = sum(w[start:i])
            levels.append(sum(x * y for x, y in zip(p[start:i], w[start:i])) / sw)
            weights.append(sw)
            start = i
    return levels, weights


def sr_levels(
    pivot_high: np.ndarray,
    pivot_low: np.ndarray,
    close: np.ndarray,
    *,
    right: int,
    max_pivots: int,
    atr: np.ndarray,
    cluster_atr_mult: float,
    cluster_pct: float,
    halflife: float,
) -> dict[str, np.ndarray]:
    n = len(close)
    out = {f"SR_SUPPORT_{i}": np.full(n, np.nan) for i in range(1, SR_LEVELS + 1)}
    out.update({f"SR_RESIST_{i}": np.full(n, np.nan) for i in range(1, SR_LEVELS + 1)})
    for c in ("SR_DIST_SUPPORT_PCT", "SR_DIST_RESIST_PCT",
              "SR_STRENGTH_SUPPORT", "SR_STRENGTH_RESIST"):
        out[c] = np.full(n, np.nan)

    # Pivot list in confirmation order. Recency is measured from the pivot
    # bar j = t_confirm - right, which is known at confirmation.
    conf_bars: list[int] = []
    prices: list[float] = []
    for t in range(n):
        if not np.isnan(pivot_high[t]):
            conf_bars.append(t)
            prices.append(pivot_high[t])
        if not np.isnan(pivot_low[t]):
            conf_bars.append(t)
            prices.append(pivot_low[t])
    conf_arr = np.array(conf_bars, dtype=int)
    price_arr = np.array(prices, dtype=float)

    sup = [out[f"SR_SUPPORT_{i}"] for i in range(1, SR_LEVELS + 1)]
    res = [out[f"SR_RESIST_{i}"] for i in range(1, SR_LEVELS + 1)]
    sorted_p: list[float] = []
    sorted_w: list[float] = []
    total_w = 0.0
    n_now = 0
    for t in range(n):
        n_prev = n_now
        while n_now < conf_arr.size and conf_arr[n_now] <= t:
            n_now += 1
        if n_now != n_prev:
            # Re-sort only when the pivot set changes. Weights are relative to
            # the newest pivot; the factor common to all weights at bar t
            # cancels in the strength share and in the weighted-mean level.
            lo = max(0, n_now - max_pivots)
            pr = price_arr[lo:n_now]
            bars = conf_arr[lo:n_now] - right
            order = np.argsort(pr, kind="stable")
            w = 0.5 ** ((bars.max() - bars) / halflife)
            sorted_p = pr[order].tolist()
            sorted_w = w[order].tolist()
            total_w = float(w.sum())
        c = close[t]
        if not sorted_p or np.isnan(c):
            continue
        a = atr[t]
        # Fallback tolerance only while ATR is in warmup (NaN).
        if np.isnan(a):
            level_list, weights = _cluster(sorted_p, sorted_w, 0.0, cluster_pct / 100.0)
        else:
            level_list, weights = _cluster(sorted_p, sorted_w, cluster_atr_mult * a)
        split = bisect.bisect_right(level_list, c)   # levels ascending
        for rank in range(min(SR_LEVELS, split)):
            sup[rank][t] = level_list[split - 1 - rank]
        for rank in range(min(SR_LEVELS, len(level_list) - split)):
            res[rank][t] = level_list[split + rank]
        if split > 0:
            out["SR_DIST_SUPPORT_PCT"][t] = (c - level_list[split - 1]) / c * 100.0
            out["SR_STRENGTH_SUPPORT"][t] = weights[split - 1] / total_w
        if split < len(level_list):
            out["SR_DIST_RESIST_PCT"][t] = (level_list[split] - c) / c * 100.0
            out["SR_STRENGTH_RESIST"][t] = weights[split] / total_w
    return out


# ──────────────────────────────────────────────────────────
# Rolling volume profile
# ──────────────────────────────────────────────────────────

def _histograms(h: np.ndarray, l: np.ndarray, v: np.ndarray, bins: int):
    """Volume histograms for a batch of windows. h, l, v: (c, w), NaN-free.

    Each bar's volume is spread uniformly over [low, high]. Instead of a
    bar × bin overlap matrix we build each bar's cumulative share below every
    bin edge, G = clip((edge - low) / (high - low), 0, 1), take the
    volume-weighted sum with one batched matmul, and difference it. A
    zero-range bar is a step at its price. Returns (hist (c, bins), lo, step).
    """
    lo = l.min(axis=1)
    hi = h.max(axis=1)
    span = hi - lo
    step = np.where(span > 0, span, 1.0) / bins
    edges = lo[:, None] + step[:, None] * np.arange(bins + 1)[None, :]
    rng = h - l
    flat = rng <= 0
    inv = np.where(flat, 0.0, 1.0 / np.where(flat, 1.0, rng))
    diff = edges[:, None, :] - l[:, :, None]                      # (c, w, bins+1)
    g = np.clip(diff * inv[:, :, None], 0.0, 1.0)
    if flat.any():
        g[flat] = diff[flat] > 0
    g[:, :, 0] = 0.0       # nothing below the window low ...
    g[:, :, -1] = 1.0      # ... everything below the window high
    cdf = (v[:, None, :] @ g)[:, 0, :]                            # (c, bins+1)
    return np.diff(cdf, axis=1), lo, step


def _value_area(hv: list[float], value_area: float) -> tuple[int, int, int]:
    """(poc, first, last) bin indices of the value area for one histogram."""
    bins = len(hv)
    poc = max(range(bins), key=hv.__getitem__)   # first max, like np.argmax
    a = b = poc
    acc = hv[poc]
    target = value_area * sum(hv)
    while acc < target and (a > 0 or b < bins - 1):
        below = hv[a - 1] if a > 0 else -1.0
        above = hv[b + 1] if b < bins - 1 else -1.0
        if above >= below:
            b += 1
            acc += hv[b]
        else:
            a -= 1
            acc += hv[a]
    return poc, a, b


# Elements per (windows × window × bins) batch: ~16 MB of float64 per temp.
_VP_BATCH_ELEMS = 2_000_000


def volume_profile(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    atr: np.ndarray,
    *,
    window: int,
    bins: int,
    value_area: float,
) -> dict[str, np.ndarray]:
    n = len(close)
    poc = np.full(n, np.nan)
    vah = np.full(n, np.nan)
    val = np.full(n, np.nan)
    if n >= window:
        bad = np.isnan(high) | np.isnan(low) | np.isnan(volume)
        bad_cum = np.concatenate([[0], np.cumsum(bad)])
        ends = np.arange(window - 1, n)
        ends = ends[(bad_cum[ends + 1] - bad_cum[ends + 1 - window]) == 0]
        wh = sliding_window_view(high, window)    # row r = bars r .. r+window-1
        wl = sliding_window_view(low, window)
        wv = sliding_window_view(volume, window)
        chunk = max(1, _VP_BATCH_ELEMS // (window * bins))
        for i in range(0, ends.size, chunk):
            t = ends[i:i + chunk]
            r = t - window + 1
            hist, lo, step = _histograms(wh[r], wl[r], wv[r], bins)
            for row, hv in enumerate(hist.tolist()):
                base, st = lo[row], step[row]
                if wh[r[row]].max() <= base:      # flat window: no range to profile
                    poc[t[row]] = val[t[row]] = vah[t[row]] = base
                    continue
                k, a, b = _value_area(hv, value_area)
                poc[t[row]] = base + st * (k + 0.5)
                val[t[row]] = base + st * a
                vah[t[row]] = base + st * (b + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mid = (vah + val) / 2
        half = (vah - val) / 2
        pos = np.where(half > 0, (close - mid) / np.where(half > 0, half, 1.0), 0.0)
        pos = np.clip(pos, -1.0, 1.0)
    pos[np.isnan(poc) | np.isnan(close)] = np.nan
    # Unclipped distance outside the value area, in ATR units (0 inside).
    with np.errstate(invalid="ignore", divide="ignore"):
        outside = np.where(close > vah, close - vah, np.where(close < val, close - val, 0.0))
        dist = np.where(atr > 0, outside / np.where(atr > 0, atr, 1.0), np.nan)
    dist[np.isnan(poc) | np.isnan(close) | np.isnan(atr)] = np.nan
    return {"VP_POC": poc, "VP_VAH": vah, "VP_VAL": val, "VP_POS": pos, "VP_DIST": dist}


# ──────────────────────────────────────────────────────────
# Fibonacci retracements
# ──────────────────────────────────────────────────────────

def fibonacci(
    pivot_high: np.ndarray,
    pivot_low: np.ndarray,
    open_: np.ndarray,
    close: np.ndarray,
    *,
    right: int,
) -> dict[str, np.ndarray]:
    n = len(close)
    swing_hi = np.full(n, np.nan)
    swing_lo = np.full(n, np.nan)
    swing_dir = np.full(n, np.nan)

    run_type = 0              # +1 in a high-run, -1 in a low-run, 0 before any pivot
    run_ext = np.nan          # extreme of the current run
    prev_ext = np.nan         # extreme of the previous (opposite) run
    cur = (np.nan, np.nan, np.nan)

    for t in range(n):
        events: list[tuple[int, float]] = []
        has_h, has_l = not np.isnan(pivot_high[t]), not np.isnan(pivot_low[t])
        if has_h and has_l:
            # Outside bar: guess the intrabar order from bar j's own candle.
            j = t - right
            bullish = close[j] >= open_[j]
            events = [(-1, pivot_low[t]), (+1, pivot_high[t])]
            if not bullish:
                events.reverse()
        elif has_h:
            events = [(+1, pivot_high[t])]
        elif has_l:
            events = [(-1, pivot_low[t])]

        for typ, price in events:
            if typ == run_type:
                run_ext = max(run_ext, price) if typ > 0 else min(run_ext, price)
            else:
                prev_ext, run_ext, run_type = run_ext, price, typ
        if events and not np.isnan(prev_ext):
            if run_type > 0:
                cur = (run_ext, prev_ext, 1.0)
            else:
                cur = (prev_ext, run_ext, -1.0)
        swing_hi[t], swing_lo[t], swing_dir[t] = cur

    out: dict[str, np.ndarray] = {}
    span = swing_hi - swing_lo
    for name, r in FIB_RATIOS.items():
        out[name] = np.where(swing_dir > 0, swing_hi - r * span, swing_lo + r * span)
    levels = np.column_stack([out[k] for k in FIB_RATIOS])
    with np.errstate(invalid="ignore"):
        diff = close[:, None] - levels
        has = ~np.isnan(diff).any(axis=1)
        nearest = np.full(n, np.nan)
        if has.any():
            k = np.argmin(np.abs(diff[has]), axis=1)
            nearest[has] = diff[has][np.arange(k.size), k] / close[has] * 100.0
    out["FIB_NEAREST_DIST_PCT"] = nearest
    out["FIB_SWING_HIGH"] = swing_hi
    out["FIB_SWING_LOW"] = swing_lo
    out["FIB_SWING_DIR"] = swing_dir
    return out


# ──────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────

def _validate(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"support_resistance.py expects a DatetimeIndex; got {type(df.index).__name__}"
        )
    if not df.index.is_monotonic_increasing:
        raise ValueError("Index must be monotonically increasing (sort_index first)")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")
