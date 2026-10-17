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
a pivot joins the current cluster while it is within ``SR_CLUSTER_PCT`` of
the cluster's lowest price. Each pivot has recency weight
``0.5 ** ((t - j) / SR_RECENCY_HALFLIFE)``. A cluster's level is the
weighted mean of its prices; its strength is the sum of its weights, i.e.
"recency-weighted touches" (a fresh level with 3 touches ≈ 3.0).

Clusters at or below the close are supports, those above are resistances,
each ranked nearest-first: ``SR_SUPPORT_1`` is the nearest support.
``SR_DIST_SUPPORT_PCT = (close - SR_SUPPORT_1) / close * 100`` (>= 0) and
``SR_DIST_RESIST_PCT = (SR_RESIST_1 - close) / close * 100`` (> 0).
``SR_STRENGTH_*`` is the strength of the nearest level.

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

import logging

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Parameters (defaults are listed in docs/DECISIONS_PENDING.md)
# ──────────────────────────────────────────────────────────

PIVOT_LEFT_BARS = 3
PIVOT_RIGHT_BARS = 3          # k: confirmation delay in bars

SR_MAX_PIVOTS = 20            # last N confirmed pivots used for levels
SR_CLUSTER_PCT = 0.5          # percent price tolerance within one cluster
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
    sr_cluster_pct: float = SR_CLUSTER_PCT,
    sr_halflife: float = SR_RECENCY_HALFLIFE,
    vp_bins: int = VP_BINS,
    vp_value_area: float = VP_VALUE_AREA,
) -> dict[str, np.ndarray]:
    """Compute all S/R columns. Use ``vp_window=VP_WINDOW_INTRADAY`` for 15m bars."""
    _validate(df)
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)

    out: dict[str, np.ndarray] = {}
    piv = swing_pivots(high, low, left=pivot_left, right=pivot_right)
    out.update(piv)
    out.update(sr_levels(
        piv["PIVOT_HIGH"], piv["PIVOT_LOW"], close,
        right=pivot_right, max_pivots=sr_max_pivots,
        cluster_pct=sr_cluster_pct, halflife=sr_halflife,
    ))
    out.update(volume_profile(
        high, low, close, volume,
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

def _cluster(prices: np.ndarray, bars: np.ndarray, cluster_pct: float, halflife: float):
    """Greedy price clustering. Returns (levels, member-bar arrays) sorted by level.

    Levels use weights relative to the newest pivot. Every weight at bar t
    is that relative weight times the same factor, so the weighted mean does
    not depend on t and clusters only change when a pivot is added or dropped.
    """
    order = np.argsort(prices, kind="stable")
    p, b = prices[order], bars[order]
    w = 0.5 ** ((b.max() - b) / halflife)
    levels: list[float] = []
    members: list[np.ndarray] = []
    start = 0
    for i in range(1, len(p) + 1):
        if i == len(p) or p[i] > p[start] * (1 + cluster_pct / 100.0):
            sl = slice(start, i)
            levels.append(float(np.sum(p[sl] * w[sl]) / np.sum(w[sl])))
            members.append(b[sl])
            start = i
    return np.array(levels), members


def sr_levels(
    pivot_high: np.ndarray,
    pivot_low: np.ndarray,
    close: np.ndarray,
    *,
    right: int,
    max_pivots: int,
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
    levels = np.empty(0)
    level_list: list[float] = []
    base_strength: list[float] = []   # strength of each level at bar t0
    t0 = 0
    n_now = 0
    for t in range(n):
        n_prev = n_now
        while n_now < conf_arr.size and conf_arr[n_now] <= t:
            n_now += 1
        if n_now != n_prev:
            lo = max(0, n_now - max_pivots)
            levels, members = _cluster(
                price_arr[lo:n_now], conf_arr[lo:n_now] - right, cluster_pct, halflife
            )
            level_list = levels.tolist()
            t0 = t
            base_strength = [float(np.sum(0.5 ** ((t0 - m) / halflife))) for m in members]
        c = close[t]
        if not level_list or np.isnan(c):
            continue
        # Strengths all decay by the same factor between pivot events.
        decay = 0.5 ** ((t - t0) / halflife)
        split = int(np.searchsorted(levels, c, side="right"))   # levels ascending
        for rank in range(min(SR_LEVELS, split)):
            sup[rank][t] = level_list[split - 1 - rank]
        for rank in range(min(SR_LEVELS, len(level_list) - split)):
            res[rank][t] = level_list[split + rank]
        if split > 0:
            out["SR_DIST_SUPPORT_PCT"][t] = (c - level_list[split - 1]) / c * 100.0
            out["SR_STRENGTH_SUPPORT"][t] = base_strength[split - 1] * decay
        if split < len(level_list):
            out["SR_DIST_RESIST_PCT"][t] = (level_list[split] - c) / c * 100.0
            out["SR_STRENGTH_RESIST"][t] = base_strength[split] * decay
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
    return {"VP_POC": poc, "VP_VAH": vah, "VP_VAL": val, "VP_POS": pos}


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
