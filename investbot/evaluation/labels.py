"""
Forward-return labels. Pure compute, one asset at a time.

Timing (daily bars)
    A signal computed on bar t is known after t's close (daily bars count from
    16:00 IST, see ``scorer.py``), so the earliest tradable price is the
    **next bar's open**. The price labels are

        FWD_RET_<h>    = close[t+h] / open[t+1] - 1      (entry next open, exit close t+h)
        FWD_CC_RET_<h> = close[t+h] / close[t]  - 1      (close-to-close, for comparison only;
                                                          it assumes a fill at t's close)
        FWD_OO_RET_<h> = open[t+1+h] / open[t+1] - 1     (open-to-open: enter at the next open,
                                                          exit at the open h bars later)

    ``FWD_RET`` drops the overnight gap close[t+h] -> open[t+h+1]. ``FWD_OO_RET``
    keeps it: consecutive h-bar periods starting every h bars tile the timeline
    with no gap, so they compound into a continuously-held portfolio
    (``metrics.held_portfolio``).

    ``t+k`` is the k-th *stored* bar after t for that asset (a missing session
    is skipped, not filled). The last h bars have no ``FWD_RET`` / ``FWD_CC_RET``
    label and the last h + 1 bars no ``FWD_OO_RET`` label (NaN).

Total return (task 200)
    ``dividends`` is the cash dividend per share credited on each stored bar
    (its ex-date, ``corporate_actions.dividends_on_bars``). D[a..b] is the sum
    over stored bars a..b; dividends are not reinvested within a period.

        FWD_TR_RET_<h>    = (close[t+h]   + D[t+2 .. t+h])   / open[t+1]  - 1
        FWD_TR_CC_RET_<h> = (close[t+h]   + D[t+1 .. t+h])   / close[t]   - 1
        FWD_TR_OO_RET_<h> = (open[t+1+h]  + D[t+2 .. t+1+h]) / open[t+1]  - 1

    A dividend with ex-date bar i is earned iff the position is held over the
    close[i-1] -> open[i] boundary (held at the record date, the price drops at
    i's open). So entering *at* the ex-date's open (open[t+1], bar t+1 = i) does
    not earn it, holding from close[t] (bar t+1 = i) does, and exiting *at* the
    ex-date's open (open[t+1+h], bar t+1+h = i) does. With no dividends (None or
    all zeros) every TR column equals its price column exactly. Dividend tax
    (slab rate) is not modelled.

Split safety
    ``invalid_from`` is a boolean per bar, True on the ex-date bar i of a split
    whose earlier history was never re-adjusted (``quality.detect_unadjusted_splits``),
    i.e. close[i-1] and open[i] are on different share scales. Every label, price
    and TR, whose period contains that close[i-1] -> open[i] boundary is NaN (the
    same bar windows as the dividends above: i in t+2..t+h, t+1..t+h and
    t+2..t+1+h). Nothing is dropped; the bar stays with NaN labels.

Separation
    Labels live in their own frame and are never passed to a feature function.
    ``build_dataset`` returns (features, labels) as two objects; the metrics
    join them per date only to score a signal. Dividends and split flags only
    enter labels, never features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Default forward horizons, in bars.
HORIZONS: tuple[int, ...] = (1, 5, 20)

#: Label stems: price-only (pre-task-200) and total return, in column order.
PRICE_STEMS: tuple[str, ...] = ("FWD_RET", "FWD_CC_RET", "FWD_OO_RET")
TR_STEMS: tuple[str, ...] = ("FWD_TR_RET", "FWD_TR_CC_RET", "FWD_TR_OO_RET")


def label_columns(horizons=HORIZONS, stems=PRICE_STEMS + TR_STEMS) -> list[str]:
    return [f"{s}_{h}" for s in stems for h in horizons]


def _lead(x: np.ndarray, k: int) -> np.ndarray:
    """x[t+k] at position t; NaN past the end."""
    out = np.full(x.shape, np.nan)
    if k < x.size:
        out[: x.size - k] = x[k:]
    return out


def forward_returns(bars: pd.DataFrame, horizons=HORIZONS, dividends: np.ndarray | None = None,
                    invalid_from: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Price and total-return labels for each h, on ``bars``' index (see the module docstring).

    ``dividends``: per-bar dividend per share (None = all zero). ``invalid_from``:
    per-bar bool, True on the ex-date bar of an unadjusted split (None = none).
    """
    open_ = bars["open"].to_numpy(dtype=np.float64)
    close = bars["close"].to_numpy(dtype=np.float64)
    n = close.size
    div = (np.zeros(n) if dividends is None
           else np.asarray(dividends, dtype=np.float64).reshape(n))
    bad = (np.zeros(n, dtype=bool) if invalid_from is None
           else np.asarray(invalid_from, dtype=bool).reshape(n))
    cum_div = np.cumsum(div)
    cum_bad = np.cumsum(bad.astype(np.float64))
    entry = _lead(open_, 1)

    for h in horizons:
        if h < 1:
            raise ValueError("horizons must be >= 1")
    # (stem, TR stem, numerator, denominator, first and last bar offset of the period's
    #  close[i-1] -> open[i] boundaries): one window serves dividends and split flags
    specs = [
        ("FWD_RET", "FWD_TR_RET", lambda h: _lead(close, h), entry, 2, lambda h: h),
        ("FWD_CC_RET", "FWD_TR_CC_RET", lambda h: _lead(close, h), close, 1, lambda h: h),
        ("FWD_OO_RET", "FWD_TR_OO_RET", lambda h: _lead(open_, 1 + h), entry, 2, lambda h: 1 + h),
    ]
    price: dict[str, np.ndarray] = {}
    tr: dict[str, np.ndarray] = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for stem, tr_stem, num, den, a, b in specs:
            for h in horizons:
                exit_ = num(h)
                d = _dividend_window(cum_div, a, b(h))
                split = _flag_window(cum_bad, a, b(h))
                price[f"{stem}_{h}"] = np.where(split, np.nan, exit_ / den - 1.0)
                tr[f"{tr_stem}_{h}"] = np.where(split, np.nan, (exit_ + d) / den - 1.0)
    return {**price, **tr}


def _dividend_window(cum: np.ndarray, a: int, b: int) -> np.ndarray:
    """D[t+a .. t+b] at position t; 0 for an empty window (a > b)."""
    if a > b:
        return np.zeros(cum.size)
    upper = _lead(cum, b)
    lower = _lead(cum, a - 1) if a >= 1 else np.zeros(cum.size)
    return upper - lower


def _flag_window(cum: np.ndarray, a: int, b: int) -> np.ndarray:
    """True where any flagged bar lies in t+a .. t+b (False past the end)."""
    with np.errstate(invalid="ignore"):
        return _dividend_window(cum, a, b) > 0.5
