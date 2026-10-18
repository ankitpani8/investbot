"""
Forward-return labels. Pure compute, one asset at a time.

Timing (daily bars)
    A signal computed on bar t is known after t's close (daily bars count from
    16:00 IST, see ``scorer.py``), so the earliest tradable price is the
    **next bar's open**. The label is

        FWD_RET_<h>    = close[t+h] / open[t+1] - 1      (entry next open, exit close t+h)
        FWD_CC_RET_<h> = close[t+h] / close[t]  - 1      (close-to-close, for comparison only;
                                                          it assumes a fill at t's close)

    ``t+k`` is the k-th *stored* bar after t for that asset (a missing session
    is skipped, not filled). The last h bars have no label (NaN).

Separation
    Labels live in their own frame and are never passed to a feature function.
    ``build_dataset`` returns (features, labels) as two objects; the metrics
    join them per date only to score a signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Default forward horizons, in bars.
HORIZONS: tuple[int, ...] = (1, 5, 20)


def label_columns(horizons=HORIZONS) -> list[str]:
    return [f"FWD_RET_{h}" for h in horizons] + [f"FWD_CC_RET_{h}" for h in horizons]


def _lead(x: np.ndarray, k: int) -> np.ndarray:
    """x[t+k] at position t; NaN past the end."""
    out = np.full(x.shape, np.nan)
    if k < x.size:
        out[: x.size - k] = x[k:]
    return out


def forward_returns(bars: pd.DataFrame, horizons=HORIZONS) -> dict[str, np.ndarray]:
    """``FWD_RET_<h>`` (next-open entry) and ``FWD_CC_RET_<h>`` for each h, on ``bars``' index."""
    open_ = bars["open"].to_numpy(dtype=np.float64)
    close = bars["close"].to_numpy(dtype=np.float64)
    entry = _lead(open_, 1)
    out: dict[str, np.ndarray] = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for h in horizons:
            if h < 1:
                raise ValueError("horizons must be >= 1")
            out[f"FWD_RET_{h}"] = _lead(close, h) / entry - 1.0
        for h in horizons:
            out[f"FWD_CC_RET_{h}"] = _lead(close, h) / close - 1.0
    return out
