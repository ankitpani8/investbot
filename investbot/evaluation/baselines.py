"""
Trivial baseline factors a technical signal has to beat. Pure compute, one asset.

Every value at bar t uses closes up to t only (trailing windows), and the
sign is chosen so that *higher = expected to outperform* under the classic
anomaly:

    BASE_MOM_20      close[t] / close[t-20] - 1                 20-day momentum
    BASE_MOM_12_1    close[t-21] / close[t-252] - 1             12-1 month momentum (skips the last month)
    BASE_REV_5       -(close[t] / close[t-5] - 1)               5-day reversal
    BASE_LOWVOL_20   -std(log returns over t-19..t, ddof=1)     low-volatility anomaly

Warmup bars are NaN; nothing is dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASELINE_COLUMNS: tuple[str, ...] = ("BASE_MOM_20", "BASE_MOM_12_1", "BASE_REV_5", "BASE_LOWVOL_20")

MOM_SHORT: int = 20
MOM_LONG: int = 252
MOM_SKIP: int = 21
REV_WINDOW: int = 5
VOL_WINDOW: int = 20


def _lag(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    if k < x.size:
        out[k:] = x[: x.size - k]
    return out


def compute_baselines(bars: pd.DataFrame) -> dict[str, np.ndarray]:
    close = bars["close"].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        logret = np.log(close / _lag(close, 1))
        vol = pd.Series(logret).rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std(ddof=1).to_numpy()
        return {
            "BASE_MOM_20": close / _lag(close, MOM_SHORT) - 1.0,
            "BASE_MOM_12_1": _lag(close, MOM_SKIP) / _lag(close, MOM_LONG) - 1.0,
            "BASE_REV_5": -(close / _lag(close, REV_WINDOW) - 1.0),
            "BASE_LOWVOL_20": -vol,
        }
