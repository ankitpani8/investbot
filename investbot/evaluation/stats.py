"""
Small statistics helpers for the evaluation harness (numpy only, no scipy).

    newey_west_se      HAC standard error of a sample mean (Bartlett kernel)
    newey_west_tstat   mean / Newey-West standard error
    normal_pvalue      two-sided p-value of a t-stat under the normal approximation
    bh_adjust          Benjamini–Hochberg adjusted q-values
    spearman           rank correlation of two 1-D arrays (NaN pairs ignored)

Newey-West
    For x_1..x_n with mean m and autocovariances
    ``gamma_l = (1/n) * sum_{t>l} (x_t - m)(x_{t-l} - m)``,
    ``Var(mean) = (gamma_0 + 2 * sum_{l=1..L} (1 - l/(L+1)) * gamma_l) / n``.
    With lag L = 0 this is the plain (biased, 1/n) variance of the mean.
    The harness uses L = h - 1 for an h-bar forward return: consecutive daily
    observations share h - 1 bars of the same return window.
"""
from __future__ import annotations

import math

import numpy as np


def _finite(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).ravel()
    return a[np.isfinite(a)]


def newey_west_se(x, lag: int) -> float:
    """Newey-West standard error of the mean of ``x`` (non-finite values ignored)."""
    if lag < 0:
        raise ValueError("lag must be >= 0")
    a = _finite(x)
    n = a.size
    if n < 2:
        return math.nan
    d = a - a.mean()
    var = float(d @ d) / n
    for lag_l in range(1, min(lag, n - 1) + 1):
        w = 1.0 - lag_l / (lag + 1.0)
        var += 2.0 * w * float(d[lag_l:] @ d[:-lag_l]) / n
    return math.sqrt(max(var, 0.0) / n)


def newey_west_tstat(x, lag: int) -> float:
    """``mean(x) / newey_west_se(x, lag)``; NaN with fewer than 2 finite values, ±inf if constant."""
    a = _finite(x)
    se = newey_west_se(a, lag)
    if not np.isfinite(se):
        return math.nan
    mean = float(a.mean())
    if se == 0.0:           # constant series (e.g. an oracle's IC of 1 every day)
        return math.copysign(math.inf, mean) if mean != 0.0 else math.nan
    return mean / se


def normal_pvalue(t: float) -> float:
    """Two-sided p-value of ``t`` under N(0, 1); NaN stays NaN."""
    if t is None or math.isnan(t):
        return math.nan
    return math.erfc(abs(t) / math.sqrt(2.0))


def bh_adjust(pvalues) -> np.ndarray:
    """
    Benjamini–Hochberg q-values (step-up FDR adjustment), same order as the input.

    NaN p-values are left NaN and do not count towards the number of tests.
    Matches ``statsmodels.stats.multitest.multipletests(p, method="fdr_bh")[1]``.
    """
    p = np.asarray(pvalues, dtype=np.float64).ravel()
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv, kind="stable")
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(ranked, 1.0)
    q[ok] = adj
    return q


def _rank(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) of a finite 1-D array."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size)
    sorted_a = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    """Spearman rank correlation over pairs where both are finite; NaN if < 3 pairs or constant."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return math.nan
    rx, ry = _rank(x[ok]), _rank(y[ok])
    rx -= rx.mean()
    ry -= ry.mean()
    den = math.sqrt(float(rx @ rx) * float(ry @ ry))
    return math.nan if den == 0.0 else float(rx @ ry) / den
