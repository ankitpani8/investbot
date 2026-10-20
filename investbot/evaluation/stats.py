"""
Small statistics helpers for the evaluation harness (numpy only, no scipy).

    newey_west_se      HAC standard error of a sample mean (Bartlett kernel)
    newey_west_tstat   mean / Newey-West standard error
    normal_pvalue      two-sided p-value of a t-stat under the normal approximation
    bh_adjust          Benjamini–Hochberg adjusted q-values
    spearman           rank correlation of two 1-D arrays (NaN pairs ignored)
    acf                sample autocorrelation at given lags
    andrews_bartlett_lag          automatic Newey-West lag (Andrews 1991, AR(1) plug-in)
    block_bootstrap_mean_pvalue   stationary block bootstrap p-value of H0 mean = 0

Newey-West
    For x_1..x_n with mean m and autocovariances
    ``gamma_l = (1/n) * sum_{t>l} (x_t - m)(x_{t-l} - m)``,
    ``Var(mean) = (gamma_0 + 2 * sum_{l=1..L} (1 - l/(L+1)) * gamma_l) / n``.
    With lag L = 0 this is the plain (biased, 1/n) variance of the mean.
    The harness uses L = h - 1 for an h-bar forward return: consecutive daily
    observations share h - 1 bars of the same return window.

Autocorrelation-robust inference (task 340, pre-registered before any run on the DB)
    The fixed lag h - 1 only covers the label overlap. A persistent IC series
    (a slow signal, a regime in the cross-section) has autocorrelation beyond
    it, and the fixed-lag t then over-rejects (task 330: |t| ~ 4 at a true
    mean of 0). Two robust p-values are computed next to the existing one,
    with the constants below fixed before any result was seen:
    ``newey_west_tstat`` at the ``andrews_bartlett_lag`` (never below h - 1),
    and ``block_bootstrap_mean_pvalue`` of the centred series.
"""
from __future__ import annotations

import math

import numpy as np

from investbot.evaluation.performance import BOOTSTRAP_MIN_PERIODS, stationary_bootstrap_indices

# ──────────────────────────────────────────────────────────
# Pre-registered constants of the robust IC inference (task 340). Do not tune.
# ──────────────────────────────────────────────────────────

#: Draws of the stationary block bootstrap behind the bootstrap p-value of a mean.
ROBUST_BOOT_N: int = 2000
#: Seed of that bootstrap, the same for every test (common random numbers across tests).
ROBUST_BOOT_SEED: int = 20261020
#: The automatic (Andrews) Newey-West lag is capped at ``floor(ROBUST_MAX_LAG_FRAC * n)``.
ROBUST_MAX_LAG_FRAC: float = 0.25
#: The lag-1 autocorrelation that drives the Andrews bandwidth is clipped to
#: ±``ROBUST_RHO_CLIP``, so a near-unit-root series does not blow the lag up.
ROBUST_RHO_CLIP: float = 0.97


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


# ──────────────────────────────────────────────────────────
# Autocorrelation-robust inference (task 340)
# ──────────────────────────────────────────────────────────

def acf(x, lags) -> dict[int, float]:
    """
    Sample autocorrelation of ``x`` (non-finite values removed first) at each lag:
    ``sum_i (x_i - m)(x_{i+l} - m) / sum_i (x_i - m)^2`` with the full-sample mean m
    (the statsmodels ``acf`` convention, as ``regime_study._lag1_autocorr``).

    NaN for a lag with ``l >= n - 1`` (fewer than two pairs) or a constant series.
    """
    a = _finite(x)
    n = a.size
    d = a - a.mean() if n else a
    den = float(d @ d)
    out: dict[int, float] = {}
    for lag in lags:
        lag = int(lag)
        if lag < 0:
            raise ValueError("lags must be >= 0")
        if lag >= n - 1 or den <= 0.0:
            out[lag] = math.nan
        else:
            out[lag] = float(d[lag:] @ d[:n - lag]) / den
    return out


def andrews_bartlett_lag(x, min_lag: int = 0, max_lag_frac: float = ROBUST_MAX_LAG_FRAC) -> int:
    """
    Automatic Newey-West lag: Andrews (1991) AR(1) plug-in bandwidth for the Bartlett kernel.

    With rho = lag-1 ``acf`` of ``x`` clipped to ±``ROBUST_RHO_CLIP`` and n finite values,

        alpha = 4 rho^2 / ((1 - rho)^2 (1 + rho)^2)
        S     = 1.1447 (alpha n)^(1/3)

    Andrews weighs autocovariance l by ``1 - l / S`` (zero from l >= S on), while
    ``newey_west_se`` weighs it by ``1 - l / (lag + 1)``. Setting lag + 1 = S and
    rounding up so that no lag Andrews keeps is cut gives ``lag = ceil(S) - 1``.

    Returns ``max(min_lag, ceil(S) - 1)``, capped at ``floor(max_lag_frac * n)``
    (the cap wins over ``min_lag``). A series whose rho is undefined (n < 3 or constant)
    gets ``min_lag`` under the same cap. The harness passes ``min_lag = h - 1`` so the
    lag never drops below the label overlap. For white noise rho is about ±1/sqrt(n),
    so at n = 700 ``ceil(S) - 1`` is typically 1, not 0.
    """
    a = _finite(x)
    n = a.size
    cap = int(math.floor(max_lag_frac * n))
    rho = acf(a, (1,))[1]
    if not np.isfinite(rho):
        return max(0, min(min_lag, cap))
    rho = min(max(rho, -ROBUST_RHO_CLIP), ROBUST_RHO_CLIP)
    alpha = 4.0 * rho * rho / ((1.0 - rho) ** 2 * (1.0 + rho) ** 2)
    s = 1.1447 * (alpha * n) ** (1.0 / 3.0)
    lag = max(int(min_lag), int(math.ceil(s)) - 1)
    return max(0, min(lag, cap))


def block_bootstrap_mean_pvalue(x, mean_block: float, n_boot: int = ROBUST_BOOT_N,
                                seed: int = ROBUST_BOOT_SEED) -> float:
    """
    Two-sided stationary block bootstrap p-value of H0: mean(x) = 0.

    The **centred** series ``x - mean(x)`` (finite values only, in order) is resampled
    with ``performance.stationary_bootstrap_indices`` (mean block ``mean_block``,
    ``default_rng(seed)``), which imposes the null while keeping the dependence within
    blocks. With mean* the mean of each draw,

        p = (1 + #{|mean*| >= |mean(x)|}) / (1 + n_boot)

    so p is never 0. NaN with fewer than ``performance.BOOTSTRAP_MIN_PERIODS`` finite values.
    The same seed for every call gives every test the same draws for a given (n, block).
    """
    a = _finite(x)
    n = a.size
    if n < BOOTSTRAP_MIN_PERIODS:
        return math.nan
    mean = float(a.mean())
    d = a - mean
    idx = stationary_bootstrap_indices(n, max(float(mean_block), 1.0), n_boot,
                                       np.random.default_rng(seed))
    boot = d[idx].mean(axis=1)
    return (1.0 + float((np.abs(boot) >= abs(mean)).sum())) / (1.0 + n_boot)
