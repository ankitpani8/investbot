"""
Risk model for the allocator (master plan §2.7, layer 2): shrunk covariance,
equal-risk-contribution (risk parity) weights and position/sector caps.

Pure compute on numpy/pandas. No DB I/O. Price inputs are wide frames
(DatetimeIndex x tickers) of daily closes. Point-in-time use goes through the
explicit ``as_of`` of ``estimate_covariance``: only rows dated <= ``as_of`` are
read, so appending future rows never changes an estimate.

Scale: the universe is ~50 assets x a 250-row window, i.e. a 50 x 50 matrix from
a 250 x 50 array. Everything here is a few dense matrix products. Nothing loads
intraday data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from investbot.config.settings import MAX_POSITION_WEIGHT, MAX_SECTOR_WEIGHT

COV_WINDOW = 250   # trailing return rows used for the covariance (~1 trading year)
COV_MIN_OBS = 120  # minimum non-NaN returns per asset, and complete rows overall

_WEIGHT_TOL = 1e-12  # slack when checking caps and sums


@dataclass
class CovEstimate:
    """Annualised shrunk covariance and how it was obtained.

    ``cov`` is indexed by the assets kept, in input column order. ``shrinkage`` is
    the Ledoit-Wolf intensity in [0, 1]. ``n_obs`` is the number of complete rows
    (every kept asset non-NaN) the estimate used. ``excluded`` maps each dropped
    ticker to the reason.
    """

    cov: pd.DataFrame
    shrinkage: float
    n_obs: int
    excluded: dict[str, str] = field(default_factory=dict)


def _check_index(frame: pd.DataFrame, name: str) -> None:
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError(f"{name} index must be sorted ascending and unique")


def log_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns ln(C_t / C_{t-1}), same shape as ``close``.

    Row t uses rows t and t-1 only. A NaN close makes its own return and the next
    one NaN; nothing is forward-filled, so the return after a gap is NaN rather
    than a multi-day return. The first row is NaN.
    """
    _check_index(close, "close")
    values = close.to_numpy(dtype=float)
    if (values[~np.isnan(values)] <= 0).any():
        raise ValueError("close prices must be positive")
    prev = close.shift(1).to_numpy(dtype=float)
    out = np.log(values / prev)
    return pd.DataFrame(out, index=close.index, columns=close.columns)


def _ledoit_wolf_constant_correlation(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Shrunk covariance (per period, not annualised) and the intensity used.

    ``x`` is T x N with no NaNs. See ``estimate_covariance`` for the formula.
    """
    t, n = x.shape
    y = x - x.mean(axis=0)
    s = y.T @ y / t
    if n == 1:
        return s, 0.0

    var = np.diag(s)
    sd = np.sqrt(var)
    corr = s / np.outer(sd, sd)
    r_bar = (corr.sum() - n) / (n * (n - 1))
    target = r_bar * np.outer(sd, sd)
    np.fill_diagonal(target, var)

    y2 = y**2
    pi_mat = y2.T @ y2 / t - s**2
    pi_hat = pi_mat.sum()

    theta = (y**3).T @ y / t - var[:, None] * s  # theta[i, j] = AsyCov(s_ii, s_ij)
    np.fill_diagonal(theta, 0.0)
    rho_hat = np.trace(pi_mat) + r_bar * (np.outer(1.0 / sd, sd) * theta).sum()

    gamma_hat = ((target - s) ** 2).sum()
    if gamma_hat <= 0.0:
        return s, 0.0  # sample already equals the target
    kappa = (pi_hat - rho_hat) / gamma_hat
    delta = float(min(1.0, max(0.0, kappa / t)))
    return delta * target + (1.0 - delta) * s, delta


def estimate_covariance(
    returns: pd.DataFrame,
    as_of=None,
    *,
    window: int = COV_WINDOW,
    min_obs: int = COV_MIN_OBS,
    periods_per_year: int = 252,
) -> CovEstimate:
    """Annualised covariance of ``returns`` (wide, DatetimeIndex x tickers).

    Data selection, point in time:
      1. keep rows dated <= ``as_of`` (all rows when None), then the last ``window`` rows;
      2. exclude assets with fewer than ``min_obs`` non-NaN returns in that window;
      3. estimate on the rows where every remaining asset is non-NaN (boolean mask).
         While fewer than ``min_obs`` such rows remain, drop the asset with the most
         NaNs in the window (ties: first by ticker name) and recompute;
      4. exclude assets with zero variance on those rows and recompute.
    Every exclusion is recorded in ``excluded``. When nothing survives, ``cov`` is
    empty and ``n_obs`` is 0.

    Shrinkage: Ledoit & Wolf (2004), "Honey, I Shrunk the Sample Covariance Matrix",
    towards the constant-correlation target. With y the T x N demeaned returns,
    S = yᵀy / T, s_ii its variances and r_ij = s_ij / sqrt(s_ii s_jj):

        r̄     = mean of r_ij over i != j
        F      : f_ii = s_ii,  f_ij = r̄ sqrt(s_ii s_jj)
        π̂     = Σ_ij π̂_ij,          π̂_ij = (1/T) Σ_t (y_ti y_tj - s_ij)²
        θ̂_ii,ij = (1/T) Σ_t (y_ti² - s_ii)(y_ti y_tj - s_ij)
        ρ̂     = Σ_i π̂_ii + Σ_{i!=j} (r̄/2) (sqrt(s_jj/s_ii) θ̂_ii,ij + sqrt(s_ii/s_jj) θ̂_jj,ij)
        γ̂     = Σ_ij (f_ij - s_ij)²
        δ*     = clip((π̂ - ρ̂) / γ̂ / T, 0, 1)
        Σ̂     = δ* F + (1 - δ*) S,   annualised as Σ̂ x periods_per_year.

    The intensity is scale-free, so annualising after shrinking gives the same δ*.
    The target is positive definite for -1/(N-1) < r̄ < 1, so Σ̂ is positive
    definite whenever δ* > 0, including when N is close to T.
    """
    _check_index(returns, "returns")
    if window < 1 or min_obs < 2:
        raise ValueError("window must be >= 1 and min_obs >= 2")

    rows = returns if as_of is None else returns.loc[returns.index <= pd.Timestamp(as_of)]
    win = rows.iloc[-window:]
    valid = win.notna().to_numpy()
    tickers = list(win.columns)
    counts = valid.sum(axis=0)

    excluded: dict[str, str] = {}
    keep = np.ones(len(tickers), dtype=bool)
    for j, ticker in enumerate(tickers):
        if counts[j] < min_obs:
            keep[j] = False
            excluded[ticker] = f"{counts[j]} non-NaN returns in the last {len(win)} rows < min_obs {min_obs}"

    values = win.to_numpy(dtype=float)
    while keep.any():
        complete = valid[:, keep].all(axis=1)
        n_complete = int(complete.sum())
        if n_complete < min_obs:
            candidates = sorted(np.flatnonzero(keep), key=lambda j: (counts[j], tickers[j]))
            worst = candidates[0]
            keep[worst] = False
            excluded[tickers[worst]] = (
                f"most NaNs ({len(win) - counts[worst]}) while only {n_complete} complete "
                f"rows < min_obs {min_obs}"
            )
            continue
        x = values[np.ix_(complete, keep)]
        flat = np.flatnonzero(keep)[x.var(axis=0) <= 0.0]
        if len(flat):
            for j in flat:
                keep[j] = False
                excluded[tickers[j]] = "zero variance"
            continue
        shrunk, delta = _ledoit_wolf_constant_correlation(x)
        kept = [tickers[j] for j in np.flatnonzero(keep)]
        cov = pd.DataFrame(shrunk * periods_per_year, index=kept, columns=kept)
        return CovEstimate(cov=cov, shrinkage=delta, n_obs=n_complete, excluded=excluded)

    return CovEstimate(cov=pd.DataFrame(dtype=float), shrinkage=0.0, n_obs=0, excluded=excluded)


def _check_cov(cov: pd.DataFrame) -> np.ndarray:
    if cov.empty:
        raise ValueError("covariance is empty")
    if list(cov.index) != list(cov.columns):
        raise ValueError("covariance index and columns must match")
    sigma = cov.to_numpy(dtype=float)
    if not np.isfinite(sigma).all():
        raise ValueError("covariance has non-finite entries")
    if not np.allclose(sigma, sigma.T, rtol=1e-10, atol=1e-14):
        raise ValueError("covariance must be symmetric")
    if (np.diag(sigma) <= 0).any():
        raise ValueError("covariance diagonal must be positive")
    return sigma


def _aligned(w: pd.Series, cov: pd.DataFrame) -> np.ndarray:
    if set(w.index) != set(cov.index) or len(w) != len(cov):
        raise ValueError("weights and covariance must cover the same tickers")
    return w.reindex(cov.index).to_numpy(dtype=float)


def inverse_volatility_weights(cov: pd.DataFrame) -> pd.Series:
    """w_i ∝ 1 / sqrt(Σ_ii), summing to 1."""
    sigma = _check_cov(cov)
    inv = 1.0 / np.sqrt(np.diag(sigma))
    return pd.Series(inv / inv.sum(), index=cov.index)


def risk_contributions(w: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Fractional risk contributions w_i (Σw)_i / wᵀΣw. They sum to 1."""
    sigma = _check_cov(cov)
    x = _aligned(w, cov)
    marginal = sigma @ x
    total = x @ marginal
    if total <= 0:
        raise ValueError("portfolio variance must be positive")
    return pd.Series(x * marginal / total, index=cov.index)


def portfolio_volatility(w: pd.Series, cov: pd.DataFrame) -> float:
    """sqrt(wᵀΣw), in the units of ``cov`` (annualised for ``CovEstimate.cov``)."""
    sigma = _check_cov(cov)
    x = _aligned(w, cov)
    return float(np.sqrt(x @ sigma @ x))


def risk_parity_weights(
    cov: pd.DataFrame, *, tol: float = 1e-10, max_iter: int = 10_000
) -> pd.Series:
    """Long-only equal-risk-contribution (ERC) weights, summing to 1.

    Spinu (2013) convex formulation: with b_i = 1/N, minimise

        f(y) = ½ yᵀΣy - Σ_i b_i ln y_i,   y > 0.

    Its stationarity condition y_i (Σy)_i = b_i makes every risk contribution
    equal, and w = y / Σy. f is strictly convex, so Newton's method with
    backtracking (kept inside y > 0) converges quadratically. The start is the
    inverse-volatility portfolio scaled to yᵀΣy = 1, which the optimum satisfies.
    Converged means max_i |RC_i - 1/N| <= ``tol`` on the fractional
    contributions. Raises RuntimeError if that is not reached in ``max_iter``
    Newton steps.

    For N = 2, or when all correlations are equal, ERC equals inverse volatility.
    """
    sigma = _check_cov(cov)
    n = len(sigma)
    b = np.full(n, 1.0 / n)

    def objective(y: np.ndarray) -> float:
        return 0.5 * y @ sigma @ y - b @ np.log(y)

    def rc_error(y: np.ndarray) -> float:
        rc = y * (sigma @ y)
        return float(np.abs(rc / rc.sum() - b).max())

    y = 1.0 / np.sqrt(np.diag(sigma))
    y /= np.sqrt(y @ sigma @ y)
    f = objective(y)
    for _ in range(max_iter):
        if rc_error(y) <= tol:
            w = y / y.sum()
            return pd.Series(w, index=cov.index)
        grad = sigma @ y - b / y
        hess = sigma + np.diag(b / y**2)
        step = -np.linalg.solve(hess, grad)
        decrement = -grad @ step
        t = 1.0
        while (y + t * step <= 0).any():
            t *= 0.5
        while True:
            y_new = y + t * step
            f_new = objective(y_new)
            if f_new <= f - 0.25 * t * decrement or t < 1e-12:
                break
            t *= 0.5
        if t < 1e-12:
            break  # no further progress possible
        y, f = y_new, f_new
    raise RuntimeError(
        f"risk parity did not converge: max risk-contribution error {rc_error(y):.3g} > tol {tol:g}"
    )


def cap_weights(
    w: pd.Series,
    *,
    max_weight: float = MAX_POSITION_WEIGHT,
    sectors: Mapping[str, str] | None = None,
    max_sector_weight: float = MAX_SECTOR_WEIGHT,
) -> pd.Series:
    """Enforce a per-asset cap and (optionally) a per-sector cap.

    ``w`` must be non-negative and sum to 1. Each pass first caps every asset above
    ``max_weight`` at ``max_weight``; otherwise it scales every sector above
    ``max_sector_weight`` down to the cap. Capped assets are frozen and the excess
    goes pro rata (by current weight) to the assets not frozen yet. Passes repeat
    until both constraints hold, so the sum stays 1 and weights already within
    the caps come back unchanged.

    ``sectors`` maps ticker → sector and must cover every ticker in ``w``.
    Raises ValueError when the caps are infeasible (``max_weight × n < 1``, or the
    sectors cannot hold a total of 1 under both caps) or when the excess has
    nowhere to go (all remaining uncapped weight is 0).
    """
    x = w.to_numpy(dtype=float).copy()
    n = len(x)
    if n == 0:
        raise ValueError("weights are empty")
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("weights must be finite and non-negative")
    if abs(x.sum() - 1.0) > 1e-8:
        raise ValueError(f"weights must sum to 1, got {x.sum():.10g}")
    if not 0 < max_weight <= 1 or not 0 < max_sector_weight <= 1:
        raise ValueError("caps must be in (0, 1]")
    if max_weight * n < 1 - _WEIGHT_TOL:
        raise ValueError(f"infeasible: max_weight {max_weight} x {n} assets < 1")

    labels = None
    if sectors is not None:
        missing = [t for t in w.index if t not in sectors]
        if missing:
            raise ValueError(f"no sector for {missing}")
        labels = np.array([sectors[t] for t in w.index], dtype=object)
        capacity = sum(
            min(max_sector_weight, max_weight * int((labels == s).sum())) for s in set(labels)
        )
        if capacity < 1 - _WEIGHT_TOL:
            raise ValueError(
                f"infeasible: sectors can hold at most {capacity:.4g} under "
                f"max_weight {max_weight} and max_sector_weight {max_sector_weight}"
            )

    frozen = np.zeros(n, dtype=bool)

    def redistribute(excess: float) -> None:
        base = x[~frozen].sum()
        if base <= 0:
            raise ValueError("infeasible: no uncapped weight left to absorb the excess")
        x[~frozen] *= 1.0 + excess / base

    for _ in range(2 * n + 1):
        over = x > max_weight + _WEIGHT_TOL
        if over.any():
            excess = (x[over] - max_weight).sum()
            x[over] = max_weight
            frozen |= over
            redistribute(excess)
            continue
        if labels is not None:
            excess = 0.0
            for s in set(labels):
                members = labels == s
                total = x[members].sum()
                if total > max_sector_weight + _WEIGHT_TOL:
                    x[members] *= max_sector_weight / total
                    frozen |= members
                    excess += total - max_sector_weight
            if excess > 0:
                redistribute(excess)
                continue
        return pd.Series(x, index=w.index)
    raise RuntimeError("cap_weights did not converge")
