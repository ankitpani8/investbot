"""
Risk and trade-quality KPIs of master plan §9.1 on a series of simple returns.

Pure functions (numpy / pandas only, no DB, no I/O). Every function takes a 1-D
array or Series of **simple per-period returns** (``p_t / p_{t-1} - 1``) and
ignores non-finite values through a boolean mask (nothing is dropped or
reindexed). Fewer than 2 finite values gives NaN, never an exception. A value at
period *t* is used only together with the periods before and after it that the
caller passes in; nothing here looks beyond the given series.

Per period, not per day
    A "period" is whatever the caller's bar is: a day for a daily equity curve,
    ``h`` bars for the harness's non-overlapping holding periods. Ratios
    (``sharpe``, ``sortino``) are annualised with ``periods_per_year``
    (``252 / h`` in the harness). ``var_cvar`` and ``distribution_stats`` are
    **not** annualised: with h = 20 the VaR is a 20-bar loss, not a daily one.
    ``drawdown_stats`` counts durations in periods (multiply by h for bars).

Master plan §9.1 coverage
    Covered: Sharpe (against a risk-free rate), Sortino, Calmar, maximum
    drawdown and its duration, VaR / CVaR, win rate, average win / loss,
    profit factor, return distribution (skew, excess kurtosis), monthly
    returns, concentration (Herfindahl of weights).
    Not covered here: CAGR and volatility (``metrics.held_portfolio``), turnover
    (``metrics.turnover``), tax drag and after-tax return (the tax layer), and
    the §9.2 go/no-go thresholds, which apply to the full after-tax strategy.

Bootstrap intervals (``bootstrap_interval``)
    A stationary block bootstrap (Politis & Romano 1994) of the given
    per-period series. In the harness these are the **non-overlapping holding-
    period returns of the OOS window** (``metrics.held_periods``: net and
    universe, resampled jointly so each period's pair stays together); nothing
    else is resampled and no feature, label or signal is touched. Blocks start
    at a uniform index, have geometric lengths with mean ``mean_block`` and wrap
    around the end of the series (circular).

    Pre-registered constants, fixed before any run and never searched over:
    ``BOOTSTRAP_N = 2000`` resamples, ``BOOTSTRAP_LEVEL = 0.90`` (percentile
    interval), ``BOOTSTRAP_SEED = 20261019`` (one seed for every row, so all
    signals share common random numbers), ``BOOTSTRAP_MIN_PERIODS = 20`` and
    the mean block length ``bootstrap_mean_block(n) = max(1, round(n ** (1/3)))``.
    No automatic block-length selection (e.g. Politis–White 2004) is used.

    The intervals are **per row and not adjusted for multiple comparisons**:
    over ~100 rows about 10% of 90% intervals exclude 0 by chance even when
    nothing works. The Benjamini–Hochberg verdict of the harness stays the
    headline.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _finite(x) -> np.ndarray:
    """The finite values of a 1-D array-like as float64, in order."""
    a = np.asarray(x, dtype=np.float64).ravel()
    return a[np.isfinite(a)]


def per_period_rf(rf_annual: float, periods_per_year: float) -> float:
    """Annual risk-free rate compounded down to one period."""
    return (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0


def _excess(r, periods_per_year: float, rf_annual: float) -> np.ndarray:
    return _finite(r) - per_period_rf(rf_annual, periods_per_year)


def sharpe(r, periods_per_year: float, rf_annual: float = 0.0) -> float:
    """
    Annualised Sharpe ratio: mean / sd (ddof = 1) of ``r - per_period_rf``
    times ``sqrt(periods_per_year)``. NaN with < 2 finite values or no variance.
    """
    x = _excess(r, periods_per_year, rf_annual)
    if x.size < 2 or x.max() == x.min():
        return math.nan
    sd = float(x.std(ddof=1))
    if not sd > 0.0:
        return math.nan
    return float(x.mean() / sd * math.sqrt(periods_per_year))


def sortino(r, periods_per_year: float, rf_annual: float = 0.0) -> float:
    """
    Annualised Sortino ratio: mean excess return / downside deviation times
    ``sqrt(periods_per_year)``. The target is the risk-free rate, so excess
    = ``r - per_period_rf``, and the downside deviation is
    ``sqrt(mean(min(excess, 0) ** 2))`` over **all** finite periods (not only the
    losing ones). NaN with < 2 finite values or no period below the target.
    """
    x = _excess(r, periods_per_year, rf_annual)
    if x.size < 2:
        return math.nan
    dd = math.sqrt(float(np.mean(np.minimum(x, 0.0) ** 2)))
    if not dd > 0.0:
        return math.nan
    return float(x.mean() / dd * math.sqrt(periods_per_year))


def drawdown_stats(r) -> dict[str, float]:
    """
    Drawdown of the compounded curve ``cumprod(1 + r)``, which starts at 1.

    ``max_drawdown``     worst ``curve / running_peak - 1`` (≤ 0; 0 if it never falls)
    ``max_dd_duration``  periods from the last time the curve stood at the peak
                         before the worst drawdown until it first gets back to
                         that peak, or to the end of the series if it never does
    ``max_dd_recovered`` True if it got back to that peak
    ``longest_underwater`` longest run of consecutive periods below the running peak

    All four are NaN with < 2 finite values.
    """
    x = _finite(r)
    if x.size < 2:
        return {"max_drawdown": math.nan, "max_dd_duration": math.nan,
                "max_dd_recovered": math.nan, "longest_underwater": math.nan}
    curve = np.cumprod(np.r_[1.0, 1.0 + x])
    dd = curve / np.maximum.accumulate(curve) - 1.0
    trough = int(np.argmin(dd))
    max_dd = float(dd[trough])
    if max_dd >= 0.0:
        duration, recovered = 0, True
    else:
        at_peak = np.flatnonzero(dd[:trough + 1] == 0.0)
        peak = int(at_peak[-1])
        after = np.flatnonzero(dd[trough + 1:] == 0.0)
        recovered = after.size > 0
        end = trough + 1 + int(after[0]) if recovered else len(curve) - 1
        duration = end - peak
    under = dd < 0.0
    longest = run = 0
    for flag in under:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return {"max_drawdown": max_dd, "max_dd_duration": duration,
            "max_dd_recovered": recovered, "longest_underwater": longest}


def calmar(cagr: float, max_drawdown: float) -> float:
    """``cagr / abs(max_drawdown)``; NaN when there is no drawdown or an input is not finite."""
    if not (math.isfinite(cagr) and math.isfinite(max_drawdown)) or max_drawdown == 0.0:
        return math.nan
    return float(cagr / abs(max_drawdown))


def var_cvar(r, level: float = 0.95) -> tuple[float, float]:
    """
    Historical value-at-risk and expected shortfall of the period returns, as
    positive losses. ``q = quantile(r, 1 - level)`` (numpy's default linear
    interpolation), VaR = ``-q``, CVaR = ``-mean(r[r <= q])``. Per period, not
    per day (see the module docstring). ``(NaN, NaN)`` with < 2 finite values.
    """
    x = _finite(r)
    if x.size < 2:
        return math.nan, math.nan
    q = float(np.quantile(x, 1.0 - level))
    return -q, -float(x[x <= q].mean())


def distribution_stats(r) -> dict[str, float]:
    """Mean, median, skew (pandas, bias-adjusted) and excess kurtosis (pandas) of the period returns."""
    x = _finite(r)
    if x.size < 2:
        return dict.fromkeys(("mean", "median", "skew", "excess_kurtosis"), math.nan)
    s = pd.Series(x)
    return {"mean": float(s.mean()), "median": float(s.median()),
            "skew": float(s.skew()), "excess_kurtosis": float(s.kurt())}


def monthly_returns(r: pd.Series) -> pd.Series:
    """
    Period returns compounded into calendar months: ``prod(1 + r) - 1`` per
    month, indexed by the month-end date. The month is that of the index date,
    so a multi-day period counts in the month it **starts** (its index date is
    its start). Non-finite values are skipped; a month with none is absent.
    The index must be date-like.
    """
    idx = pd.DatetimeIndex(r.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    vals = r.to_numpy(dtype=np.float64)
    ok = np.isfinite(vals)
    months = idx[ok].to_period("M")
    grown = pd.Series(1.0 + vals[ok]).groupby(months).prod() - 1.0
    out = pd.Series(grown.to_numpy(), index=grown.index.to_timestamp(how="end").normalize())
    out.index.name = None
    return out


def trade_stats(pnl) -> dict[str, float]:
    """
    Win rate, average win / average loss and profit factor of per-trade (or
    per-period) P&L. ``win_rate`` is the share of finite values > 0 (a zero is
    not a win). ``avg_win_loss_ratio`` = mean of the positives / |mean of the
    negatives|; ``profit_factor`` = sum of the positives / |sum of the negatives|.
    Both ratios are NaN with no losses or no wins; all three with < 2 finite values.
    """
    x = _finite(pnl)
    out = dict.fromkeys(("win_rate", "avg_win_loss_ratio", "profit_factor"), math.nan)
    if x.size < 2:
        return out
    wins, losses = x[x > 0.0], x[x < 0.0]
    out["win_rate"] = float((x > 0.0).mean())
    if wins.size and losses.size:
        out["avg_win_loss_ratio"] = float(wins.mean() / abs(losses.mean()))
        out["profit_factor"] = float(wins.sum() / abs(losses.sum()))
    return out


def herfindahl(weights: pd.DataFrame) -> pd.Series:
    """
    Concentration per row (date × asset weights): the sum of squared weights
    after normalising the finite non-negative weights of the row to sum 1.
    1 / n for n equal weights, 1 for a single holding; NaN for a row without
    a positive finite weight. Not used by the harness yet (backtest engine).
    """
    w = weights.to_numpy(dtype=np.float64)
    w = np.where(np.isfinite(w) & (w >= 0.0), w, 0.0)
    total = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        hhi = np.where(total > 0.0, (w * w).sum(axis=1) / (total * total), np.nan)
    return pd.Series(hhi, index=weights.index, name="HHI")


# ──────────────────────────────────────────────────────────
# Stationary block bootstrap (pre-registered, see the module docstring)
# ──────────────────────────────────────────────────────────

BOOTSTRAP_N: int = 2000
BOOTSTRAP_LEVEL: float = 0.90
BOOTSTRAP_SEED: int = 20261019
BOOTSTRAP_MIN_PERIODS: int = 20


def bootstrap_mean_block(n: int) -> int:
    """Pre-registered mean block length ``max(1, round(n ** (1 / 3)))``; no
    data-driven selection (Politis–White) is used."""
    return max(1, round(n ** (1.0 / 3.0)))


def stationary_bootstrap_indices(n: int, mean_block: float, n_boot: int,
                                 rng: np.random.Generator) -> np.ndarray:
    """
    Index matrix ``(n_boot, n)`` of the stationary bootstrap (Politis & Romano 1994).

    Each row starts at a uniform index; each next index is ``(prev + 1) % n``
    with probability ``1 - 1 / mean_block`` and a fresh uniform draw otherwise
    (blocks wrap around the end). Vectorised: a position's index is the uniform
    draw at the start of its block plus its offset in the block, mod n, which is
    the same recursion.
    """
    fresh_draw = rng.integers(0, n, size=(n_boot, n), dtype=np.int64)
    cont = rng.random((n_boot, n)) < 1.0 - 1.0 / mean_block
    cont[:, 0] = False
    pos = np.arange(n, dtype=np.int64)
    start = np.where(cont, 0, pos)
    np.maximum.accumulate(start, axis=1, out=start)      # block start of each position
    idx = np.take_along_axis(fresh_draw, start, axis=1)
    idx += pos
    idx -= start                                         # first index + offset in block, < 2n
    idx[idx >= n] -= n                                   # circular wrap
    return idx


def bootstrap_intervals(columns: np.ndarray, stat_fns: dict[str, callable], *,
                        mean_block: float | None = None, n_boot: int = BOOTSTRAP_N,
                        level: float = BOOTSTRAP_LEVEL, seed: int = BOOTSTRAP_SEED,
                        min_periods: int = BOOTSTRAP_MIN_PERIODS) -> dict[str, dict[str, float]]:
    """``bootstrap_interval`` for several statistics on the **same** resamples
    (one draw of indices, identical to calling ``bootstrap_interval`` per statistic)."""
    x = np.asarray(columns, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    x = x[np.isfinite(x).all(axis=1)]
    n = x.shape[0]
    mb = float(bootstrap_mean_block(n) if mean_block is None else mean_block)
    empty = {"lo": math.nan, "hi": math.nan, "share_pos": math.nan,
             "n_finite_draws": 0, "mean_block": mb}
    if n < max(min_periods, 1):
        return {k: dict(empty) for k in stat_fns}
    idx = stationary_bootstrap_indices(n, mb, n_boot, np.random.default_rng(seed))
    # (n_boot, n, k) view of a (k, n_boot, n) array, so each column is contiguous
    sample = np.moveaxis(np.take(np.ascontiguousarray(x.T), idx, axis=1), 0, -1)
    alpha = (1.0 - level) / 2.0
    out = {}
    for name, fn in stat_fns.items():
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            draws = np.asarray(fn(sample), dtype=np.float64)
        ok = np.isfinite(draws)
        if not ok.any():
            out[name] = dict(empty)
            continue
        lo, hi = np.nanquantile(np.where(ok, draws, np.nan), [alpha, 1.0 - alpha])
        out[name] = {"lo": float(lo), "hi": float(hi),
                     "share_pos": float((draws[ok] > 0.0).mean()),
                     "n_finite_draws": int(ok.sum()), "mean_block": mb}
    return out


def bootstrap_interval(columns: np.ndarray, stat_fn, *, mean_block: float | None = None,
                       n_boot: int = BOOTSTRAP_N, level: float = BOOTSTRAP_LEVEL,
                       seed: int = BOOTSTRAP_SEED,
                       min_periods: int = BOOTSTRAP_MIN_PERIODS) -> dict[str, float]:
    """
    Percentile interval of ``stat_fn`` under the stationary block bootstrap.

    ``columns`` ``(n, k)``: k aligned per-period series, resampled **jointly**
    (the same indices for every column). Rows with any non-finite value are
    left out through a boolean mask. ``stat_fn`` gets the resampled array
    ``(n_boot, n, k)`` and returns ``(n_boot,)``. Returns ``lo`` / ``hi``
    (``np.nanquantile`` at ``(1 ± level) / 2``), ``share_pos`` (share of finite
    draws > 0), ``n_finite_draws`` and ``mean_block``
    (``bootstrap_mean_block(n)`` of the finite rows when None). Fewer than
    ``min_periods`` finite rows or no finite draw gives NaN for lo / hi /
    share_pos; it never raises.
    """
    return bootstrap_intervals(columns, {"stat": stat_fn}, mean_block=mean_block,
                               n_boot=n_boot, level=level, seed=seed,
                               min_periods=min_periods)["stat"]


def _ratio(x: np.ndarray, scale: float) -> np.ndarray:
    """mean / sd (ddof 1) along axis 1 times ``scale``; NaN where the row is constant."""
    sd = x.std(axis=1, ddof=1)
    flat = (x.max(axis=1) == x.min(axis=1)) | ~(sd > 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(flat, np.nan, x.mean(axis=1) / np.where(flat, 1.0, sd) * scale)


def boot_info_ratio(scale: float):
    """Information ratio per draw: column 0 = net, column 1 = universe;
    mean / sd (ddof 1) of ``net - universe`` times ``scale`` (as ``metrics.held_portfolio``)."""
    def stat(a: np.ndarray) -> np.ndarray:
        return _ratio(a[:, :, 0] - a[:, :, 1], scale)
    return stat


def boot_sharpe(periods_per_year: float, rf_annual: float):
    """Annualised Sharpe of column 0 per draw (as ``sharpe``)."""
    rf = per_period_rf(rf_annual, periods_per_year)
    scale = math.sqrt(periods_per_year)

    def stat(a: np.ndarray) -> np.ndarray:
        return _ratio(a[:, :, 0] - rf, scale)
    return stat


def _boot_cagr_col(a: np.ndarray, col: int, years: float) -> np.ndarray:
    growth = np.prod(1.0 + a[:, :, col], axis=1)
    pos = growth > 0.0
    return np.where(pos, np.where(pos, growth, 1.0) ** (1.0 / years) - 1.0, -1.0)


def boot_cagr(years: float):
    """CAGR of column 0 per draw: ``prod(1 + r) ** (1 / years) - 1``, -1.0 when growth ≤ 0
    (as ``metrics._cagr``)."""
    def stat(a: np.ndarray) -> np.ndarray:
        return _boot_cagr_col(a, 0, years)
    return stat


def boot_excess_cagr(years: float):
    """CAGR of column 0 minus CAGR of column 1 on the same draw."""
    def stat(a: np.ndarray) -> np.ndarray:
        return _boot_cagr_col(a, 0, years) - _boot_cagr_col(a, 1, years)
    return stat
