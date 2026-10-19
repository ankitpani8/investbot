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
