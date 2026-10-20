"""
Cross-sectional signal metrics on wide (date × ticker) matrices.

Every function takes a signal matrix ``S`` and a label matrix ``Y`` with the
same index (dates) and columns (tickers). On each date only names where both
are finite take part (the joint mask), and a date with fewer than
``MIN_NAMES`` such names is skipped (NaN).

IC
    Daily Spearman rank correlation between S and Y across names (average
    ranks for ties). Summary: mean, std, ICIR = mean / std (daily, not
    annualised), Newey-West t-stat with lag h - 1, two-sided normal p-value,
    hit rate = share of days with IC > 0.

Quintiles
    On each date names are ranked by S (average ranks), quintile
    ``q = ceil(5 * rank / n)``, so tied names always share a bucket. A
    discrete signal (e.g. ±1) fills only some buckets, so "top" and "bottom"
    are the highest and lowest **non-empty** buckets of that date (Q5 and Q1
    for any continuous signal); a date with a single non-empty bucket has no
    spread. Spread = mean Y(top) - mean Y(bottom), Newey-West t with lag h - 1.
    Monotonicity = Spearman of the bucket number vs its time-mean return.

Turnover and costs
    The top-bucket portfolio is rebalanced every h dates (non-overlapping
    holding periods). Turnover = share of the new top bucket that was not in
    the previous one (1 at the first rebalance). Net return per period =
    gross - turnover * ``ROUND_TRIP_COST``: each replaced name costs one round
    trip (sell the old, buy the new) on its weight. The universe equal-weight
    return over the same periods is the benchmark. These ``lo_*`` numbers use
    the open[t+1] -> close[t+h] label, so the overnight gap before the next
    entry is not held (simple annualisation, not compounded).

Held portfolio (``held_*``)
    The same top-bucket portfolio and turnover, but each period's return is
    the open-to-open label ``FWD_OO_RET_h`` (open[t+1] -> open[t+1+h]), so
    consecutive periods tile the timeline and compound into an equity curve:
    CAGR (gross, net, universe), volatility, max drawdown and information
    ratio against the universe, plus the §9.1 risk and trade-quality KPIs from
    ``performance.py`` (Sharpe against ``RISK_FREE_RATE_ANNUAL``, Sortino,
    Calmar, drawdown duration, VaR / CVaR per holding period, win rates) and
    90% stationary block-bootstrap intervals for IR, Sharpe net, CAGR net,
    net − universe CAGR, Sortino net and Calmar net (``performance.bootstrap_intervals``,
    per row, not adjusted for multiple comparisons). The harness asks for them only
    on the rows its report reads (``evaluate.BOOTSTRAP_ROWS``); ``bootstrap=False``
    leaves them NaN and runs no resampling.

Pooled IC (``pooled_*``)
    For thin subsets (e.g. one regime) where few dates reach ``MIN_NAMES``:
    per-date centred ranks scaled to [-0.5, 0.5], pooled over every
    (date, name) row of dates with >= ``POOLED_MIN_NAMES`` names.

Pre-registered (task 340, robust IC inference)
    Fixed before any run on the DB and not to be changed after seeing results:
    the IC autocorrelations reported at lags ``ROBUST_ACF_LAGS`` = (1, 5, 20);
    an automatic Newey-West lag from the Andrews (1991) AR(1) plug-in for the
    Bartlett kernel, never below h - 1, with the AR(1) coefficient clipped to
    ±``ROBUST_RHO_CLIP`` = 0.97 and the lag capped at
    ``floor(ROBUST_MAX_LAG_FRAC * n)`` = floor(0.25 n); and a stationary block
    bootstrap p-value of H0 mean IC = 0 with ``ROBUST_BOOT_N`` = 2000 draws,
    seed ``ROBUST_BOOT_SEED`` = 20261020 and mean block
    max(h, auto lag + 1, round(n^(1/3))). These are a robustness check of the
    BH verdicts next to ``ic_p``; the verdict rule (``ic_q``, ``promising``)
    does not change.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from investbot.evaluation.performance import (
    BOOTSTRAP_MIN_PERIODS,
    boot_cagr,
    boot_calmar,
    boot_excess_cagr,
    boot_info_ratio,
    boot_sharpe,
    boot_sortino,
    bootstrap_intervals,
    calmar,
    drawdown_stats,
    sharpe,
    sortino,
    trade_stats,
    var_cvar,
)
from investbot.evaluation.stats import (
    ROBUST_BOOT_N,
    ROBUST_BOOT_SEED,
    ROBUST_MAX_LAG_FRAC,
    ROBUST_RHO_CLIP,
    acf,
    andrews_bartlett_lag,
    block_bootstrap_mean_pvalue,
    newey_west_se,
    newey_west_tstat,
    normal_pvalue,
    spearman,
)

#: Fewer names than this on a date: the date is skipped.
MIN_NAMES: int = 10
#: Minimum joint names on a date for it to enter the pooled IC.
POOLED_MIN_NAMES: int = 3
N_QUANTILES: int = 5
#: Order size at which the flat ₹20 brokerage is converted into a percentage.
REFERENCE_TRADE_VALUE: float = 100_000.0
#: Market-impact / spread allowance per side for liquid NIFTY50 names
#: (2.5 bp). Not in config or the master plan; see docs/DECISIONS_PENDING.md.
SLIPPAGE_PER_SIDE: float = 0.00025


def statutory_round_trip_cost(trade_value: float = REFERENCE_TRADE_VALUE) -> float:
    """Round-trip STT + stamp + exchange + SEBI + brokerage + GST from the current equity regime."""
    from investbot.trading.tax_rules import regime_params
    from investbot.trading.transaction_costs import round_trip_cost_pct
    return round_trip_cost_pct(trade_value, regime_params("IN_EQUITY_LISTED"))


#: Round-trip cost of one equity delivery trade as a fraction of the traded
#: value: statutory charges from ``investbot/config/tax_regimes.py`` at
#: ``REFERENCE_TRADE_VALUE`` (0.270%) plus slippage on both sides (0.05%),
#: ≈ 0.32%, inside the master plan's §2.8 "~0.25-0.35%".
ROUND_TRIP_COST: float = statutory_round_trip_cost() + 2 * SLIPPAGE_PER_SIDE
TRADING_DAYS: int = 252
#: Annual risk-free rate for Sharpe and Sortino. A constant placeholder for the
#: 91-day T-bill (master plan §9.1), not a dated series; see docs/DECISIONS_PENDING.md.
RISK_FREE_RATE_ANNUAL: float = 0.065
#: Pre-registered (task 340): lags at which the autocorrelation of the per-date IC
#: series is reported (``ic_acf1``, ``ic_acf5``, ``ic_acf20``). Descriptive only.
ROBUST_ACF_LAGS: tuple[int, ...] = (1, 5, 20)
#: The other pre-registered constants (``ROBUST_BOOT_N``, ``ROBUST_BOOT_SEED``,
#: ``ROBUST_MAX_LAG_FRAC``, ``ROBUST_RHO_CLIP``) live next to their helpers in ``stats.py``
#: and are imported here.


def joint_mask(S: pd.DataFrame, Y: pd.DataFrame, min_names: int = MIN_NAMES):
    """(S, Y) with each cell kept only where both are finite and the date has >= min_names."""
    ok = np.isfinite(S.to_numpy(dtype=np.float64)) & np.isfinite(Y.to_numpy(dtype=np.float64))
    ok &= (ok.sum(axis=1) >= min_names)[:, None]
    return S.where(ok), Y.where(ok)


def daily_ic(S: pd.DataFrame, Y: pd.DataFrame, min_names: int = MIN_NAMES) -> pd.Series:
    """Spearman IC per date (NaN where fewer than ``min_names`` joint names or no variance)."""
    s, y = joint_mask(S, Y, min_names)
    rs = s.rank(axis=1, method="average").to_numpy(dtype=np.float64)
    ry = y.rank(axis=1, method="average").to_numpy(dtype=np.float64)
    ok = np.isfinite(rs)
    n = ok.sum(axis=1)
    nn = np.maximum(n, 1)[:, None]
    rs = np.where(ok, rs - np.where(ok, rs, 0.0).sum(axis=1, keepdims=True) / nn, 0.0)
    ry = np.where(ok, ry - np.where(ok, ry, 0.0).sum(axis=1, keepdims=True) / nn, 0.0)
    den = np.sqrt((rs * rs).sum(axis=1) * (ry * ry).sum(axis=1))
    ic = np.full(len(rs), np.nan)
    good = (den > 0) & (n >= min_names)
    ic[good] = (rs * ry).sum(axis=1)[good] / den[good]
    return pd.Series(ic, index=S.index, name="IC")


def ic_summary(ic: pd.Series, horizon: int) -> dict[str, float]:
    a = ic.to_numpy(dtype=np.float64)
    a = a[np.isfinite(a)]
    n = a.size
    mean = float(a.mean()) if n else math.nan
    std = float(a.std(ddof=1)) if n > 1 else math.nan
    t = newey_west_tstat(a, max(horizon - 1, 0))
    return {
        "n_days": n,
        "ic_mean": mean,
        "ic_std": std,
        "icir": mean / std if n > 1 and std > 0 else math.nan,
        "ic_t_nw": t,
        "ic_p": normal_pvalue(t),
        "ic_hit": float((a > 0).mean()) if n else math.nan,
    }


#: Keys of ``ic_robust``, in this order (appended after the existing metrics columns).
ROBUST_KEYS: tuple[str, ...] = (*(f"ic_acf{lag}" for lag in ROBUST_ACF_LAGS), "ic_lag_auto",
                                "ic_t_auto", "ic_p_auto", "ic_boot_block", "ic_p_boot")


def ic_robust(ic: pd.Series, horizon: int) -> dict[str, float]:
    """
    Autocorrelation-robust inference on the mean of a per-date IC series (pre-registered,
    task 340). Non-finite dates are left out, as in ``ic_summary``, which is unchanged.

        ic_acf1/5/20   sample autocorrelation of the IC series at ``ROBUST_ACF_LAGS``
        ic_lag_auto    ``andrews_bartlett_lag(ic, min_lag=h - 1)``: Andrews AR(1) plug-in,
                       never below the fixed lag h - 1, capped at
                       floor(``ROBUST_MAX_LAG_FRAC`` n), rho clipped at ``ROBUST_RHO_CLIP``
        ic_t_auto      Newey-West t of the mean at ``ic_lag_auto``
        ic_p_auto      two-sided normal p-value of ``ic_t_auto``
        ic_boot_block  mean block of the bootstrap: max(h, ic_lag_auto + 1, round(n^(1/3)))
        ic_p_boot      ``block_bootstrap_mean_pvalue`` (``ROBUST_BOOT_N`` draws, seed
                       ``ROBUST_BOOT_SEED``, centred series), NaN below
                       ``BOOTSTRAP_MIN_PERIODS`` dates

    A robustness check next to ``ic_p``, not a new verdict rule.
    """
    a = ic.to_numpy(dtype=np.float64)
    a = a[np.isfinite(a)]
    n = a.size
    out: dict[str, float] = {f"ic_acf{lag}": v for lag, v in acf(a, ROBUST_ACF_LAGS).items()}
    lag = andrews_bartlett_lag(a, min_lag=max(horizon - 1, 0), max_lag_frac=ROBUST_MAX_LAG_FRAC)
    t = newey_west_tstat(a, lag)
    block = max(horizon, lag + 1, round(n ** (1.0 / 3.0)))
    out.update({
        "ic_lag_auto": lag,
        "ic_t_auto": t,
        "ic_p_auto": normal_pvalue(t),
        "ic_boot_block": block,
        "ic_p_boot": block_bootstrap_mean_pvalue(a, block, ROBUST_BOOT_N, ROBUST_BOOT_SEED),
    })
    return {k: out[k] for k in ROBUST_KEYS}


def quantile_buckets(S: pd.DataFrame, Y: pd.DataFrame, min_names: int = MIN_NAMES,
                     n_q: int = N_QUANTILES) -> pd.DataFrame:
    """Bucket 1..n_q per cell (NaN outside the joint mask); ties share a bucket."""
    s, _ = joint_mask(S, Y, min_names)
    rank = s.rank(axis=1, method="average")
    n = s.notna().sum(axis=1)
    q = np.ceil(rank.to_numpy() * n_q / n.to_numpy()[:, None].clip(min=1))
    q = np.clip(q, 1, n_q)
    return pd.DataFrame(q, index=S.index, columns=S.columns)


def bucket_returns(buckets: pd.DataFrame, Y: pd.DataFrame, n_q: int = N_QUANTILES) -> pd.DataFrame:
    """Mean label per bucket and date (columns 1..n_q, NaN where a bucket is empty)."""
    y = Y.to_numpy(dtype=np.float64)
    b = buckets.to_numpy()
    out = np.full((len(Y), n_q), np.nan)
    for k in range(1, n_q + 1):
        m = b == k
        cnt = m.sum(axis=1)
        tot = np.where(m, y, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, k - 1] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return pd.DataFrame(out, index=Y.index, columns=list(range(1, n_q + 1)))


def top_bottom(buckets: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Highest and lowest non-empty bucket per date (NaN on empty dates)."""
    b = buckets.to_numpy()
    with np.errstate(all="ignore"):
        has = np.isfinite(b).any(axis=1)
        top = np.where(has, np.nanmax(np.where(np.isfinite(b), b, -np.inf), axis=1), np.nan)
        bot = np.where(has, np.nanmin(np.where(np.isfinite(b), b, np.inf), axis=1), np.nan)
    return top, bot


def spread_series(buckets: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    """Top-minus-bottom bucket return per date (NaN with < 2 non-empty buckets)."""
    top, bot = top_bottom(buckets)
    r = rets.to_numpy()
    rows = np.arange(len(r))
    ok = np.isfinite(top) & np.isfinite(bot) & (top > bot)
    spread = np.full(len(r), np.nan)
    spread[ok] = r[rows[ok], top[ok].astype(int) - 1] - r[rows[ok], bot[ok].astype(int) - 1]
    return pd.Series(spread, index=buckets.index, name="SPREAD")


def top_membership(buckets: pd.DataFrame) -> pd.DataFrame:
    """Boolean matrix: name is in that date's top non-empty bucket."""
    top, _ = top_bottom(buckets)
    b = buckets.to_numpy()
    with np.errstate(invalid="ignore"):
        m = np.isfinite(top)[:, None] & (b == top[:, None])
    return pd.DataFrame(m, index=buckets.index, columns=buckets.columns)


def rebalance_dates(index: pd.Index, step: int) -> pd.Index:
    return index[::max(step, 1)]


def turnover(members: pd.DataFrame) -> pd.Series:
    """Share of each date's members that were not members on the previous row (1 on the first)."""
    m = members.to_numpy(dtype=bool)
    prev = np.vstack([np.zeros((1, m.shape[1]), dtype=bool), m[:-1]]) if len(m) else m
    size = m.sum(axis=1)
    new = (m & ~prev).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        to = np.where(size > 0, new / np.maximum(size, 1), np.nan)
    return pd.Series(to, index=members.index, name="TURNOVER")


def long_only_costs(members: pd.DataFrame, Y: pd.DataFrame, horizon: int,
                    cost: float = ROUND_TRIP_COST) -> dict[str, float]:
    """Top-bucket long-only portfolio on non-overlapping h-bar periods: gross, net, universe."""
    reb = rebalance_dates(members.index, horizon)
    m = members.loc[reb]
    y = Y.loc[reb].to_numpy(dtype=np.float64)
    mm = m.to_numpy(dtype=bool) & np.isfinite(y)
    cnt = mm.sum(axis=1)
    has = cnt > 0
    if not has.any():
        return {"n_periods": 0, "turnover": math.nan, "gross_ann": math.nan,
                "net_ann": math.nan, "universe_ann": math.nan, "excess_net_ann": math.nan}
    gross = np.where(mm, y, 0.0).sum(axis=1)[has] / cnt[has]
    to = turnover(m[has]).to_numpy()
    net = gross - to * cost
    finite = np.isfinite(y[has])
    univ = np.where(finite, y[has], 0.0).sum(axis=1) / np.maximum(finite.sum(axis=1), 1)
    per_year = TRADING_DAYS / horizon
    return {
        "n_periods": int(has.sum()),
        "turnover": float(np.nanmean(to[1:])) if has.sum() > 1 else math.nan,
        "gross_ann": float(gross.mean() * per_year),
        "net_ann": float(net.mean() * per_year),
        "universe_ann": float(univ.mean() * per_year),
        "excess_net_ann": float((net - univ).mean() * per_year),
    }


HELD_KEYS: tuple[str, ...] = ("n_periods", "turnover", "cagr_gross", "cagr_net", "cagr_universe",
                              "vol_ann_net", "max_drawdown_net", "info_ratio_net",
                              "sharpe_net", "sharpe_universe", "sortino_net", "calmar_net",
                              "max_dd_duration_days_net", "max_dd_recovered_net",
                              "max_drawdown_universe", "var95_net", "cvar95_net",
                              "period_win_rate_net", "excess_win_rate_net",
                              "boot_mean_block", "ir_net_lo", "ir_net_hi", "ir_net_share_pos",
                              "sharpe_net_lo", "sharpe_net_hi", "cagr_net_lo", "cagr_net_hi",
                              "excess_cagr_lo", "excess_cagr_hi",
                              "sortino_net_lo", "sortino_net_hi", "calmar_net_lo", "calmar_net_hi")


def held_periods(members: pd.DataFrame, Y_oo: pd.DataFrame, horizon: int,
                 cost: float = ROUND_TRIP_COST) -> pd.DataFrame:
    """
    Per-period returns of the continuously-held top-bucket portfolio.

    One row per rebalance date (every ``horizon`` rows) that has at least one
    member with a finite ``FWD_OO_RET_h``: ``gross`` (equal-weight mean over
    those members), ``turnover`` (as in ``long_only_costs``), ``net`` =
    gross - turnover * cost, and ``universe`` (equal-weight mean over every
    name with a finite label, cost-free). A rebalance date without members is
    left out, so the curve covers invested periods only.
    """
    reb = rebalance_dates(members.index, horizon)
    m = members.loc[reb]
    y = Y_oo.loc[reb].reindex(columns=members.columns).to_numpy(dtype=np.float64)
    finite = np.isfinite(y)
    mm = m.to_numpy(dtype=bool) & finite
    cnt = mm.sum(axis=1)
    has = cnt > 0
    gross = np.where(mm, y, 0.0).sum(axis=1)[has] / cnt[has]
    to = turnover(m[has]).to_numpy()
    univ = np.where(finite, y, 0.0).sum(axis=1)[has] / np.maximum(finite.sum(axis=1)[has], 1)
    return pd.DataFrame({"gross": gross, "turnover": to, "net": gross - to * cost,
                         "universe": univ}, index=reb[has])


def _cagr(r: np.ndarray, years: float) -> float:
    growth = float(np.prod(1.0 + r))
    if growth <= 0.0:
        return -1.0
    return growth ** (1.0 / years) - 1.0


def held_portfolio(members: pd.DataFrame, Y_oo: pd.DataFrame, horizon: int,
                   cost: float = ROUND_TRIP_COST, bootstrap: bool = True) -> dict[str, float]:
    """
    Compounded top-bucket portfolio held open-to-open between rebalances.

    Period returns come from ``held_periods``. With n periods of h bars,
    years = n * h / 252 and

        cagr_*            prod(1 + r) ** (1 / years) - 1, for gross, net and the universe
        vol_ann_net       sd of the net period returns * sqrt(252 / h)
        max_drawdown_net  worst peak-to-trough of the compounded net curve (starting
                          at 1), as a negative fraction (0 if it never falls)
        info_ratio_net    mean / sd of (net - universe) period returns * sqrt(252 / h)
        turnover          mean turnover after the first rebalance

    The keys after ``info_ratio_net`` come from ``performance.py`` (n >= 2 periods,
    else NaN), with ``periods_per_year = 252 / h`` and ``RISK_FREE_RATE_ANNUAL``:

        sharpe_net / sharpe_universe  annualised Sharpe against the risk-free rate
        sortino_net                   annualised Sortino, target = the risk-free rate
        calmar_net                    cagr_net / |max_drawdown_net|
        max_dd_duration_days_net      periods of the longest-to-recover worst drawdown
                                      (peak to recovery, or to the end) * h bars
        max_dd_recovered_net          1.0 if the worst drawdown got back to its peak, else 0.0
        max_drawdown_universe         as max_drawdown_net, for the universe curve
        var95_net / cvar95_net        95% historical VaR / CVaR of the net period
                                      returns, positive losses **per holding period of h bars**
        period_win_rate_net           share of periods with net > 0
        excess_win_rate_net           share of periods with net - universe > 0

    The IR, vol and CAGR use no risk-free rate: they compare against the
    equal-weight universe over the same periods, not against cash.

    The last keys are stationary block-bootstrap intervals (``performance.py``,
    pre-registered ``BOOTSTRAP_*`` constants, the same seed for every call so
    signals share common random numbers) of the period pairs (net, universe),
    resampled jointly; NaN with fewer than ``BOOTSTRAP_MIN_PERIODS`` periods:

        boot_mean_block                    mean block length in periods
        ir_net_lo / _hi / _share_pos       interval of info_ratio_net, share of draws > 0
        sharpe_net_lo / _hi                interval of sharpe_net
        cagr_net_lo / _hi                  interval of cagr_net
        excess_cagr_lo / _hi               interval of cagr_net - cagr_universe
        sortino_net_lo / _hi               interval of sortino_net
        calmar_net_lo / _hi                interval of calmar_net (NaN draws without a drawdown)

    All six statistics share one draw. With ``bootstrap=False`` these keys stay NaN
    and no resampling runs; every other key is unchanged.
    """
    p = held_periods(members, Y_oo, horizon, cost)
    n = len(p)
    out = dict.fromkeys(HELD_KEYS, math.nan)
    out["n_periods"] = n
    if n == 0:
        return out
    years = n * horizon / TRADING_DAYS
    net = p["net"].to_numpy()
    active = net - p["universe"].to_numpy()
    scale = math.sqrt(TRADING_DAYS / horizon)
    curve = np.cumprod(np.r_[1.0, 1.0 + net])
    out.update({
        "turnover": float(p["turnover"].iloc[1:].mean()) if n > 1 else math.nan,
        "cagr_gross": _cagr(p["gross"].to_numpy(), years),
        "cagr_net": _cagr(net, years),
        "cagr_universe": _cagr(p["universe"].to_numpy(), years),
        "max_drawdown_net": float((curve / np.maximum.accumulate(curve) - 1.0).min()),
    })
    if n > 1:
        out["vol_ann_net"] = float(net.std(ddof=1) * scale)
        sd = float(active.std(ddof=1))
        out["info_ratio_net"] = float(active.mean() / sd * scale) if sd > 0 else math.nan
        univ = p["universe"].to_numpy()
        per_year = TRADING_DAYS / horizon
        dd = drawdown_stats(net)
        var, cvar = var_cvar(net, 0.95)
        out.update({
            "sharpe_net": sharpe(net, per_year, RISK_FREE_RATE_ANNUAL),
            "sharpe_universe": sharpe(univ, per_year, RISK_FREE_RATE_ANNUAL),
            "sortino_net": sortino(net, per_year, RISK_FREE_RATE_ANNUAL),
            "calmar_net": calmar(out["cagr_net"], out["max_drawdown_net"]),
            "max_dd_duration_days_net": dd["max_dd_duration"] * horizon,
            "max_dd_recovered_net": float(dd["max_dd_recovered"]),
            "max_drawdown_universe": drawdown_stats(univ)["max_drawdown"],
            "var95_net": var,
            "cvar95_net": cvar,
            "period_win_rate_net": trade_stats(net)["win_rate"],
            "excess_win_rate_net": trade_stats(active)["win_rate"],
        })
    if bootstrap and n >= BOOTSTRAP_MIN_PERIODS:
        per_year = TRADING_DAYS / horizon
        boot = bootstrap_intervals(
            np.column_stack([net, p["universe"].to_numpy()]),
            {"ir": boot_info_ratio(scale),
             "sharpe": boot_sharpe(per_year, RISK_FREE_RATE_ANNUAL),
             "cagr": boot_cagr(years), "excess": boot_excess_cagr(years),
             "sortino": boot_sortino(per_year, RISK_FREE_RATE_ANNUAL),
             "calmar": boot_calmar(years)})
        out.update({
            "boot_mean_block": boot["ir"]["mean_block"],
            "ir_net_lo": boot["ir"]["lo"], "ir_net_hi": boot["ir"]["hi"],
            "ir_net_share_pos": boot["ir"]["share_pos"],
            "sharpe_net_lo": boot["sharpe"]["lo"], "sharpe_net_hi": boot["sharpe"]["hi"],
            "cagr_net_lo": boot["cagr"]["lo"], "cagr_net_hi": boot["cagr"]["hi"],
            "excess_cagr_lo": boot["excess"]["lo"], "excess_cagr_hi": boot["excess"]["hi"],
            "sortino_net_lo": boot["sortino"]["lo"], "sortino_net_hi": boot["sortino"]["hi"],
            "calmar_net_lo": boot["calmar"]["lo"], "calmar_net_hi": boot["calmar"]["hi"],
        })
    return out


def pooled_ic(S: pd.DataFrame, Y: pd.DataFrame, horizon: int,
              min_names: int = POOLED_MIN_NAMES) -> dict[str, float]:
    """
    Pooled (panel) rank IC over every date with >= ``min_names`` joint names.

    On each date S and Y become average ranks scaled to [-0.5, 0.5]
    ((rank - 1) / (n - 1) - 0.5), demeaned per date: r_s, r_y. Then

        pooled_ic = sum_d sum_i r_s * r_y / sqrt(sum r_s^2 * sum r_y^2)

    over all rows. Dates where S or Y has no cross-sectional variance are left
    out (as in ``daily_ic``). Treating the denominator as fixed, pooled_ic is
    a constant times the mean of the per-date sums a_d = sum_i r_s * r_y, so
    its t-stat is mean(a) / Newey-West se(a) with lag h - 1.
    With the same names and rank dispersion on every date this equals the
    mean daily Spearman IC; otherwise dates with more names weigh more.
    """
    s, y = joint_mask(S, Y, max(min_names, 2))
    rs = s.rank(axis=1, method="average").to_numpy(dtype=np.float64)
    ry = y.rank(axis=1, method="average").to_numpy(dtype=np.float64)
    ok = np.isfinite(rs)
    n = ok.sum(axis=1)
    den = np.maximum(n - 1, 1)[:, None]
    nn = np.maximum(n, 1)[:, None]
    rs = np.where(ok, (rs - 1.0) / den - 0.5, 0.0)
    ry = np.where(ok, (ry - 1.0) / den - 0.5, 0.0)
    rs = np.where(ok, rs - rs.sum(axis=1, keepdims=True) / nn, 0.0)
    ry = np.where(ok, ry - ry.sum(axis=1, keepdims=True) / nn, 0.0)
    a = (rs * ry).sum(axis=1)
    b = (rs * rs).sum(axis=1)
    c = (ry * ry).sum(axis=1)
    use = (n >= max(min_names, 2)) & (b > 0) & (c > 0)
    out = {"pooled_ic": math.nan, "pooled_t_nw": math.nan,
           "pooled_n_obs": int(n[use].sum()), "pooled_n_dates": int(use.sum())}
    if not use.any():
        return out
    a = a[use]
    scale = a.size / math.sqrt(float(b[use].sum()) * float(c[use].sum()))
    out["pooled_ic"] = float(a.sum()) / math.sqrt(float(b[use].sum()) * float(c[use].sum()))
    se = newey_west_se(a, max(horizon - 1, 0)) * scale
    if np.isfinite(se):
        out["pooled_t_nw"] = (out["pooled_ic"] / se if se > 0
                              else math.copysign(math.inf, out["pooled_ic"]))
    return out


def quintile_summary(S: pd.DataFrame, Y: pd.DataFrame, horizon: int,
                     min_names: int = MIN_NAMES, cost: float = ROUND_TRIP_COST,
                     Y_oo: pd.DataFrame | None = None,
                     bootstrap: bool = True) -> dict[str, float]:
    """Quintile means, spread, ``lo_*`` (open→close) and ``held_*`` (open→open, if ``Y_oo``).
    ``bootstrap=False`` leaves the held-portfolio intervals NaN."""
    buckets = quantile_buckets(S, Y, min_names)
    rets = bucket_returns(buckets, Y)
    spread = spread_series(buckets, rets)
    means = rets.mean(axis=0).to_numpy()
    out: dict[str, float] = {f"q{k}_mean": float(means[k - 1]) for k in range(1, N_QUANTILES + 1)}
    sp = spread.to_numpy()
    sp = sp[np.isfinite(sp)]
    out["spread_mean"] = float(sp.mean()) if sp.size else math.nan
    out["spread_t_nw"] = newey_west_tstat(sp, max(horizon - 1, 0))
    out["monotonicity"] = spearman(np.arange(1, N_QUANTILES + 1), means)
    members = top_membership(buckets)
    out.update({f"lo_{k}": v for k, v in long_only_costs(members, Y, horizon, cost).items()})
    out["top_turnover"] = out.pop("lo_turnover")
    held = (held_portfolio(members, Y_oo, horizon, cost, bootstrap) if Y_oo is not None
            else dict.fromkeys(HELD_KEYS, math.nan))
    out.update({f"held_{k}": v for k, v in held.items() if k != "turnover"})
    return out


def signal_metrics(S: pd.DataFrame, Y: pd.DataFrame, horizon: int,
                   min_names: int = MIN_NAMES, cost: float = ROUND_TRIP_COST,
                   Y_oo: pd.DataFrame | None = None,
                   bootstrap: bool = True) -> dict[str, float]:
    """IC summary + quintile summary for one signal, label and horizon on the given rows.

    ``Y_oo`` is the matching ``FWD_OO_RET_h`` matrix for the held portfolio
    (``held_*`` keys are NaN without it). Buckets always come from ``Y``'s joint mask.
    ``bootstrap=False`` leaves the held-portfolio bootstrap intervals NaN (no resampling).
    """
    return {**ic_summary(daily_ic(S, Y, min_names), horizon),
            **quintile_summary(S, Y, horizon, min_names, cost, Y_oo, bootstrap)}
