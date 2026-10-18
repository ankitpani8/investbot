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
    return over the same periods is the benchmark.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from investbot.evaluation.stats import newey_west_tstat, normal_pvalue, spearman

#: Fewer names than this on a date: the date is skipped.
MIN_NAMES: int = 10
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


def quintile_summary(S: pd.DataFrame, Y: pd.DataFrame, horizon: int,
                     min_names: int = MIN_NAMES, cost: float = ROUND_TRIP_COST) -> dict[str, float]:
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
    return out


def signal_metrics(S: pd.DataFrame, Y: pd.DataFrame, horizon: int,
                   min_names: int = MIN_NAMES, cost: float = ROUND_TRIP_COST) -> dict[str, float]:
    """IC summary + quintile summary for one signal, label and horizon on the given rows."""
    return {**ic_summary(daily_ic(S, Y, min_names), horizon),
            **quintile_summary(S, Y, horizon, min_names, cost)}
