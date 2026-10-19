"""
Tests for investbot/evaluation/performance.py (master plan §9.1 KPIs) and their
wiring into ``metrics.held_portfolio``. Hand-computed fixtures, synthetic data
only, no network or DB.
"""
from __future__ import annotations

import inspect
import math

import numpy as np
import pandas as pd
import pytest

from investbot.evaluation import performance
from investbot.evaluation.metrics import (
    HELD_KEYS,
    RISK_FREE_RATE_ANNUAL,
    held_periods,
    held_portfolio,
)
from investbot.evaluation.performance import (
    calmar,
    distribution_stats,
    drawdown_stats,
    herfindahl,
    monthly_returns,
    per_period_rf,
    sharpe,
    sortino,
    trade_stats,
    var_cvar,
)


# ──────────────────────────────────────────────────────────
# Sharpe, Sortino, Calmar
# ──────────────────────────────────────────────────────────

def test_per_period_rf():
    assert per_period_rf(0.0, 252) == 0.0
    assert per_period_rf(1.05**4 - 1, 4) == pytest.approx(0.05)
    assert per_period_rf(0.065, 1) == pytest.approx(0.065)


def test_sharpe_by_hand():
    r = [0.01, 0.03]               # mean 0.02, sd (ddof=1) = 0.01 * sqrt(2)
    assert sharpe(r, 4) == pytest.approx(0.02 / (0.01 * math.sqrt(2)) * 2)
    # rf 21.550625% a year = exactly 5% per quarter: excess = [-0.04, -0.02]
    assert sharpe(r, 4, rf_annual=1.05**4 - 1) == pytest.approx(-0.03 / (0.01 * math.sqrt(2)) * 2)


def test_sharpe_constant_series_is_nan():
    assert math.isnan(sharpe([0.01] * 10, 252))
    assert math.isnan(sharpe([0.1] * 7, 12, rf_annual=0.05))
    assert math.isnan(sharpe([0.0] * 5, 252))


def test_sharpe_is_zero_when_rf_equals_own_annualised_return():
    r = np.array([0.02, -0.01, 0.03, -0.02, 0.015])
    ppy = 12
    rf_annual = (1.0 + r.mean()) ** ppy - 1.0
    assert sharpe(r, ppy, rf_annual) == pytest.approx(0.0, abs=1e-12)


def test_sortino_by_hand():
    r = [0.02, -0.01, 0.03, -0.02]   # target 0: downside over all 4 periods
    dd = math.sqrt((0.01**2 + 0.02**2) / 4)
    assert sortino(r, 1) == pytest.approx(0.005 / dd)
    assert sortino(r, 4) == pytest.approx(0.005 / dd * 2)
    # target = rf of 5% per period: excess [-0.04, -0.02], downside sqrt(mean([0.0016, 0.0004]))
    assert sortino([0.01, 0.03], 4, 1.05**4 - 1) == pytest.approx(-0.03 / math.sqrt(0.001) * 2)


def test_sortino_all_positive_excess_is_nan():
    assert math.isnan(sortino([0.01, 0.02, 0.03], 252))
    assert math.isnan(sortino([0.05, 0.06], 4, rf_annual=1.01**4 - 1))
    assert not math.isnan(sortino([0.05, 0.06], 4, rf_annual=1.055**4 - 1))


def test_calmar():
    assert calmar(0.20, -0.10) == pytest.approx(2.0)
    assert calmar(-0.05, -0.25) == pytest.approx(-0.2)
    assert math.isnan(calmar(0.20, 0.0))
    assert math.isnan(calmar(math.nan, -0.1))
    assert math.isnan(calmar(0.2, math.nan))


# ──────────────────────────────────────────────────────────
# Drawdown
# ──────────────────────────────────────────────────────────

def test_drawdown_recovers():
    # curve 1, 1.10, 0.88, 0.968, 1.21, 1.089
    out = drawdown_stats([0.10, -0.20, 0.10, 0.25, -0.10])
    assert out["max_drawdown"] == pytest.approx(-0.20)
    assert out["max_dd_duration"] == 3            # peak at 1, back above it at 4
    assert out["max_dd_recovered"] is True
    assert out["longest_underwater"] == 2         # indices 2-3; the last dip is 1 long


def test_drawdown_does_not_recover():
    # curve 1, 1.10, 0.88, 0.924, 0.9702
    out = drawdown_stats([0.10, -0.20, 0.05, 0.05])
    assert out["max_drawdown"] == pytest.approx(-0.20)
    assert out["max_dd_duration"] == 3            # peak at 1 to the end at 4
    assert out["max_dd_recovered"] is False
    assert out["longest_underwater"] == 3


def test_drawdown_worst_is_not_the_first():
    # curve 1, 0.95, 1.0925 (recovers, new peak), 0.98325 (-10%), 1.07..., never back
    out = drawdown_stats([-0.05, 0.15, -0.10, 0.05])
    assert out["max_drawdown"] == pytest.approx(-0.10)
    assert out["max_dd_duration"] == 2            # peak at 2, end at 4
    assert out["max_dd_recovered"] is False
    assert out["longest_underwater"] == 2


def test_drawdown_never_falls_and_starts_at_one():
    out = drawdown_stats([0.01, 0.0, 0.02])
    assert out["max_drawdown"] == 0.0 and out["max_dd_duration"] == 0
    assert out["max_dd_recovered"] is True and out["longest_underwater"] == 0
    # a first-period loss is a drawdown from the starting 1
    first = drawdown_stats([-0.10, 0.05])
    assert first["max_drawdown"] == pytest.approx(-0.10)
    assert first["max_dd_duration"] == 2 and first["max_dd_recovered"] is False


# ──────────────────────────────────────────────────────────
# VaR / CVaR, distribution
# ──────────────────────────────────────────────────────────

def test_var_cvar_against_numpy_quantile():
    r = np.arange(-10, 10) / 100.0                 # 20 values
    var, cvar = var_cvar(r, 0.95)
    q = np.quantile(r, 0.05)
    assert q == pytest.approx(-0.0905)
    assert var == pytest.approx(-q) == pytest.approx(0.0905)
    assert cvar == pytest.approx(-r[r <= q].mean()) == pytest.approx(0.10)
    var80, cvar80 = var_cvar(r, 0.80)
    q80 = np.quantile(r, 0.20)
    assert var80 == pytest.approx(-q80)
    assert cvar80 == pytest.approx(-r[r <= q80].mean())
    assert cvar80 >= var80


def test_var_cvar_shuffled_input_is_the_same():
    r = np.random.default_rng(3).normal(0, 0.02, 200)
    assert var_cvar(r) == var_cvar(r[::-1].copy())


def test_distribution_stats():
    x = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    out = distribution_stats(x)
    n, m, s = len(x), x.mean(), x.std(ddof=1)
    z = (x - m) / s
    assert out["mean"] == pytest.approx(4.0) and out["median"] == pytest.approx(3.0)
    assert out["skew"] == pytest.approx(n / ((n - 1) * (n - 2)) * (z**3).sum())
    kurt = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * (z**4).sum()
            - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    assert out["excess_kurtosis"] == pytest.approx(kurt)
    assert out["skew"] > 0
    sym = distribution_stats([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert sym["skew"] == pytest.approx(0.0) and sym["excess_kurtosis"] < 0


# ──────────────────────────────────────────────────────────
# Monthly returns
# ──────────────────────────────────────────────────────────

def test_monthly_returns_compound_across_a_month_boundary():
    idx = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02", "2024-03-04"])
    r = pd.Series([0.10, 0.10, -0.05, 0.02, 0.03], index=idx)
    m = monthly_returns(r)
    assert list(m.index) == list(pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"]))
    np.testing.assert_allclose(m.to_numpy(), [1.1 * 1.1 - 1, 0.95 * 1.02 - 1, 0.03])


def test_monthly_returns_period_counts_in_the_month_it_starts():
    # a 20-bar period starting 2024-01-29 is January's, though it ends in February
    r = pd.Series([0.07, 0.01], index=pd.to_datetime(["2024-01-29", "2024-02-26"]))
    m = monthly_returns(r)
    np.testing.assert_allclose(m.to_numpy(), [0.07, 0.01])


def test_monthly_returns_skips_non_finite_and_empty_months():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-01", "2024-03-01"])
    m = monthly_returns(pd.Series([0.1, np.nan, np.nan, 0.02], index=idx))
    assert list(m.index) == list(pd.to_datetime(["2024-01-31", "2024-03-31"]))
    np.testing.assert_allclose(m.to_numpy(), [0.1, 0.02])
    assert monthly_returns(pd.Series([], dtype=float, index=pd.DatetimeIndex([]))).empty


def test_monthly_returns_completed_months_ignore_later_data():
    idx = pd.bdate_range("2024-01-01", periods=60)
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 60), index=idx)
    full, cut = monthly_returns(r), monthly_returns(r.iloc[:40])
    assert cut.index[-2] == full.index[len(cut) - 2]
    np.testing.assert_allclose(cut.iloc[:-1].to_numpy(), full.iloc[:len(cut) - 1].to_numpy())


# ──────────────────────────────────────────────────────────
# Trade stats, Herfindahl
# ──────────────────────────────────────────────────────────

def test_trade_stats_with_zeros():
    out = trade_stats([100.0, -20.0, 0.0, 50.0, -40.0, -30.0])
    assert out["win_rate"] == pytest.approx(2 / 6)            # the zero is not a win
    assert out["avg_win_loss_ratio"] == pytest.approx(75.0 / 30.0)
    assert out["profit_factor"] == pytest.approx(150.0 / 90.0)


def test_trade_stats_no_losses_and_no_wins():
    no_loss = trade_stats([1.0, 2.0, 0.0])
    assert no_loss["win_rate"] == pytest.approx(2 / 3)
    assert math.isnan(no_loss["avg_win_loss_ratio"]) and math.isnan(no_loss["profit_factor"])
    no_win = trade_stats([-1.0, -2.0, 0.0])
    assert no_win["win_rate"] == 0.0
    assert math.isnan(no_win["avg_win_loss_ratio"]) and math.isnan(no_win["profit_factor"])


def test_herfindahl():
    w = pd.DataFrame([[0.25, 0.25, 0.25, 0.25],
                      [2.0, 2.0, np.nan, -1.0],      # NaN and negative left out -> (0.5, 0.5)
                      [3.0, 1.0, 0.0, 0.0],
                      [np.nan, np.nan, np.nan, np.nan],
                      [0.0, 0.0, 0.0, 0.0],
                      [0.0, 5.0, 0.0, 0.0]],
                     index=pd.bdate_range("2024-01-01", periods=6), columns=list("ABCD"))
    h = herfindahl(w)
    assert h.index.equals(w.index)
    assert h.iloc[0] == pytest.approx(1 / 4)
    assert h.iloc[1] == pytest.approx(0.5)
    assert h.iloc[2] == pytest.approx(10 / 16)
    assert math.isnan(h.iloc[3]) and math.isnan(h.iloc[4])
    assert h.iloc[5] == pytest.approx(1.0)
    for n in (2, 5, 50):
        assert herfindahl(pd.DataFrame(np.ones((1, n)))).iloc[0] == pytest.approx(1 / n)


# ──────────────────────────────────────────────────────────
# NaN handling, purity
# ──────────────────────────────────────────────────────────

_FUNCS = [
    lambda r: sharpe(r, 252, 0.065), lambda r: sortino(r, 252, 0.065),
    lambda r: list(drawdown_stats(r).values()), lambda r: list(var_cvar(r)),
    lambda r: list(distribution_stats(r).values()), lambda r: list(trade_stats(r).values()),
]


@pytest.mark.parametrize("bad", [[], [math.nan], [0.01], [0.01, math.nan, math.inf],
                                 [math.nan, math.nan, math.nan], np.array([]),
                                 pd.Series([math.nan, 0.02])])
def test_fewer_than_two_finite_values_give_nan(bad):
    for fn in _FUNCS:
        vals = np.atleast_1d(np.array(fn(bad), dtype=float))
        assert np.isnan(vals).all()


def test_non_finite_values_are_ignored():
    clean = np.array([0.02, -0.01, 0.03, -0.02, 0.015, -0.005])
    dirty = np.array([np.nan, 0.02, -0.01, np.inf, 0.03, -0.02, np.nan, 0.015, -0.005, -np.inf])
    for fn in _FUNCS:
        np.testing.assert_allclose(np.array(fn(dirty), dtype=float),
                                   np.array(fn(clean), dtype=float))
    ser = pd.Series(dirty, index=pd.bdate_range("2024-01-01", periods=len(dirty)))
    assert sharpe(ser, 12) == pytest.approx(sharpe(clean, 12))


def test_kpis_read_only_the_periods_given():
    # trailing NaN padding (periods not yet known) leaves every KPI unchanged
    r = np.random.default_rng(5).normal(0.001, 0.02, 40)
    padded = np.r_[r, np.full(10, np.nan)]
    for fn in _FUNCS:
        np.testing.assert_array_equal(np.array(fn(r), dtype=float),
                                      np.array(fn(padded), dtype=float))


def test_no_dropna_in_the_module():
    assert "dropna" not in inspect.getsource(performance)


# ──────────────────────────────────────────────────────────
# Wiring into metrics.held_portfolio
# ──────────────────────────────────────────────────────────

OLD_KEYS = ("n_periods", "turnover", "cagr_gross", "cagr_net", "cagr_universe",
            "vol_ann_net", "max_drawdown_net", "info_ratio_net")


def _pinned_fixture():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2024-01-01", periods=120)
    cols = [f"T{i:02d}" for i in range(8)]
    members = pd.DataFrame(rng.random((120, 8)) < 0.4, index=idx, columns=cols)
    y = pd.DataFrame(rng.normal(0.002, 0.03, (120, 8)), index=idx, columns=cols)
    y = y.mask(rng.random((120, 8)) < 0.03)
    return members, y


def test_held_keys_extend_the_old_ones_in_order():
    assert HELD_KEYS[:8] == OLD_KEYS
    assert HELD_KEYS[8:19] == ("sharpe_net", "sharpe_universe", "sortino_net", "calmar_net",
                             "max_dd_duration_days_net", "max_dd_recovered_net",
                             "max_drawdown_universe", "var95_net", "cvar95_net",
                             "period_win_rate_net", "excess_win_rate_net")
    assert RISK_FREE_RATE_ANNUAL == 0.065


def test_preexisting_held_values_are_unchanged():
    # values computed with metrics.held_portfolio before the KPI keys were added
    members, y = _pinned_fixture()
    out = held_portfolio(members, y, 5)
    pinned = {
        "n_periods": 24,
        "turnover": 0.4971014492753623,
        "cagr_gross": 0.1767677237394618,
        "cagr_net": 0.08261616595027022,
        "cagr_universe": 0.14646765083356406,
        "vol_ann_net": 0.14688508481688695,
        "max_drawdown_net": -0.07940704767606732,
        "info_ratio_net": -0.43858604000691415,
    }
    assert out["n_periods"] == pinned["n_periods"]
    for k in OLD_KEYS[1:]:
        assert out[k] == pytest.approx(pinned[k], rel=1e-12, abs=0), k
    assert list(out) == list(HELD_KEYS)


def test_held_max_drawdown_matches_performance_drawdown_stats():
    members, y = _pinned_fixture()
    out = held_portfolio(members, y, 5)
    net = held_periods(members, y, 5)["net"].to_numpy()
    assert out["max_drawdown_net"] == drawdown_stats(net)["max_drawdown"]


def test_held_kpis_by_hand():
    m = pd.DataFrame([[1, 0], [0, 1], [0, 1], [1, 1]], columns=["A", "B"],
                     index=pd.bdate_range("2024-01-01", periods=4)).astype(bool)
    y = pd.DataFrame([[0.10, 0.00], [9.0, 9.0], [0.05, -0.20], [9.0, 9.0]],
                     columns=["A", "B"], index=m.index)
    out = held_portfolio(m, y, horizon=2, cost=0.01)
    net, univ = np.array([0.09, -0.21]), np.array([0.05, -0.075])
    ppy = 126.0
    rf = RISK_FREE_RATE_ANNUAL
    assert out["sharpe_net"] == pytest.approx(sharpe(net, ppy, rf))
    assert out["sharpe_universe"] == pytest.approx(sharpe(univ, ppy, rf))
    ex = net - ((1 + rf) ** (1 / ppy) - 1)
    assert out["sortino_net"] == pytest.approx(
        ex.mean() / math.sqrt(np.mean(np.minimum(ex, 0) ** 2)) * math.sqrt(ppy))
    assert out["calmar_net"] == pytest.approx(out["cagr_net"] / 0.21)
    assert out["max_dd_duration_days_net"] == 2          # 1 period of h = 2 bars, not recovered
    assert out["max_dd_recovered_net"] == 0.0
    assert out["max_drawdown_universe"] == pytest.approx(-0.075)
    assert out["var95_net"] == pytest.approx(0.195)      # -0.21 + 0.05 * 0.30 = -0.195
    assert out["cvar95_net"] == pytest.approx(0.21)
    assert out["period_win_rate_net"] == pytest.approx(0.5)
    assert out["excess_win_rate_net"] == pytest.approx(0.5)   # net - univ = [0.04, -0.135]


def test_held_kpis_are_nan_without_two_periods():
    m = pd.DataFrame([[True, False]], columns=["A", "B"], index=pd.bdate_range("2024-01-01", periods=1))
    y = pd.DataFrame([[0.02, 0.0]], columns=["A", "B"], index=m.index)
    out = held_portfolio(m, y, horizon=1)
    assert out["n_periods"] == 1 and math.isfinite(out["max_drawdown_net"])
    for k in HELD_KEYS[8:]:
        assert math.isnan(out[k]), k
    empty = held_portfolio(m.iloc[:0], y.iloc[:0], horizon=1)
    assert empty["n_periods"] == 0 and all(math.isnan(empty[k]) for k in HELD_KEYS[1:])


def test_held_kpis_truncation_invariance():
    # a period's return depends only on its own rows, so the periods of a truncated
    # input are a prefix of the periods of the full input, and the KPIs read only those
    members, y = _pinned_fixture()
    full = held_periods(members, y, 5)
    cut = held_periods(members.iloc[:62], y.iloc[:62], 5)
    pd.testing.assert_frame_equal(cut, full.iloc[:len(cut)])
