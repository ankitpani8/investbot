"""
Stationary block bootstrap in ``investbot/evaluation/performance.py`` and its
intervals in ``metrics.held_portfolio``. Synthetic data only, no DB.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from investbot.evaluation.metrics import (
    HELD_KEYS,
    RISK_FREE_RATE_ANNUAL,
    TRADING_DAYS,
    held_periods,
    held_portfolio,
)
from investbot.evaluation.performance import (
    BOOTSTRAP_LEVEL,
    BOOTSTRAP_MIN_PERIODS,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    boot_cagr,
    boot_calmar,
    boot_excess_cagr,
    boot_info_ratio,
    boot_sharpe,
    boot_sortino,
    bootstrap_interval,
    bootstrap_intervals,
    bootstrap_mean_block,
    calmar,
    drawdown_stats,
    sortino,
    stationary_bootstrap_indices,
)

# task 300 keys (bit-identical across task 310)
NEW_KEYS = ("boot_mean_block", "ir_net_lo", "ir_net_hi", "ir_net_share_pos",
            "sharpe_net_lo", "sharpe_net_hi", "cagr_net_lo", "cagr_net_hi",
            "excess_cagr_lo", "excess_cagr_hi")


# task 310 keys, appended after NEW_KEYS
SORT_CALM_KEYS = ("sortino_net_lo", "sortino_net_hi", "calmar_net_lo", "calmar_net_hi")


def _mean0(a: np.ndarray) -> np.ndarray:
    return a[:, :, 0].mean(axis=1)


def _consecutive(idx: np.ndarray, n: int) -> np.ndarray:
    """True where a step continues the block (circular: n - 1 -> 0 counts)."""
    return idx[:, 1:] == (idx[:, :-1] + 1) % n


# ──────────────────────────────────────────────────────────
# Constants and block rule
# ──────────────────────────────────────────────────────────

def test_preregistered_constants():
    assert (BOOTSTRAP_N, BOOTSTRAP_LEVEL, BOOTSTRAP_SEED, BOOTSTRAP_MIN_PERIODS) == (
        2000, 0.90, 20261019, 20)


@pytest.mark.parametrize("n, block", [(0, 1), (1, 1), (8, 2), (34, 3), (138, 5), (690, 9)])
def test_bootstrap_mean_block(n, block):
    assert bootstrap_mean_block(n) == block


# ──────────────────────────────────────────────────────────
# Indices
# ──────────────────────────────────────────────────────────

def test_indices_shape_dtype_range():
    idx = stationary_bootstrap_indices(37, 3.0, 50, np.random.default_rng(1))
    assert idx.shape == (50, 37)
    assert np.issubdtype(idx.dtype, np.integer)
    assert idx.min() >= 0 and idx.max() < 37


def test_indices_seed_reproducible_and_seed_dependent():
    a = stationary_bootstrap_indices(100, 4.0, 20, np.random.default_rng(5))
    b = stationary_bootstrap_indices(100, 4.0, 20, np.random.default_rng(5))
    c = stationary_bootstrap_indices(100, 4.0, 20, np.random.default_rng(6))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_mean_block_one_forces_no_continuation():
    # every index is a fresh uniform draw, so a step equals prev + 1 only by chance (1 / n)
    n = 1000
    idx = stationary_bootstrap_indices(n, 1.0, 200, np.random.default_rng(2))
    share = _consecutive(idx, n).mean()
    assert share < 5.0 / n


def test_mean_run_length_matches_mean_block():
    n = 10_000
    idx = stationary_bootstrap_indices(n, 10.0, 4, np.random.default_rng(3))
    for row in idx:
        breaks = int((~_consecutive(row[None, :], n)).sum())
        mean_run = n / (breaks + 1)
        assert mean_run == pytest.approx(10.0, rel=0.10)


def test_wrap_around_counts_as_consecutive():
    # with a huge mean block every row is one block, which must wrap from n - 1 to 0
    n = 5
    idx = stationary_bootstrap_indices(n, 1e12, 30, np.random.default_rng(4))
    assert _consecutive(idx, n).all()
    wrapped = (idx[:, :-1] == n - 1) & (idx[:, 1:] == 0)
    assert wrapped.any()


def test_vectorised_indices_follow_the_recursion():
    # re-derive each row from the same uniform draws with the explicit recursion
    n, n_boot, mb = 40, 25, 3.0
    idx = stationary_bootstrap_indices(n, mb, n_boot, np.random.default_rng(8))
    rng = np.random.default_rng(8)
    fresh = rng.integers(0, n, size=(n_boot, n))
    cont = rng.random((n_boot, n)) < 1.0 - 1.0 / mb
    ref = np.empty_like(idx)
    ref[:, 0] = fresh[:, 0]
    for t in range(1, n):
        ref[:, t] = np.where(cont[:, t], (ref[:, t - 1] + 1) % n, fresh[:, t])
    np.testing.assert_array_equal(idx, ref)


# ──────────────────────────────────────────────────────────
# bootstrap_interval
# ──────────────────────────────────────────────────────────

def test_joint_resampling_keeps_columns_paired():
    x = np.random.default_rng(9).normal(size=100)
    cols = np.column_stack([x, x])
    seen = []

    def stat(a):
        d = (a[:, :, 0] - a[:, :, 1]).mean(axis=1)
        seen.append(d)
        return d

    res = bootstrap_interval(cols, stat, n_boot=300)
    assert np.all(seen[0] == 0.0)
    assert res["lo"] == 0.0 and res["hi"] == 0.0 and res["share_pos"] == 0.0


def test_interval_keys_and_defaults():
    x = np.random.default_rng(10).normal(0.01, 0.02, size=125)
    res = bootstrap_interval(x[:, None], _mean0)
    assert set(res) == {"lo", "hi", "share_pos", "n_finite_draws", "mean_block"}
    assert res["mean_block"] == 5.0                     # round(125 ** (1/3))
    assert res["n_finite_draws"] == BOOTSTRAP_N
    assert res["lo"] < x.mean() < res["hi"]
    again = bootstrap_interval(x[:, None], _mean0)
    assert again == res                                 # same seed, same interval


def test_bootstrap_intervals_shares_draws_with_single_calls():
    rng = np.random.default_rng(11)
    cols = rng.normal(0.01, 0.03, size=(60, 2))
    fns = {"ir": boot_info_ratio(3.0), "cagr": boot_cagr(2.0)}
    both = bootstrap_intervals(cols, fns)
    for name, fn in fns.items():
        assert both[name] == bootstrap_interval(cols, fn)


def test_cached_indices_equal_a_fresh_draw():
    # the memoised index matrix is the one stationary_bootstrap_indices draws from the seed
    rng = np.random.default_rng(13)
    x = rng.normal(0.0, 1.0, size=(45, 1))
    for _ in range(2):                                  # second call hits the cache
        res = bootstrap_interval(x, _mean0, n_boot=400, seed=99)
        idx = stationary_bootstrap_indices(45, 4.0, 400, np.random.default_rng(99))
        draws = x[idx, 0].mean(axis=1)
        lo, hi = np.nanquantile(draws, [0.05, 0.95])
        # same draws; only the summation order of the strided view differs (last bits)
        assert res["lo"] == pytest.approx(lo, rel=1e-12)
        assert res["hi"] == pytest.approx(hi, rel=1e-12)
        assert res["share_pos"] == pytest.approx((draws > 0).mean(), abs=1.0 / 400)


def test_coverage_of_the_mean_on_iid_normal():
    hits = 0
    for i in range(200):
        x = np.random.default_rng(1000 + i).normal(0.0, 1.0, size=250)
        res = bootstrap_interval(x[:, None], _mean0, n_boot=500, seed=i)
        hits += res["lo"] <= 0.0 <= res["hi"]
    assert 0.80 <= hits / 200 <= 0.97


def test_too_few_finite_rows_gives_nan():
    x = np.full((30, 2), np.nan)
    x[:BOOTSTRAP_MIN_PERIODS - 1] = 0.01
    res = bootstrap_interval(x, _mean0)
    assert math.isnan(res["lo"]) and math.isnan(res["hi"]) and math.isnan(res["share_pos"])
    assert res["n_finite_draws"] == 0
    empty = bootstrap_interval(np.empty((0, 2)), _mean0)
    assert math.isnan(empty["lo"])


def test_constant_series_ir_is_nan_and_does_not_raise():
    cols = np.column_stack([np.full(50, 0.01), np.full(50, 0.004)])
    res = bootstrap_interval(cols, boot_info_ratio(2.0))
    assert res["n_finite_draws"] == 0
    assert math.isnan(res["lo"]) and math.isnan(res["hi"]) and math.isnan(res["share_pos"])
    sh = bootstrap_interval(cols, boot_sharpe(12.0, 0.0))
    assert math.isnan(sh["lo"])


def test_non_finite_rows_are_ignored():
    rng = np.random.default_rng(12)
    clean = rng.normal(0.01, 0.02, size=(80, 2))
    dirty = clean.copy()
    dirty = np.insert(dirty, [3, 40, 80], [[np.nan, 0.1], [0.2, np.inf], [np.nan, np.nan]], axis=0)
    fn = boot_excess_cagr(1.5)
    assert bootstrap_interval(dirty, fn) == bootstrap_interval(clean, fn)


# ──────────────────────────────────────────────────────────
# Stat helpers vs the held_portfolio point estimates
# ──────────────────────────────────────────────────────────

def _pinned_fixture():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2024-01-01", periods=120)
    cols = [f"T{i:02d}" for i in range(8)]
    members = pd.DataFrame(rng.random((120, 8)) < 0.4, index=idx, columns=cols)
    y = pd.DataFrame(rng.normal(0.002, 0.03, (120, 8)), index=idx, columns=cols)
    y = y.mask(rng.random((120, 8)) < 0.03)
    return members, y


H = 5


@pytest.fixture(scope="module")
def pinned():
    members, y = _pinned_fixture()
    out = held_portfolio(members, y, H)
    p = held_periods(members, y, H)
    cols = np.column_stack([p["net"].to_numpy(), p["universe"].to_numpy()])
    return out, cols


def test_identity_resample_matches_point_estimates(pinned):
    out, cols = pinned
    a = cols[None, :, :]
    n = cols.shape[0]
    years = n * H / TRADING_DAYS
    per_year = TRADING_DAYS / H
    assert out["n_periods"] == n >= BOOTSTRAP_MIN_PERIODS
    checks = {
        "info_ratio_net": boot_info_ratio(math.sqrt(per_year))(a)[0],
        "sharpe_net": boot_sharpe(per_year, RISK_FREE_RATE_ANNUAL)(a)[0],
        "cagr_net": boot_cagr(years)(a)[0],
    }
    for k, v in checks.items():
        assert v == pytest.approx(out[k], abs=1e-12), k
    assert boot_excess_cagr(years)(a)[0] == pytest.approx(
        out["cagr_net"] - out["cagr_universe"], abs=1e-12)


def test_boot_cagr_wiped_out_is_minus_one():
    a = np.array([[[0.5, 0.0], [-1.2, 0.1], [0.3, 0.0]]])
    assert boot_cagr(1.0)(a)[0] == -1.0
    assert boot_excess_cagr(1.0)(a)[0] == pytest.approx(-1.0 - (1.1 ** 1.0 - 1.0))


def test_held_interval_contains_the_point(pinned):
    out, _ = pinned
    assert out["ir_net_lo"] <= out["info_ratio_net"] <= out["ir_net_hi"]
    assert out["sharpe_net_lo"] <= out["sharpe_net"] <= out["sharpe_net_hi"]
    assert out["cagr_net_lo"] <= out["cagr_net"] <= out["cagr_net_hi"]
    excess = out["cagr_net"] - out["cagr_universe"]
    assert out["excess_cagr_lo"] <= excess <= out["excess_cagr_hi"]
    assert 0.0 <= out["ir_net_share_pos"] <= 1.0
    assert out["boot_mean_block"] == bootstrap_mean_block(out["n_periods"])


def test_held_intervals_match_bootstrap_interval(pinned):
    out, cols = pinned
    years = cols.shape[0] * H / TRADING_DAYS
    ref = bootstrap_interval(cols, boot_excess_cagr(years))
    assert (out["excess_cagr_lo"], out["excess_cagr_hi"]) == (ref["lo"], ref["hi"])


def test_held_intervals_are_reproducible():
    members, y = _pinned_fixture()
    a = held_portfolio(members, y, H)
    b = held_portfolio(members, y, H)
    assert all(a[k] == b[k] for k in NEW_KEYS)


def test_new_held_keys_are_appended():
    assert HELD_KEYS[-len(NEW_KEYS) - len(SORT_CALM_KEYS):] == NEW_KEYS + SORT_CALM_KEYS
    assert len(HELD_KEYS) == 19 + len(NEW_KEYS) + len(SORT_CALM_KEYS)


def test_held_intervals_nan_below_min_periods():
    members, y = _pinned_fixture()
    out = held_portfolio(members.iloc[:60], y.iloc[:60], H)     # 12 periods
    assert 2 <= out["n_periods"] < BOOTSTRAP_MIN_PERIODS
    assert math.isfinite(out["info_ratio_net"])
    for k in NEW_KEYS:
        assert math.isnan(out[k]), k


def test_held_intervals_read_only_the_given_periods():
    # trailing rows that add no held period (no member has a finite label) change nothing
    members, y = _pinned_fixture()
    extra = pd.bdate_range(y.index[-1] + pd.offsets.BDay(), periods=10)
    m2 = pd.concat([members, pd.DataFrame(False, index=extra, columns=members.columns)])
    y2 = pd.concat([y, pd.DataFrame(np.nan, index=extra, columns=y.columns)])
    a, b = held_portfolio(members, y, H), held_portfolio(m2, y2, H)
    assert a["n_periods"] == b["n_periods"]
    assert all(a[k] == b[k] for k in NEW_KEYS)


# ──────────────────────────────────────────────────────────
# Task 310: Sortino and Calmar intervals, bootstrap switch
# ──────────────────────────────────────────────────────────

def _identity(r: np.ndarray) -> np.ndarray:
    """Index matrix = arange(n) gives the original series as the single draw (1, n, 1)."""
    idx = np.arange(len(r))[None, :]
    return np.moveaxis(np.take(np.asarray(r, dtype=np.float64)[None, :], idx, axis=1), 0, -1)


def test_boot_sortino_identity_matches_sortino(pinned):
    out, cols = pinned
    per_year = TRADING_DAYS / H
    net = cols[:, 0]
    got = boot_sortino(per_year, RISK_FREE_RATE_ANNUAL)(_identity(net))[0]
    assert got == pytest.approx(sortino(net, per_year, RISK_FREE_RATE_ANNUAL), abs=1e-12)
    assert got == pytest.approx(out["sortino_net"], abs=1e-12)


def test_boot_sortino_no_downside_is_nan():
    r = np.full(30, 0.05)
    assert math.isnan(sortino(r, 52.0, 0.0))
    assert math.isnan(boot_sortino(52.0, 0.0)(_identity(r))[0])
    # a draw above the target in every period is NaN, one with a loss is finite
    mixed = np.stack([r, np.r_[-0.01, r[1:]]])[:, :, None]
    got = boot_sortino(52.0, 0.0)(mixed)
    assert math.isnan(got[0]) and math.isfinite(got[1])


def test_boot_sortino_uses_all_periods_in_the_deviation():
    r = np.array([0.04, -0.02, 0.03, 0.01, -0.01, 0.02])
    x = r - per_period_rf_ref(0.06, 12.0)
    expected = x.mean() / math.sqrt(np.mean(np.minimum(x, 0.0) ** 2)) * math.sqrt(12.0)
    assert boot_sortino(12.0, 0.06)(_identity(r))[0] == pytest.approx(expected, abs=1e-12)


def per_period_rf_ref(rf_annual, ppy):
    return (1.0 + rf_annual) ** (1.0 / ppy) - 1.0


def test_boot_calmar_identity_matches_calmar(pinned):
    out, cols = pinned
    net = cols[:, 0]
    years = len(net) * H / TRADING_DAYS
    got = boot_calmar(years)(_identity(net))[0]
    cagr = boot_cagr(years)(_identity(net))[0]
    assert got == pytest.approx(calmar(cagr, drawdown_stats(net)["max_drawdown"]), abs=1e-12)
    assert got == pytest.approx(out["calmar_net"], abs=1e-12)


def test_boot_calmar_no_drawdown_is_nan_and_wipeout_is_finite():
    up = np.linspace(0.001, 0.02, 25)
    assert math.isnan(calmar(0.5, drawdown_stats(up)["max_drawdown"]))
    assert math.isnan(boot_calmar(1.0)(_identity(up))[0])
    wiped = np.array([0.1, -1.2, 0.3])
    got = boot_calmar(1.0)(_identity(wiped))[0]
    cagr = -1.0
    assert got == pytest.approx(calmar(cagr, drawdown_stats(wiped)["max_drawdown"]), abs=1e-12)


def test_boot_calmar_per_draw_equals_loop():
    rng = np.random.default_rng(3)
    a = rng.normal(0.003, 0.03, (40, 25, 2))
    years = 25 * H / TRADING_DAYS
    got = boot_calmar(years)(a)
    for i in range(a.shape[0]):
        cagr = boot_cagr(years)(a[i:i + 1])[0]
        ref = calmar(cagr, drawdown_stats(a[i, :, 0])["max_drawdown"])
        assert (math.isnan(ref) and math.isnan(got[i])) or got[i] == pytest.approx(ref, abs=1e-12)


# values of held_portfolio on _pinned_fixture() at the parent commit (before task 310)
PARENT_INTERVALS = {
    "ir_net_lo": -2.4024904480772427, "ir_net_hi": 2.1223297389730016,
    "ir_net_share_pos": 0.3885,
    "sharpe_net_lo": -1.6958995736803055, "sharpe_net_hi": 2.4896573951733196,
    "cagr_net_lo": -0.19477196846057895, "cagr_net_hi": 0.46098034582479375,
    "excess_cagr_lo": -0.321242482335429, "excess_cagr_hi": 0.24172033152053382,
}


def test_existing_intervals_are_bit_identical_to_the_parent(pinned):
    out, _ = pinned
    for k, v in PARENT_INTERVALS.items():
        assert out[k] == v, k


def test_new_intervals_contain_their_points_and_share_the_draw(pinned):
    out, cols = pinned
    assert out["sortino_net_lo"] <= out["sortino_net"] <= out["sortino_net_hi"]
    assert out["calmar_net_lo"] <= out["calmar_net"] <= out["calmar_net_hi"]
    per_year = TRADING_DAYS / H
    years = cols.shape[0] * H / TRADING_DAYS
    ref = bootstrap_interval(cols, boot_sortino(per_year, RISK_FREE_RATE_ANNUAL))
    assert (out["sortino_net_lo"], out["sortino_net_hi"]) == (ref["lo"], ref["hi"])
    ref = bootstrap_interval(cols, boot_calmar(years))
    assert (out["calmar_net_lo"], out["calmar_net_hi"]) == (ref["lo"], ref["hi"])


def test_bootstrap_false_gives_nan_and_changes_nothing_else(pinned, monkeypatch):
    out, _ = pinned
    members, y = _pinned_fixture()
    import investbot.evaluation.metrics as metrics

    def boom(*a, **k):
        raise AssertionError("resampled with bootstrap=False")
    monkeypatch.setattr(metrics, "bootstrap_intervals", boom)
    off = held_portfolio(members, y, H, bootstrap=False)
    assert tuple(off) == tuple(out)
    boot_keys = NEW_KEYS + SORT_CALM_KEYS
    for k in boot_keys:
        assert math.isnan(off[k]), k
        assert math.isfinite(out[k]), k
    for k in out:
        if k not in boot_keys:
            assert off[k] == out[k] or (math.isnan(off[k]) and math.isnan(out[k])), k


def _panel(seed=4):
    from investbot.evaluation.labels import PRICE_STEMS, TR_STEMS, label_columns
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", "2023-12-29")
    tickers = [f"S{i:02d}" for i in range(20)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    n = len(idx)
    labels = pd.DataFrame({c: rng.normal(0, 0.02, n)
                           for c in label_columns((1, 5), PRICE_STEMS)}, index=idx)
    labels = labels.assign(**{c.replace(p, t, 1): labels[c] for p, t in zip(PRICE_STEMS, TR_STEMS)
                              for c in label_columns((1, 5), (p,))})
    features = pd.DataFrame({
        "TECH_SCORE": rng.normal(size=n),
        "ORACLE": labels["FWD_TR_OO_RET_5"].to_numpy(),
        "REGIME": rng.choice([0.0, 1.0, 2.0], n, p=[0.45, 0.45, 0.10]),
        "REGIME_CONFIDENCE": rng.uniform(size=n),
    }, index=idx)
    return features, labels


def test_evaluate_computes_intervals_only_on_bootstrap_rows():
    from investbot.evaluation import evaluate as ev
    features, labels = _panel()
    table, _ = ev.evaluate(features, labels, (1, 5))
    keys = ["held_" + k for k in NEW_KEYS + SORT_CALM_KEYS]
    wanted = table.apply(lambda r: (r["label"], r["subset"]) in ev.BOOTSTRAP_ROWS, axis=1)
    assert ev.BOOTSTRAP_ROWS == (("open", "oos"),)
    assert wanted.any()
    on = table[wanted & (table["held_n_periods"] >= BOOTSTRAP_MIN_PERIODS)]
    assert len(on) > 0
    # the oracle never has a losing period, so its Sortino / Calmar draws can be NaN
    for k in keys:
        assert on.loc[on["signal"] == "TECH_SCORE", k].notna().all(), k
    for k in ["held_" + k for k in NEW_KEYS]:
        assert on[k].notna().all(), k
    assert table.loc[~wanted, keys].isna().all().all()
    # rows without intervals still carry every other held_* column
    off = table[~wanted & table["subset"].isin(["all", "year:2022"]) & (table["label"] == "open")]
    assert off["held_cagr_net"].notna().any() and off["held_sharpe_net"].notna().any()


def test_evaluate_non_interval_columns_do_not_depend_on_the_switch(monkeypatch):
    from investbot.evaluation import evaluate as ev
    features, labels = _panel()
    base, _ = ev.evaluate(features, labels, (5,))
    monkeypatch.setattr(ev, "BOOTSTRAP_ROWS",
                        (("open", "oos"), ("open", "all"), ("close", "oos"), ("price", "oos")))
    wide, _ = ev.evaluate(features, labels, (5,))
    interval = [c for c in base.columns if c.startswith("held_") and (
        c.endswith(("_lo", "_hi")) or c in ("held_ir_net_share_pos", "held_boot_mean_block"))]
    rest = [c for c in base.columns if c not in interval]
    pd.testing.assert_frame_equal(base[rest], wide[rest])
    wanted = (base["label"] == "open") & (base["subset"] == "oos")
    pd.testing.assert_frame_equal(base.loc[wanted, interval], wide.loc[wanted, interval])
