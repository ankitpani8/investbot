"""
Autocorrelation-robust IC inference (task 340): ``stats.acf``, ``stats.andrews_bartlett_lag``,
``stats.block_bootstrap_mean_pvalue``, ``metrics.ic_robust`` and the robust columns of
``evaluate()``. No network, no DB.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from investbot.evaluation import stats
from investbot.evaluation.evaluate import (
    BH_LABELS,
    INVERTED_SUBSET,
    PROMISING_Q,
    PROMISING_SIGN_SHARE,
    ROBUST_TIMING_KEY,
    ROBUST_VERDICT_COLUMNS,
    evaluate,
)
from investbot.evaluation.metrics import (
    ROBUST_ACF_LAGS,
    ROBUST_BOOT_N,
    ROBUST_BOOT_SEED,
    ROBUST_KEYS,
    ROBUST_MAX_LAG_FRAC,
    ROBUST_RHO_CLIP,
    SURVIVORSHIP_KEYS,
    ic_robust,
    ic_summary,
)
from investbot.evaluation.regime_study import _lag1_autocorr
from tests.test_evaluation_universe import synthetic_panel


def _ar1(rho: float, n: int, rng: np.random.Generator, burn: int = 200) -> np.ndarray:
    e = rng.standard_normal(n + burn)
    x = np.empty(n + burn)
    x[0] = e[0] / math.sqrt(1.0 - rho * rho)
    for i in range(1, n + burn):
        x[i] = rho * x[i - 1] + e[i]
    return x[burn:]


# ──────────────────────────────────────────────────────────
# Pre-registered constants
# ──────────────────────────────────────────────────────────

def test_preregistered_constants():
    assert ROBUST_ACF_LAGS == (1, 5, 20)
    assert ROBUST_BOOT_N == 2000 and ROBUST_BOOT_SEED == 20261020
    assert ROBUST_MAX_LAG_FRAC == 0.25 and ROBUST_RHO_CLIP == 0.97
    assert ROBUST_KEYS == ("ic_acf1", "ic_acf5", "ic_acf20", "ic_lag_auto", "ic_t_auto",
                           "ic_p_auto", "ic_boot_block", "ic_p_boot")


# ──────────────────────────────────────────────────────────
# acf
# ──────────────────────────────────────────────────────────

def test_acf_hand_example_and_nan_tolerance():
    # d = [-2, -1, 0, 1, 2], sum d^2 = 10
    got = stats.acf([1.0, np.nan, 2.0, 3.0, np.inf, 4.0, 5.0], (0, 1, 2, 3, 4, 5))
    assert got[0] == pytest.approx(1.0)
    assert got[1] == pytest.approx(0.4)       # (2 + 0 + 0 + 2) / 10
    assert got[2] == pytest.approx(-0.1)      # (0 - 1 + 0) / 10
    assert got[3] == pytest.approx(-0.4)      # (-2 - 2) / 10
    assert math.isnan(got[4]) and math.isnan(got[5])     # l >= n - 1
    assert math.isnan(stats.acf(np.ones(10), (1,))[1])   # constant
    with pytest.raises(ValueError):
        stats.acf([1.0, 2.0, 3.0], (-1,))


def test_acf_matches_regime_study_lag1():
    rng = np.random.default_rng(11)
    rows = np.vstack([_ar1(r, 120, rng) for r in (0.0, 0.3, -0.4, 0.8)])
    ref = _lag1_autocorr(rows)
    for row, want in zip(rows, ref):
        assert stats.acf(row, (1,))[1] == pytest.approx(want, rel=1e-12, abs=1e-15)


# ──────────────────────────────────────────────────────────
# andrews_bartlett_lag
# ──────────────────────────────────────────────────────────

def test_andrews_lag_white_noise_is_min_lag():
    x = np.random.default_rng(700).standard_normal(700)
    # |rho| ~ 1/sqrt(n) for white noise, so ceil(S) - 1 is a small number (0 to 2) and any
    # fixed lag h - 1 >= 2 wins; at min_lag 0 the rule itself gives about 1 (documented).
    for min_lag in (4, 19):
        assert stats.andrews_bartlett_lag(x, min_lag=min_lag) == min_lag
    assert stats.andrews_bartlett_lag(x) <= 2


def test_andrews_lag_hand_value_and_growth_with_rho():
    rng = np.random.default_rng(2026)
    lags = []
    for rho in (0.3, 0.6, 0.9):
        x = _ar1(rho, 700, rng)
        r = min(max(stats.acf(x, (1,))[1], -ROBUST_RHO_CLIP), ROBUST_RHO_CLIP)
        alpha = 4 * r * r / ((1 - r) ** 2 * (1 + r) ** 2)
        want = math.ceil(1.1447 * (alpha * 700) ** (1 / 3)) - 1
        lag = stats.andrews_bartlett_lag(x)
        assert lag == want
        lags.append(lag)
    assert lags[0] < lags[1] < lags[2]
    assert 5 <= lags[0] <= 10 and 35 <= lags[2] <= 60     # theory: about 7, 15 and 45


def test_andrews_lag_cap_and_clip():
    rng = np.random.default_rng(5)
    x = _ar1(0.95, 40, rng)
    assert stats.andrews_bartlett_lag(x) == 10                         # floor(0.25 * 40)
    assert stats.andrews_bartlett_lag(x, max_lag_frac=0.1) == 4
    assert stats.andrews_bartlett_lag(x, min_lag=30) == 10             # the cap wins
    # a straight line has rho close to 1: clipped to 0.97, so the lag stays finite
    trend = np.arange(700, dtype=float)
    a = 4 * 0.97 ** 2 / ((0.03) ** 2 * (1.97) ** 2)
    assert stats.andrews_bartlett_lag(trend) == min(math.ceil(1.1447 * (a * 700) ** (1 / 3)) - 1,
                                                    175)
    # undefined rho: min_lag
    assert stats.andrews_bartlett_lag([1.0, 1.0, 1.0, 1.0], min_lag=0) == 0
    assert stats.andrews_bartlett_lag(np.ones(100), min_lag=4) == 4


# ──────────────────────────────────────────────────────────
# block_bootstrap_mean_pvalue
# ──────────────────────────────────────────────────────────

def test_block_bootstrap_pvalue_deterministic_and_small_for_large_mean():
    rng = np.random.default_rng(9)
    x = _ar1(0.5, 500, rng)
    p1 = stats.block_bootstrap_mean_pvalue(x, 8)
    assert p1 == stats.block_bootstrap_mean_pvalue(x.copy(), 8)
    assert 0.0 < p1 <= 1.0
    assert stats.block_bootstrap_mean_pvalue(x + 1.0, 8) < 0.01
    # never 0: the smallest value is 1 / (1 + n_boot)
    assert stats.block_bootstrap_mean_pvalue(x + 100.0, 8) == pytest.approx(1 / (1 + ROBUST_BOOT_N))
    # NaN-tolerant, and NaN below BOOTSTRAP_MIN_PERIODS finite values
    with_nan = np.r_[x[:250], np.nan, x[250:]]
    assert stats.block_bootstrap_mean_pvalue(with_nan, 8) == p1
    assert math.isnan(stats.block_bootstrap_mean_pvalue(x[:19], 3))
    assert not math.isnan(stats.block_bootstrap_mean_pvalue(x[:20], 3))


# ──────────────────────────────────────────────────────────
# ic_robust
# ──────────────────────────────────────────────────────────

def test_ic_robust_keys_and_definitions():
    rng = np.random.default_rng(21)
    x = _ar1(0.4, 650, rng) * 0.05
    ic = pd.Series(np.r_[np.nan, x, np.nan])
    for h in (1, 5, 20):
        out = ic_robust(ic, h)
        assert tuple(out) == ROBUST_KEYS
        a = x
        acfs = stats.acf(a, (1, 5, 20))
        assert (out["ic_acf1"], out["ic_acf5"], out["ic_acf20"]) == (acfs[1], acfs[5], acfs[20])
        lag = stats.andrews_bartlett_lag(a, min_lag=h - 1)
        assert out["ic_lag_auto"] == lag >= h - 1
        assert out["ic_t_auto"] == stats.newey_west_tstat(a, lag)
        assert out["ic_p_auto"] == stats.normal_pvalue(out["ic_t_auto"])
        assert out["ic_boot_block"] == max(h, lag + 1, round(650 ** (1 / 3)))
        assert out["ic_p_boot"] == stats.block_bootstrap_mean_pvalue(a, out["ic_boot_block"])
    short = ic_robust(pd.Series([0.1, -0.2, 0.05]), 1)
    assert math.isnan(short["ic_p_boot"]) and math.isnan(short["ic_acf20"])


# ──────────────────────────────────────────────────────────
# Size check (seeds fixed once; observed rates are in the assertion messages)
# ──────────────────────────────────────────────────────────

N_RUNS = 200
N_OBS = 700


def _rejection_rates(series: list[np.ndarray]) -> dict[str, float]:
    naive = auto = boot = 0
    for x in series:
        s = pd.Series(x)
        naive += ic_summary(s, 1)["ic_p"] < 0.05           # lag h - 1 = 0: a plain t-test
        r = ic_robust(s, 1)
        auto += r["ic_p_auto"] < 0.05
        boot += r["ic_p_boot"] < 0.05
    n = len(series)
    return {"naive": naive / n, "auto": auto / n, "boot": boot / n}


def test_size_ar1_robust_pvalues_hold_nominal_level():
    rng = np.random.default_rng(340)
    rates = _rejection_rates([_ar1(0.5, N_OBS, rng) for _ in range(N_RUNS)])
    # theory for the naive t: P(|Z| > 1.96 / sqrt(3)) ~ 0.26
    assert rates["naive"] > 0.15, rates
    assert rates["auto"] <= 0.12, rates
    assert rates["boot"] <= 0.12, rates


def test_size_iid_all_three_near_five_percent():
    rng = np.random.default_rng(341)
    rates = _rejection_rates([rng.standard_normal(N_OBS) for _ in range(N_RUNS)])
    # binomial sd of a 5% rate over 200 runs is 1.5 pp; tolerance [1%, 10%] is about ±3 sd
    for k, v in rates.items():
        assert 0.01 <= v <= 0.10, (k, rates)


# ──────────────────────────────────────────────────────────
# evaluate(): robust columns
# ──────────────────────────────────────────────────────────

HORIZONS = (1, 5)


@pytest.fixture(scope="module")
def tables():
    f, y = synthetic_panel()
    timings: dict[str, float] = {}
    with_robust, _ = evaluate(f, y, HORIZONS, timings=timings)
    without, _ = evaluate(f, y, HORIZONS, robust=False)
    return with_robust, without, timings


NEW_COLUMNS = (*ROBUST_KEYS, *(c for c, _ in ROBUST_VERDICT_COLUMNS))


def test_existing_columns_identical_with_robust_step_disabled(tables):
    t, base, timings = tables
    old = [c for c in t.columns if c not in (*NEW_COLUMNS, *SURVIVORSHIP_KEYS)]
    # new columns are appended after every existing column (the task-420 survivorship split last)
    assert list(t.columns) == [*old, *NEW_COLUMNS, *SURVIVORSHIP_KEYS]
    pd.testing.assert_frame_equal(t[old], base[old], check_exact=True)
    assert base[list(ROBUST_KEYS) + ["ic_q_auto", "ic_q_boot"]].isna().all().all()
    assert not base["promising_auto"].any() and not base["promising_boot"].any()
    assert timings[ROBUST_TIMING_KEY] > 0.0


def test_robust_columns_only_on_bh_rows(tables):
    t, _, _ = tables
    bh = t["label"].isin(BH_LABELS) & (t["subset"] == "oos")
    assert bh.sum() == 2 * t.loc[bh, "signal"].nunique() * len(HORIZONS)
    assert np.isfinite(t.loc[bh, list(ROBUST_KEYS)].to_numpy(dtype=float)).all()
    assert t.loc[~bh, list(ROBUST_KEYS) + ["ic_q_auto", "ic_q_boot"]].isna().all().all()
    assert not t.loc[~bh, "promising_auto"].any() and not t.loc[~bh, "promising_boot"].any()
    assert t["promising_auto"].dtype == bool and t["promising_boot"].dtype == bool


def test_robust_q_and_promising_follow_bh_within_each_family(tables):
    t, _, _ = tables
    for label in BH_LABELS:
        fam = t[(t["label"] == label) & (t["subset"] == "oos")]
        for kind in ("auto", "boot"):
            q = stats.bh_adjust(fam[f"ic_p_{kind}"].to_numpy())
            np.testing.assert_array_equal(fam[f"ic_q_{kind}"].to_numpy(), q)
            want = (q < PROMISING_Q) & (fam["block_sign_share"].to_numpy() >= PROMISING_SIGN_SHARE)
            np.testing.assert_array_equal(fam[f"promising_{kind}"].to_numpy(), want)
        # the robust lag never drops below the fixed lag h - 1
        assert (fam["ic_lag_auto"] >= fam["horizon"] - 1).all()


def test_robust_values_match_the_row_ic_series(tables):
    """The BH row's robust stats come from the same daily IC series as its ``ic_p``."""
    from investbot.evaluation.evaluate import evaluation_calendar, signal_matrix
    from investbot.evaluation.metrics import daily_ic
    from investbot.evaluation.walkforward import oos_dates, walk_forward_splits
    t, _, _ = tables
    f, y = synthetic_panel()
    cal = evaluation_calendar(f)
    oos = oos_dates(walk_forward_splits(cal, embargo=max(HORIZONS)))
    tickers = sorted(f.index.get_level_values("ticker").unique())
    S = signal_matrix(f, "TECH_SCORE", cal, tickers, None)
    for h in HORIZONS:
        Y = y[f"FWD_TR_RET_{h}"].unstack("ticker").reindex(index=cal, columns=tickers)
        ic = daily_ic(S.loc[oos], Y.loc[oos])
        row = t[(t["signal"] == "TECH_SCORE") & (t["horizon"] == h) & (t["label"] == "open")
                & (t["subset"] == "oos")].iloc[0]
        assert row["ic_p"] == ic_summary(ic, h)["ic_p"]
        want = ic_robust(ic, h)
        for k in ROBUST_KEYS:
            assert row[k] == want[k], k


def test_inverted_rows_get_no_robust_values():
    # the panel of tests/test_evaluation_inverted.py: NEG predicts negatively
    f, y = synthetic_panel(n_dates=700, n_names=15, seed=3)
    rng = np.random.default_rng(11)
    f = f.assign(NEG=-y["FWD_RET_1"].to_numpy() + rng.normal(0, 0.002, len(f)))
    t, _ = evaluate(f, y, HORIZONS, signals=["NEG", "TECH_SCORE"])
    inv = t["subset"] == INVERTED_SUBSET
    assert inv.any()
    assert t.loc[inv, list(ROBUST_KEYS) + ["ic_q_auto", "ic_q_boot"]].isna().all().all()
    assert not t.loc[inv, "promising_auto"].any() and not t.loc[inv, "promising_boot"].any()


# ──────────────────────────────────────────────────────────
# Report section
# ──────────────────────────────────────────────────────────

def test_robust_section_and_its_place_in_the_report(tables):
    from datetime import date

    from investbot.evaluation.report import RunInfo, render_report, robust_rows, robust_section
    from investbot.evaluation.walkforward import walk_forward_splits
    t, base, _ = tables
    info = RunInfo(15, pd.Timestamp("2020-01-01"), pd.Timestamp("2022-09-01"), HORIZONS, None,
                   robust_s=1.25, robust_n_tests=8)
    text = robust_section(t, info)
    assert text.startswith("## IC inference robustness (pre-registered)")
    assert "**The verdict rule is unchanged**" in text
    assert "Runtime of the robust step: 1.2 s for 8 tests" in text
    m = len(t[(t["label"] == "open") & (t["subset"] == "oos")])
    assert f"smallest attainable q boot is {m} / {ROBUST_BOOT_N + 1}" in text
    head = t[(t["label"] == "open") & (t["subset"] == "oos")]
    assert f"Headline family (next-open, primary return): {len(head)} tests" in text
    rows = robust_rows(head)
    assert set(rows.loc[rows["signal"] == "TECH_SCORE", "horizon"]) == set(HORIZONS)
    assert all(f"| TECH_SCORE | {h} |" in text for h in HORIZONS)
    # the price table is printed only if that family's verdicts change
    price = t[(t["label"] == "price") & (t["subset"] == "oos")]
    changed = bool((price["promising"] != price["promising_auto"]).any()
                   or (price["promising"] != price["promising_boot"]).any())
    assert ("Price-only comparison family: every test" in text) == changed
    assert "did not run" in robust_section(base, info)
    f, _ = synthetic_panel()
    from investbot.evaluation.evaluate import evaluation_calendar
    splits = walk_forward_splits(evaluation_calendar(f), embargo=max(HORIZONS))
    md = render_report(t, splits, info, date(2026, 10, 20))
    v, r, top = (md.index("## Verdict"), md.index("## IC inference robustness"),
                 md.index("## Top "))
    assert v < r < top
