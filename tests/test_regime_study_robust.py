"""
Tests for the pre-registered robust inference of the regime study (task 350):
``regime_study.robust_summary``, the ``robust`` flags, ``run_study(robust=...)``,
the size check and the runner's section and CSV columns. Synthetic data, no network.
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from investbot.evaluation import regime_study as rs
from investbot.evaluation import stats
from investbot.evaluation.metrics import MIN_NAMES, daily_ic
from investbot.evaluation.performance import BOOTSTRAP_MIN_PERIODS
from investbot.evaluation.stats import bh_adjust, newey_west_tstat
from investbot.evaluation.walkforward import oos_dates
from tests.test_regime_study import pit_reports  # noqa: F401  (fixture)
from tests.test_regime_study_h6 import _old_run_study, small_study  # noqa: F401  (fixture)


def test_pre_registered_constants_are_the_task_340_ones():
    assert rs.ROBUST_BOOT_N is stats.ROBUST_BOOT_N and rs.ROBUST_BOOT_N == 2000
    assert rs.ROBUST_BOOT_SEED is stats.ROBUST_BOOT_SEED and rs.ROBUST_BOOT_SEED == 20261020
    assert rs.ROBUST_MAX_LAG_FRAC is stats.ROBUST_MAX_LAG_FRAC and rs.ROBUST_MAX_LAG_FRAC == 0.25
    assert rs.ROBUST_RHO_CLIP is stats.ROBUST_RHO_CLIP and rs.ROBUST_RHO_CLIP == 0.97
    assert rs.ROBUST_COLUMNS == ("acf1", "lag_auto", "t_auto", "p_auto", "boot_block", "p_boot")
    assert rs.ROBUST_VERDICT_COLUMNS == ("q_auto", "q_boot", "verdict_auto", "verdict_boot")
    assert (rs.SIZE_CHECK_N, rs.SIZE_CHECK_HORIZON, rs.SIZE_CHECK_RUNS) == (690, 20, 200)
    assert (rs.SIZE_CHECK_AR_RHO, rs.SIZE_CHECK_WINDOW, rs.SIZE_CHECK_LEVEL) == (0.99, 250, 0.05)


# ──────────────────────────────────────────────────────────
# robust_summary
# ──────────────────────────────────────────────────────────

def _ar1(phi: float, n: int, rng, mean: float = 0.0) -> np.ndarray:
    x = np.empty(n)
    x[0] = rng.standard_normal() / math.sqrt(1 - phi * phi)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.standard_normal()
    return x + mean


def test_white_noise_lag_never_below_h():
    rng = np.random.default_rng(1)
    for h in (1, 20, 60):
        r = rs.robust_summary(pd.Series(rng.standard_normal(690)), h)
        assert list(r) == list(rs.ROBUST_COLUMNS)
        assert r["lag_auto"] >= h
        assert r["boot_block"] == max(h, r["lag_auto"] + 1, round(690 ** (1 / 3)))
        assert r["p_auto"] == stats.normal_pvalue(r["t_auto"])


def test_short_series_keeps_lag_h_when_the_cap_is_below_h():
    """The H6 chaos − rest case: n ~ 45, H = 20, cap floor(0.25 n) = 11 < H."""
    rng = np.random.default_rng(2)
    x = pd.Series(_ar1(0.5, 45, rng))
    assert math.floor(0.25 * 45) < 20
    assert stats.andrews_bartlett_lag(x, min_lag=20) == 11        # the cap beats min_lag
    r = rs.robust_summary(x, 20)
    assert r["lag_auto"] == 20
    assert r["t_auto"] == newey_west_tstat(x.to_numpy(), 20)
    assert r["boot_block"] == 21


def test_persistent_ar1_gets_a_longer_lag_and_a_smaller_t():
    rng = np.random.default_rng(3)
    x = pd.Series(_ar1(0.95, 690, rng, mean=0.5))
    h = 20
    r = rs.robust_summary(x, h)
    assert r["lag_auto"] > h
    assert abs(r["t_auto"]) < abs(newey_west_tstat(x.to_numpy(), h))
    assert r["acf1"] > 0.8


def test_p_boot_deterministic_and_nan_below_min_periods():
    rng = np.random.default_rng(4)
    x = pd.Series(_ar1(0.5, 300, rng, mean=0.1))
    a, b = rs.robust_summary(x, 5), rs.robust_summary(x.copy(), 5)
    assert a == b and 0 < a["p_boot"] <= 1
    assert a["p_boot"] == stats.block_bootstrap_mean_pvalue(x, a["boot_block"])
    short = pd.Series(rng.standard_normal(BOOTSTRAP_MIN_PERIODS - 1))
    assert math.isnan(rs.robust_summary(short, 5)["p_boot"])
    # exactly at the minimum the bootstrap runs
    ok = pd.Series(rng.standard_normal(BOOTSTRAP_MIN_PERIODS))
    assert np.isfinite(rs.robust_summary(ok, 5)["p_boot"])


def test_non_finite_values_are_left_out_as_in_summary():
    rng = np.random.default_rng(5)
    a = rng.standard_normal(200)
    x = pd.Series(np.r_[np.nan, a[:100], np.inf, np.nan, a[100:]])
    assert rs.robust_summary(x, 5) == rs.robust_summary(pd.Series(a), 5)
    assert rs._summary(x, 5)["n_days"] == 200


# ──────────────────────────────────────────────────────────
# Test functions and run_study on a small synthetic panel
# ──────────────────────────────────────────────────────────

def _run_study_pre_350(features, targets, null_features, null_targets, splits=None,
                       extra_null_panels=None):
    """``regime_study.run_study`` before task 350 (task 330 plus H6), as a dict."""
    out = _old_run_study(features, targets, null_features, null_targets, splits)
    out["h6"] = pd.DataFrame()
    if extra_null_panels is not None:
        out["h6"] = rs.null_excess_tests(features, targets,
                                         [(null_features, null_targets), *extra_null_panels],
                                         out["oos"], out["splits"])
    return out


@pytest.fixture(scope="module")
def studies(small_study):  # noqa: F811
    panels, extra = small_study
    timings: dict[str, float] = {}
    return {
        "old": _run_study_pre_350(*panels, extra_null_panels=extra),
        "off": rs.run_study(*panels, extra_null_panels=extra, robust=False),
        "on": rs.run_study(*panels, extra_null_panels=extra, timings=timings),
        "timings": timings,
        "panels": panels, "extra": extra,
    }


def _assert_same(res, old: dict, cols_only: bool = False):
    for k, v in old.items():
        got = getattr(res, k)
        if isinstance(v, pd.DataFrame):
            if cols_only and k in ("tests", "h6"):
                got = got[list(v.columns)]
            pd.testing.assert_frame_equal(got, v, check_exact=True, obj=k)
        elif isinstance(v, pd.DatetimeIndex):
            assert got.equals(v)
        elif k == "splits":
            assert len(got) == len(v) and all(
                a.train.equals(b.train) and a.test.equals(b.test) for a, b in zip(got, v))
        else:
            assert got == v or (k.startswith("hurst") and str(got) == str(v)), k


def test_robust_false_is_the_pre_350_output(studies):
    _assert_same(studies["off"], studies["old"])
    assert not set(rs.ROBUST_COLUMNS) & set(studies["off"].tests.columns)
    assert len(studies["off"].h6) == 6


def test_robust_true_appends_columns_after_the_existing_ones(studies):
    on, old = studies["on"], studies["old"]
    _assert_same(on, old, cols_only=True)
    new = [*rs.ROBUST_COLUMNS, *rs.ROBUST_VERDICT_COLUMNS]
    for k in ("tests", "h6"):
        assert list(getattr(on, k).columns) == [*old[k].columns, *new], k
    # the null sanity tables are untouched
    assert list(on.null_conditional.columns) == list(old["null_conditional"].columns)
    assert list(on.null_rank.columns) == list(old["null_rank"].columns)
    assert studies["timings"]["robust"] > 0.0


def test_robust_columns_use_the_same_series_and_dates(studies):
    on = studies["on"]
    F, Y, _, _ = studies["panels"]
    splits = on.splits
    oos = on.oos
    labels = F["REGIME"].loc[F["REGIME"].index.isin(oos)]
    checked = 0
    for _, row in on.tests.iterrows():
        h, col = int(row["horizon"]), row["target"]
        if row["section"] == "conditional":
            a, b = rs.COMPARISONS[row["name"]]
            y = Y[col].reindex(index=labels.index, columns=labels.columns)
            x = rs.group_difference(labels, y, a, b)
        elif row["section"] == "rank":
            S = F[row["name"]].loc[F[row["name"]].index.isin(oos)]
            x = daily_ic(S, Y[col].reindex(index=S.index, columns=S.columns), MIN_NAMES)
        else:
            continue
        n = int(np.isfinite(x.to_numpy(dtype=np.float64)).sum())
        assert n == row["n_days"]
        r = rs.robust_summary(x, h)
        for c in rs.ROBUST_COLUMNS:
            assert r[c] == row[c] or (np.isnan(r[c]) and np.isnan(row[c])), (row["target"], c)
        checked += 1
    assert checked == 25
    # calibration: rerun with robust and compare the same rows
    cal, _, _ = rs.calibration_tests(F, Y, oos, splits, robust=True)
    got = on.tests[on.tests["section"] == "calibration"].reset_index(drop=True)
    pd.testing.assert_frame_equal(got[list(cal.columns)], cal, check_exact=True,
                                  check_dtype=False)          # span is float after the concat
    # H6: the robust columns are robust_summary of the excess series
    nulls = [(studies["panels"][2], studies["panels"][3]), *studies["extra"]]
    for (keys, x, _), (_, row) in zip(rs.null_excess_series(F, Y, nulls, oos),
                                      on.h6.iterrows()):
        assert row["n_days"] == int(np.isfinite(x.to_numpy()).sum())
        assert rs.robust_summary(x, keys["horizon"])["lag_auto"] == row["lag_auto"]
        assert row["lag_auto"] >= row["horizon"]


def test_q_auto_and_q_boot_are_bh_within_each_family(studies):
    on = studies["on"]
    assert len(on.tests) == 35 and len(on.h6) == 6
    for df in (on.tests, on.h6):
        for rule in ("auto", "boot"):
            q = bh_adjust(df[f"p_{rule}"].to_numpy(dtype=np.float64))
            np.testing.assert_array_equal(df[f"q_{rule}"].to_numpy(), q)
    # not one pooled family of 41
    pooled = bh_adjust(np.r_[on.tests["p_auto"].to_numpy(), on.h6["p_auto"].to_numpy()])
    assert not np.allclose(pooled[-6:], on.h6["q_auto"].to_numpy(), equal_nan=True)


def test_verdict_auto_is_verdict_with_q_auto(studies):
    on = studies["on"]
    for df in (on.tests, on.h6):
        for _, r in df.iterrows():
            for rule in ("auto", "boot"):
                assert r[f"verdict_{rule}"] == rs.verdict({**r.to_dict(), "q": r[f"q_{rule}"]})
    base = {"section": "rank", "group": "H4", "name": "X", "target": "Y", "horizon": 20,
            "expected_sign": 1, "n_days": 500, "mean": 0.1, "t_nw": 5.0, "p": 1e-6,
            "block_share": 1.0, "span": math.nan, "q": 1e-5, "verdict": "supported",
            "acf1": 0.9, "lag_auto": 80, "t_auto": 1.0, "p_auto": 0.3, "boot_block": 81,
            "p_boot": 0.004}
    rows = pd.DataFrame([base, {**base, "mean": -0.1, "verdict": "wrong sign"},
                         {**base, "block_share": 0.5, "verdict": "not supported"}])
    out = rs.add_robust_verdicts(rows)
    assert list(out["verdict_auto"]) == ["not supported"] * 3
    assert list(out["verdict_boot"]) == ["supported", "wrong sign", "not supported"]
    assert list(out["verdict"]) == list(rows["verdict"])           # unchanged


# ──────────────────────────────────────────────────────────
# Size check
# ──────────────────────────────────────────────────────────

def test_size_check_series_are_seeded_and_as_specified():
    a = rs.size_check_series(1, 0)
    np.testing.assert_array_equal(a, rs.size_check_series(1, 0))
    assert a.shape == (rs.SIZE_CHECK_N,) and not np.array_equal(a, rs.size_check_series(1, 1))
    rng = np.random.default_rng((rs.SIZE_CHECK_SEED, 2, 0))
    e, u = rng.standard_normal(rs.SIZE_CHECK_N + 249), rng.standard_normal(rs.SIZE_CHECK_N)
    t = 17
    expected = math.sqrt(250) * e[t:t + 250].mean() + u[t]
    assert rs.size_check_series(2, 0)[t] == pytest.approx(expected, rel=1e-12)
    with pytest.raises(ValueError):
        rs.size_check_series(3, 0)


def test_reduced_size_check_robust_rules_reject_no_more_than_lag_h():
    size = rs.size_check(runs=40, dgps=(1,))
    r = size.iloc[0]
    assert r["runs"] == 40 and r["median_lag_auto"] > rs.SIZE_CHECK_HORIZON
    assert r["reject_auto"] <= r["reject_fixed"], r.to_dict()
    assert r["reject_boot"] <= r["reject_fixed"], r.to_dict()


# ──────────────────────────────────────────────────────────
# Runner: section, CSV order, flag
# ──────────────────────────────────────────────────────────

def _fake_size() -> pd.DataFrame:
    return pd.DataFrame([{"dgp": 1, "name": "AR(1) rho 0.99", "runs": 200, "n": 690,
                          "horizon": 20, "reject_fixed": 0.5, "reject_auto": 0.3,
                          "reject_boot": 0.04, "median_lag_auto": 103.0, "median_acf1": 0.98}])


def test_report_section_and_options(studies):
    from scripts import regime_study as runner

    on = studies["on"]
    info = {"n_assets": 12, "skipped": [], "n_rows": 10800, "first": on.oos[0],
            "last": on.oos[-1], "robust_s": 3.0, "size_check_s": 2.0}
    text = runner.render_report(on, info, date(2026, 10, 20), elapsed_s=1.0, rss_mb=100.0,
                                git_hash="abc1234", end=None, size_check=_fake_size())
    assert "## Robust inference (pre-registered)" in text
    assert text.index("## H6: real minus shuffled null") < text.index("## Robust inference")
    assert text.index("## Robust inference") < text.index("## Options for Ankit")
    assert "robust inference 3 s (size check 2 s of it)" in text
    assert "35/2001" in text and "6/2001" in text and "at the floor" in text
    assert runner.robust_options_paragraph(on) in text
    assert "**The robust rules still over-reject**" in text and "auto on AR(1)" in text
    sup = runner.supported_tests(on.tests)
    for _, x in sup.iterrows():
        assert f"{x['group']} {x['name']} → {x['target']} (H {x['horizon']})" in text
    assert "The H1–H5 tests that are *supported* under the lag-H rule in this run:" in text
    tail = text.split("The H1–H5 tests that are *supported* under the lag-H rule in this run:\n")[1]
    assert tail.lstrip().startswith("None of the") == sup.empty
    off = runner.render_report(studies["off"], info, date(2026, 10, 20), elapsed_s=1.0,
                               rss_mb=100.0, git_hash="abc1234", end=None)
    assert "## Robust inference" not in off


def test_resolution_note_counts_ties_at_the_floor():
    from scripts import regime_study as runner

    floor = 1 / (rs.ROBUST_BOOT_N + 1)
    p = np.r_[floor, floor, floor, np.full(32, 0.5)]
    tests = pd.DataFrame({"p_boot": p, "q_boot": bh_adjust(p)})
    res = rs.StudyResult(splits=[], oos=pd.DatetimeIndex([]), base_rates_real=pd.DataFrame(),
                         base_rates_null=pd.DataFrame(), hurst_real={}, hurst_null={},
                         tests=tests, group_means=pd.DataFrame(),
                         null_conditional=pd.DataFrame(), null_rank=pd.DataFrame(),
                         hit_quintiles=pd.DataFrame(), churn=pd.DataFrame())
    note = runner.resolution_note(res)
    assert tests["q_boot"].min() == pytest.approx(35 / (3 * 2001))
    assert "H1–H5: 3 of 35 at the floor, min q boot 0.006" in note
    assert note.count("\n") == 1


def test_robust_changes_lists_every_disagreement(studies):
    from scripts import regime_study as runner

    on = studies["on"]
    ch = runner.robust_changes(on)
    both = pd.concat([on.tests, on.h6], ignore_index=True)
    n = ((both["verdict_auto"] != both["verdict"]) | (both["verdict_boot"] != both["verdict"])).sum()
    assert len(ch) == n


def test_robust_last_keeps_universe_before_the_new_columns(studies):
    from scripts import regime_study as runner

    on = studies["on"]
    out = runner.robust_last(pd.concat([on.tests.assign(universe="index"),
                                        on.tests.assign(universe="active")], ignore_index=True))
    old = list(studies["old"]["tests"].columns)
    assert list(out.columns) == [*old, "universe", *rs.ROBUST_COLUMNS, *rs.ROBUST_VERDICT_COLUMNS]
    assert list(runner.robust_last(studies["old"]["tests"]).columns) == old


def test_no_robust_flag_is_passed_to_run(monkeypatch):
    from scripts import regime_study as runner

    seen = {}

    class _Res:
        tests = pd.DataFrame({"verdict": ["not supported"]})
        h6 = pd.DataFrame()

    def fake_run(end, **kw):
        seen.update(kw)
        return "r.md", _Res()

    class _Session:
        def close(self):
            pass

    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr("investbot.data.db.SessionLocal", _Session)
    monkeypatch.setattr("investbot.evaluation.loader.has_index_membership", lambda s, c: False)
    runner.main(["--universe", "active", "--no-robust"])
    assert seen["robust"] is False
    runner.main(["--universe", "active"])
    assert seen["robust"] is True


def test_runner_index_report_and_csvs(pit_reports):  # noqa: F811
    """End to end on the in-memory point-in-time DB: section per universe, CSV order."""
    text, res, report = pit_reports["index"]
    sec = text.split("## Robust inference (pre-registered)")[1].split("## Options for Ankit")[0]
    assert "### index universe" in sec and "### active universe" in sec
    assert "Size check" in sec and "reject, auto" in sec and "robust inference" in text
    assert text.index("## Universe effect") < text.index("## Robust inference")
    tests_csv = report.parent / report.name.replace("regime_study_", "regime_study_tests_") \
        .replace(".md", ".csv")
    cols = list(pd.read_csv(tests_csv, nrows=1).columns)
    new = [*rs.ROBUST_COLUMNS, *rs.ROBUST_VERDICT_COLUMNS]
    assert cols[-len(new):] == new and cols[-len(new) - 1] == "universe"
    assert list(res.tests.columns[-len(new):]) == new
    active_text = pit_reports["active"][0]
    assert "## Robust inference (pre-registered)" in active_text
    assert "### index universe" not in active_text


def test_no_dropna_in_the_study_module():
    import inspect

    src = inspect.getsource(rs)
    assert ".dropna(" not in src
