"""
Tests for the pre-registered H6 null-excess family of the regime study (task 330):
``regime_study.null_excess_tests`` and the runner's extra shuffled-null seeds.
Synthetic data, no network; the runner tests use the in-memory DB of
``tests/test_regime_study.py``.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from investbot.evaluation import regime_study as rs
from investbot.evaluation.metrics import MIN_NAMES, daily_ic
from investbot.evaluation.stats import bh_adjust
from investbot.evaluation.walkforward import oos_dates
from investbot.scoring.technical import regime as R
from tests.test_regime_study import (
    PIT_ASSETS,
    PIT_COMPLETE_FROM,
    _bars,
    _member,
    _pit_bars,
    _pit_session,
)


def test_pre_registered_constants():
    assert rs.NULL_EXCESS_SEEDS == (20261018, 20261019, 20261020, 20261021, 20261022)
    assert rs.NULL_EXCESS_SEEDS[0] == rs.NULL_SEED
    assert rs.NULL_EXCESS_SIGN == +1
    assert rs.NULL_EXCESS_TARGETS == ("FWD_VOLX_20", "FWD_VOLX_60")
    assert len(rs.NULL_EXCESS_TESTS) * len(rs.REGIME_HORIZONS) == 6


# ──────────────────────────────────────────────────────────
# Statistical behaviour: iid returns vs volatility clustering
# ──────────────────────────────────────────────────────────

def _bars_from_returns(r: np.ndarray, rng) -> pd.DataFrame:
    n = len(r)
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.integers(100_000, 1_000_000, n).astype(float)},
                        index=pd.bdate_range("2020-01-01", periods=n, name="date"))


def _garch(n: int, rng, omega=1e-6, a=0.10, b=0.88) -> np.ndarray:
    r, v = np.empty(n), omega / (1 - a - b)
    for i in range(n):
        r[i] = np.sqrt(v) * rng.standard_normal()
        v = omega + a * r[i] ** 2 + b * v
    return r


#: Test-only chaos cut on VOL_RATIO, low enough that >= 3 names are "chaos" on most dates.
_TEST_CHAOS_CUT = 1.05


def _vol_frames(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    The H6 columns without the (slow) Lyapunov estimate: REALVOL_20/250 and VOL_RATIO
    exactly as ``regime.compute_regime``, the volatility part of the chaos score
    (monotone in VOL_RATIO, so the same ranks), REGIME = chaos above a test cut, else
    trend. Targets from ``forward_regime_targets`` with the production REALVOL_250
    normaliser, so the shared-denominator artefact is the real one.
    """
    c = bars["close"].to_numpy(dtype=np.float64)
    ret = pd.Series(np.r_[np.nan, np.diff(np.log(c))])
    vs = ret.rolling(R.VOL_SHORT, min_periods=R.VOL_SHORT).std().to_numpy()
    vl = ret.rolling(R.VOL_LONG, min_periods=R.VOL_LONG).std().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        vr = np.where(vl > 0, vs / vl, np.nan)
    score = 1.0 / (1.0 + np.exp(-R.VOL_WEIGHT * (vr - R.VOL_RATIO_CHAOS) / R.VOL_RATIO_SCALE))
    lab = np.where(np.isfinite(vr), np.where(vr > _TEST_CHAOS_CUT, rs.CHAOS, rs.TREND), np.nan)
    f = pd.DataFrame({"REGIME": lab, "REGIME_SCORE_CHAOS": score, "VOL_RATIO": vr},
                     index=bars.index)
    y = rs.forward_regime_targets(c, realvol=vl)
    return f, pd.DataFrame({k: y[k] for k in rs.NULL_EXCESS_TARGETS}, index=bars.index)


def _vol_panel(kind: str, seed: int, n_names: int = 40, n_bars: int = 3000):
    # ~2500 OOS dates: on ~500 the lag-H Newey-West t over-rejects (REALVOL_250 keeps
    # the per-date statistic persistent for ~250 bars; |t| up to 4 on iid data)
    rng = np.random.default_rng(seed)
    real, nulls = {}, [{} for _ in rs.NULL_EXCESS_SEEDS]
    for k in range(n_names):
        # iid: fat-tailed (Student t, 3 df) like daily stock returns; the mechanical
        # REALVOL_250 artefact needs the tails (on Gaussian iid returns it is ~0)
        r = _garch(n_bars, rng) if kind == "garch" else rng.standard_t(3, n_bars) * 0.008
        b = _bars_from_returns(r, rng)
        real[f"S{k:02d}"] = _vol_frames(b)
        for j, s in enumerate(rs.NULL_EXCESS_SEEDS):
            nulls[j][f"S{k:02d}"] = _vol_frames(rs.shuffled_null_bars(b, seed=s))

    def wide(d):
        return (rs.to_wide({t: f for t, (f, _) in d.items()}),
                rs.to_wide({t: y for t, (_, y) in d.items()}))

    F, Y = wide(real)
    splits = rs.study_splits(F["REGIME"])
    return F, Y, [wide(n) for n in nulls], oos_dates(splits), splits


@pytest.fixture(scope="module")
def iid_h6():
    F, Y, nulls, oos, splits = _vol_panel("iid", seed=1)
    return rs.null_excess_tests(F, Y, nulls, oos, splits), F, Y, nulls, oos, splits


@pytest.fixture(scope="module")
def garch_h6():
    F, Y, nulls, oos, splits = _vol_panel("garch", seed=5)
    return rs.null_excess_tests(F, Y, nulls, oos, splits), F, Y, nulls, oos, splits


def test_iid_returns_have_no_null_excess(iid_h6):
    h6, F, Y, nulls, oos, splits = iid_h6
    assert len(h6) == 6 and (h6["group"] == "H6").all() and (h6["n_seeds"] == 5).all()
    assert (h6["n_days"] > 100).all()                      # the conditional tests have dates too
    assert (h6["t_nw"].abs() < 2.5).all(), h6[["name", "target", "mean", "t_nw"]]
    assert (h6["verdict"] != "supported").all()
    # ...while the raw rank IC is mechanically positive on iid returns (the artefact)
    raw = daily_ic(F["VOL_RATIO"].loc[oos], Y["FWD_VOLX_20"].reindex(index=oos,
                                                                    columns=F["VOL_RATIO"].columns))
    assert raw.mean() > 0.03
    rank = h6[h6["section"] == "rank"]
    assert (rank["null_mean"] > 0.03).all()


def test_volatility_clustering_gives_positive_rank_excess(garch_h6):
    h6 = garch_h6[0]
    rank = h6[h6["section"] == "rank"]
    assert len(rank) == 4
    assert (rank["mean"] > 0).all() and (rank["t_nw"] > 3).all()
    assert (rank["q"] < rs.PROMISING_Q).all() and (rank["verdict"] == "supported").all()


def test_own_bh_family_and_columns(garch_h6):
    h6 = garch_h6[0]
    np.testing.assert_allclose(h6["q"], bh_adjust(h6["p"].to_numpy()), equal_nan=True)
    assert list(h6.columns) == ["section", "group", "name", "target", "horizon",
                                "expected_sign", "n_days", "mean", "t_nw", "p", "block_share",
                                "span", "q", "verdict", "null_mean", "n_seeds"]
    assert (h6["expected_sign"] == 1).all()
    assert h6["horizon"].tolist() == [20, 60, 20, 60, 20, 60]


# ──────────────────────────────────────────────────────────
# Arithmetic: hand computation, NaN dates
# ──────────────────────────────────────────────────────────

def test_one_seed_equals_real_minus_null_per_date(iid_h6):
    _, F, Y, nulls, oos, splits = iid_h6
    nf, ny = nulls[1]
    h6 = rs.null_excess_tests(F, Y, [(nf, ny)], oos, splits).set_index(["name", "target"])
    for col, h in (("FWD_VOLX_20", 20), ("FWD_VOLX_60", 60)):
        lab = F["REGIME"].loc[oos]
        lab_n = nf["REGIME"].loc[oos]
        d_real = rs.group_difference(lab, Y[col].reindex_like(lab), (rs.CHAOS,),
                                     (rs.TREND, rs.RANGE))
        d_null = rs.group_difference(lab_n, ny[col].reindex_like(lab_n), (rs.CHAOS,),
                                     (rs.TREND, rs.RANGE))
        x = (d_real - d_null).to_numpy()
        ok = np.isfinite(x)
        row = h6.loc[("chaos − rest", col)]
        assert row["n_days"] == ok.sum()
        assert row["mean"] == pytest.approx(x[ok].mean(), rel=1e-12)
        assert row["null_mean"] == pytest.approx(d_null.to_numpy()[ok].mean(), rel=1e-12)
        s = rs._summary(pd.Series(x, index=lab.index), h)
        assert row["t_nw"] == pytest.approx(s["t_nw"], rel=1e-12)
        for sig in ("REGIME_SCORE_CHAOS", "VOL_RATIO"):
            S, Sn = F[sig].loc[oos], nf[sig].loc[oos]
            ic = daily_ic(S, Y[col].reindex_like(S), MIN_NAMES)
            icn = daily_ic(Sn, ny[col].reindex_like(Sn), MIN_NAMES)
            x = (ic - icn).to_numpy()
            ok = np.isfinite(x)
            row = h6.loc[(sig, col)]
            assert row["n_days"] == ok.sum()
            assert row["mean"] == pytest.approx(x[ok].mean(), rel=1e-12)
            assert row["block_share"] == rs.block_sign_share(pd.Series(x, index=S.index),
                                                             splits, +1)


_UP = [float(i) for i in range(1, MIN_NAMES + 1)]          # MIN_NAMES names, ranks 1..n
_COLS = [f"N{i}" for i in range(MIN_NAMES)]


def _tiny_panel():
    """3 dates x MIN_NAMES names; hand-checkable ICs and group differences."""
    idx = pd.bdate_range("2024-01-01", periods=3, name="date")

    def frame(rows):
        return pd.DataFrame(rows, idx, _COLS, dtype=float)

    swap = _UP[:-2] + _UP[-1:] + _UP[-2:-1]                  # last two swapped
    feats = {"REGIME": frame([[2.0] * (MIN_NAMES - 1) + [1.0]] * 3),
             "REGIME_SCORE_CHAOS": frame([_UP] * 3),
             "VOL_RATIO": frame([_UP] * 3)}
    targets = {"FWD_VOLX_20": frame([_UP, _UP[::-1], swap]),
               "FWD_VOLX_60": frame([_UP] * 3)}
    return feats, targets


def test_mean_over_finite_seeds_and_all_nan_date_is_skipped():
    from investbot.evaluation.walkforward import Split

    feats, targets = _tiny_panel()
    idx = feats["REGIME"].index
    # null 1: VOL_RATIO reversed (IC -1, +1, NaN); null 2: ascending targets on date 0
    # only (IC +1, NaN, NaN); on date 2 every null is NaN
    n1f = {k: v.copy() for k, v in feats.items()}
    n1f["VOL_RATIO"] = pd.DataFrame([_UP[::-1]] * 3, idx, _COLS)
    n2f = {k: v.copy() for k, v in feats.items()}
    n1t = {k: v.copy() for k, v in targets.items()}
    n2t = {k: v.copy() for k, v in targets.items()}
    n2t["FWD_VOLX_20"] = pd.DataFrame([_UP] * 3, idx, _COLS)
    n2t["FWD_VOLX_20"].iloc[1] = np.nan
    n1f["VOL_RATIO"].iloc[2] = np.nan
    n2t["FWD_VOLX_20"].iloc[2] = np.nan
    nulls = [(n1f, n1t), (n2f, n2t)]
    series = {(k["name"], k["target"]): (x, nm)
              for k, x, nm in rs.null_excess_series(feats, targets, nulls, idx)}
    x, nm = series[("VOL_RATIO", "FWD_VOLX_20")]
    # real IC: +1, -1, 1 - 6*2/(n(n^2-1)); null mean: (-1 + 1) / 2 = 0, then +1 (seed 2
    # NaN), then none: the date is skipped
    np.testing.assert_allclose(nm.to_numpy(), [0.0, 1.0, np.nan], atol=1e-12)
    np.testing.assert_allclose(x.to_numpy(), [1.0, -2.0, np.nan], atol=1e-12)
    splits = [Split(train=idx[:0], test=idx)]
    h6 = rs.null_excess_tests(feats, targets, nulls, idx, splits).set_index(["name", "target"])
    row = h6.loc[("VOL_RATIO", "FWD_VOLX_20")]
    assert row["n_days"] == 2 and row["mean"] == pytest.approx(-0.5)
    assert row["null_mean"] == pytest.approx(0.5)
    assert row["n_seeds"] == 2
    # chaos - rest has 1 "rest" name (< MIN_GROUP_NAMES): no date at all, not an error
    assert h6.loc[("chaos − rest", "FWD_VOLX_20"), "n_days"] == 0
    assert h6.loc[("chaos − rest", "FWD_VOLX_20"), "verdict"] == "not supported"


# ──────────────────────────────────────────────────────────
# run_study: H1-H5 unchanged, h6 only with extra nulls
# ──────────────────────────────────────────────────────────

def _old_run_study(features, targets, null_features, null_targets, splits=None):
    """``regime_study.run_study`` before task 330, verbatim (minus the dataclass)."""
    from investbot.evaluation.stats import bh_adjust as _bh

    if splits is None:
        splits = rs.study_splits(features["REGIME"])
    oos = oos_dates(splits)
    cond, means = rs.conditional_tests(features, targets, oos, splits)
    rank = rs.rank_tests(features, targets, oos, splits)
    calib, quint, churn = rs.calibration_tests(features, targets, oos, splits)
    tests = pd.concat([cond, rank, calib], ignore_index=True)
    tests["q"] = _bh(tests["p"].to_numpy(dtype=np.float64))
    tests["verdict"] = [rs.verdict(r) for _, r in tests.iterrows()]
    null_cond, _ = rs.conditional_tests(null_features, null_targets, oos, splits)
    null_rank = rs.rank_tests(null_features, null_targets, oos, splits)
    return {"splits": splits, "oos": oos,
            "base_rates_real": rs.label_base_rates(features["REGIME"], oos),
            "base_rates_null": rs.label_base_rates(null_features["REGIME"], oos),
            "hurst_real": rs.distribution(features[rs.HURST_COL], oos),
            "hurst_null": rs.distribution(null_features[rs.HURST_COL], oos),
            "tests": tests, "group_means": means, "null_conditional": null_cond,
            "null_rank": null_rank, "hit_quintiles": quint, "churn": churn}


@pytest.fixture(scope="module")
def small_study():
    pytest.importorskip("talib")
    real, null, extra = {}, {}, {}
    for k in range(12):                     # >= MIN_NAMES (10), or there is no OOS
        bars = _bars(900, seed=200 + k, phi=0.1 * (k % 3))
        real[f"S{k}"] = rs.asset_study_frames(bars)
        null[f"S{k}"] = rs.asset_study_frames(rs.shuffled_null_bars(bars))
        f, y = rs.asset_study_frames(rs.shuffled_null_bars(bars, seed=rs.NULL_EXCESS_SEEDS[1]))
        extra[f"S{k}"] = (f[list(rs.NULL_EXCESS_FEATURES)], y[list(rs.NULL_EXCESS_TARGETS)])

    def wide(d):
        return (rs.to_wide({t: f for t, (f, _) in d.items()}),
                rs.to_wide({t: y for t, (_, y) in d.items()}))

    return (*wide(real), *wide(null)), [wide(extra)]


def test_run_study_without_extra_nulls_is_unchanged(small_study):
    panels, extra = small_study
    old = _old_run_study(*panels)
    plain = rs.run_study(*panels, robust=False)          # task 350's columns are tested there
    assert len(plain.oos) > 200 and plain.tests["n_days"].gt(0).sum() >= 20
    with_h6 = rs.run_study(*panels, extra_null_panels=extra, robust=False)
    assert plain.h6.empty
    for res in (plain, with_h6):
        for k, v in old.items():
            got = getattr(res, k)
            if isinstance(v, pd.DataFrame):
                pd.testing.assert_frame_equal(got, v, check_exact=True, obj=k)
            elif isinstance(v, pd.DatetimeIndex):
                assert got.equals(v)
            elif k == "splits":
                assert len(got) == len(v) and all(
                    a.train.equals(b.train) and a.test.equals(b.test) for a, b in zip(got, v))
            else:
                assert got == v or (k.startswith("hurst") and str(got) == str(v)), k
    h6 = with_h6.h6
    assert len(h6) == 6 and (h6["n_seeds"] == 2).all()
    # the H6 real statistic is the H4 one: real = null mean + excess on the H6 dates
    expected = rs.null_excess_tests(panels[0], panels[1], [(panels[2], panels[3]), *extra],
                                    with_h6.oos, with_h6.splits)
    pd.testing.assert_frame_equal(h6, expected, check_exact=True)


def test_h6_rank_real_series_is_the_h4_series(small_study):
    """On dates where the null is finite, real = x + null mean reproduces H4's daily IC."""
    panels, extra = small_study
    F, Y, NF, NY = panels
    splits = rs.study_splits(F["REGIME"])
    oos = oos_dates(splits)
    out = {(k["name"], k["target"]): (x, nm)
           for k, x, nm in rs.null_excess_series(F, Y, [(NF, NY), *extra], oos)}
    S = F["VOL_RATIO"].loc[F["VOL_RATIO"].index.isin(oos)]
    ic = daily_ic(S, Y["FWD_VOLX_20"].reindex(index=S.index, columns=S.columns), MIN_NAMES)
    x, nm = out[("VOL_RATIO", "FWD_VOLX_20")]
    ok = np.isfinite(x.to_numpy())
    assert ok.sum() > 100
    np.testing.assert_allclose((x + nm).to_numpy()[ok], ic.to_numpy()[ok], rtol=1e-12)


# ──────────────────────────────────────────────────────────
# Runner: extra seeds, masking, truncation invariance, report
# ──────────────────────────────────────────────────────────

SEEDS = rs.NULL_EXCESS_SEEDS[1:3]          # two extra seeds keep the tests quick


@pytest.fixture(scope="module")
def pit_h6():
    pytest.importorskip("talib")
    from scripts import regime_study as runner

    s = _pit_session()
    try:
        return runner.build_study_panels(s, universe="index", complete_from=PIT_COMPLETE_FROM,
                                         compare_active=True, null_seeds=SEEDS)
    finally:
        s.close()


def test_runner_masks_extra_seed_features_on_non_member_dates(pit_h6):
    from scripts import regime_study as runner

    feats, _, nfeats, _, info = pit_h6
    panels = info["null_excess_panels"]
    assert len(panels) == len(SEEDS) and info["h6_s"] > 0
    studied = list(feats["REGIME"].columns)
    for (ef, ey), seed in zip(panels, SEEDS):
        assert list(ef) == list(rs.NULL_EXCESS_FEATURES)
        assert list(ey) == list(rs.NULL_EXCESS_TARGETS)
        assert list(ef["REGIME"].columns) == studied
        for ticker in studied:
            bars = _pit_bars(ticker)
            f_full, y_full = rs.asset_study_frames(rs.shuffled_null_bars(bars, seed=seed))
            m = _member(ticker, bars.index)
            for col in rs.NULL_EXCESS_FEATURES:
                got = ef[col][ticker].reindex(bars.index)
                pd.testing.assert_series_equal(got, f_full[col].where(m), check_names=False,
                                               check_exact=True)
                # same mask as the existing null features
                np.testing.assert_array_equal(
                    got.isna().to_numpy() | ~m,
                    nfeats[col][ticker].reindex(bars.index).isna().to_numpy() | ~m)
            for col in rs.NULL_EXCESS_TARGETS:                  # targets never masked
                pd.testing.assert_series_equal(ey[col][ticker].reindex(bars.index), y_full[col],
                                               check_names=False, check_exact=True)
    # not vacuous: NEW.NS joins late, GONE.NS leaves
    ef = panels[0][0]["VOL_RATIO"]
    late = ef.index >= pd.Timestamp(PIT_COMPLETE_FROM)
    assert ef.loc[late & (ef.index < pd.Timestamp(date(2025, 2, 3))), "NEW.NS"].isna().all()
    assert ef.loc[ef.index >= pd.Timestamp(date(2025, 3, 3)), "GONE.NS"].isna().all()
    assert runner.extra_null_frames(_pit_bars("S00.NS"), SEEDS[0])[0].shape[1] == 3


def test_runner_active_comparison_extra_seeds_are_unmasked(pit_h6):
    info = pit_h6[4]
    act = info["active_null_excess_panels"]
    active = {t for t, (n, a, _) in PIT_ASSETS.items() if a and n >= 251}
    for (ef, ey), seed in zip(act, SEEDS):
        assert set(ef["REGIME"].columns) == active                 # incl. OUT.NS
        for ticker in ("NEW.NS", "OUT.NS"):
            bars = _pit_bars(ticker)
            f_full, _ = rs.asset_study_frames(rs.shuffled_null_bars(bars, seed=seed))
            pd.testing.assert_series_equal(ef["VOL_RATIO"][ticker].reindex(bars.index),
                                           f_full["VOL_RATIO"], check_names=False,
                                           check_exact=True)


def test_runner_without_seeds_leaves_info_unchanged(pit_h6):
    from tests.test_regime_study import _assert_panels_equal

    from scripts import regime_study as runner

    s = _pit_session()
    try:
        plain = runner.build_study_panels(s, universe="index", complete_from=PIT_COMPLETE_FROM,
                                          compare_active=True)
    finally:
        s.close()
    for a, b in zip(plain[:4], pit_h6[:4]):
        _assert_panels_equal(a, b)
    extra_keys = {"null_excess_panels", "active_null_excess_panels", "h6_s"}
    assert not extra_keys & set(plain[4])
    assert set(pit_h6[4]) - set(plain[4]) == extra_keys


@pytest.mark.parametrize("cut", [300, 517])
def test_extra_seed_features_are_truncation_invariant(cut):
    """Shuffle a fixed series with an H6 seed, then truncate: features at t do not change."""
    from scripts import regime_study as runner

    shuffled = rs.shuffled_null_bars(_bars(600, seed=13), seed=rs.NULL_EXCESS_SEEDS[2])
    full = rs.regime_conditioning(shuffled)
    trunc = rs.regime_conditioning(shuffled.iloc[:cut])
    for k in rs.NULL_EXCESS_FEATURES:
        np.testing.assert_allclose(trunc[k], full[k][:cut], rtol=1e-9, atol=1e-12,
                                   equal_nan=True, err_msg=k)
    # the runner's frames are those columns of the same computation
    f, _ = rs.asset_study_frames(shuffled)
    for k in rs.NULL_EXCESS_FEATURES:
        np.testing.assert_array_equal(f[k].to_numpy(), full[k])
    assert list(runner.extra_null_frames(_bars(600, seed=13), rs.NULL_EXCESS_SEEDS[2])[0]) \
        == list(rs.NULL_EXCESS_FEATURES)


@pytest.fixture(scope="module")
def pit_h6_reports(tmp_path_factory, monkeypatch_module):
    pytest.importorskip("talib")
    from scripts import regime_study as runner

    monkeypatch_module.setattr(rs, "NULL_EXCESS_SEEDS", rs.NULL_EXCESS_SEEDS[:3])
    root = tmp_path_factory.mktemp("h6_reports")
    s = _pit_session()
    out = {}
    try:
        for h6 in (False, True):
            report, res = runner.run(universe="index", session=s, report_dir=root, csv_dir=root,
                                     report_date=date(2026, 10, 19),
                                     complete_from=PIT_COMPLETE_FROM, h6=h6,
                                     robust=False)       # task 350's section: own tests
            out[h6] = (report.read_text(), res, report)
    finally:
        s.close()
    return out


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def _drop_h6_and_runtime(text: str) -> str:
    head, _, rest = text.partition("## H6: real minus shuffled null (pre-registered)")
    if rest:
        head += "## Options for Ankit" + rest.split("## Options for Ankit", 1)[1]
    return "\n".join(ln for ln in head.splitlines() if not ln.startswith("- SCORER_VERSION"))


def test_h6_report_section_and_everything_else_unchanged(pit_h6_reports):
    plain, _, _ = pit_h6_reports[False]
    text, res, report = pit_h6_reports[True]
    assert report.name == "regime_study_20261019_b.md"
    assert "## H6" not in plain and "H6's" not in plain
    sec = text.split("## H6: real minus shuffled null (pre-registered)")[1]
    sec = sec.split("## Options for Ankit")[0]
    assert "**index** universe" in sec and "**active** universe" in sec
    assert "Universe effect (H6)" in sec and "options (a)–(d)" in sec
    assert "mean excess" in sec and "BH q (H6)" in sec and "null mean" in sec
    assert sum(ln.startswith("- H6 ") for ln in sec.splitlines()) == 6
    assert "H6's 2 extra null seeds" in text
    assert (res.h6["n_seeds"] == 3).all()
    assert _drop_h6_and_runtime(text) == _drop_h6_and_runtime(plain)
    csv = report.parent / "regime_study_h6_20261019_b.csv"
    assert csv.exists() and len(pd.read_csv(csv)) == 12
    pd.testing.assert_frame_equal(res.tests, pit_h6_reports[False][1].tests, check_exact=True)
