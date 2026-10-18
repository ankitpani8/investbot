"""
Tests for investbot.evaluation.regime_study (synthetic data, no DB or network).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from investbot.evaluation import regime_study as rs
from investbot.scoring.technical import regime as R

CLOSE_12 = np.array([10.0, 10.5, 10.2, 10.8, 11.0, 10.7, 10.9, 11.4, 11.1, 11.6, 11.3, 11.9])


def _bars(n: int, seed: int = 0, phi: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0003, 0.015, n)
    r = np.empty(n)
    r[0] = e[0]
    for i in range(1, n):
        r[i] = phi * r[i - 1] + e[i]
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    vol = rng.integers(100_000, 1_000_000, n).astype(float)
    idx = pd.bdate_range("2020-01-01", periods=n, name="date")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": vol}, index=idx)


# ──────────────────────────────────────────────────────────
# Targets (labels)
# ──────────────────────────────────────────────────────────

def _hand_targets(close, t, h, rv):
    """Straight-line reference implementation for one bar."""
    r = [math.log(close[t + i + 1] / close[t + i]) for i in range(h)]
    er = abs(sum(r)) / sum(abs(x) for x in r)
    m = sum(r) / h
    ac1 = sum((r[i] - m) * (r[i + 1] - m) for i in range(h - 1)) / sum((x - m) ** 2 for x in r)
    sd = math.sqrt(sum((x - m) ** 2 for x in r) / (h - 1))
    peak, mdd = close[t], 0.0
    for c in close[t: t + h + 1]:
        peak = max(peak, c)
        mdd = min(mdd, c / peak - 1.0)
    return er, ac1, sd / rv, mdd


def test_targets_hand_computed_h3():
    rv = np.full(12, 0.02)
    out = rs.forward_regime_targets(CLOSE_12, horizons=(3,), realvol=rv)
    # t = 0 fully by hand: r = ln(10.5/10), ln(10.2/10.5), ln(10.8/10.2)
    r = np.log([1.05, 10.2 / 10.5, 10.8 / 10.2])
    assert out["FWD_ER_3"][0] == pytest.approx(abs(r.sum()) / np.abs(r).sum())
    d = r - r.mean()
    assert out["FWD_AC1_3"][0] == pytest.approx((d[0] * d[1] + d[1] * d[2]) / (d @ d))
    assert out["FWD_VOLX_3"][0] == pytest.approx(r.std(ddof=1) / 0.02)
    assert out["FWD_MDD_3"][0] == pytest.approx(10.2 / 10.5 - 1)          # peak 10.5, trough 10.2
    for t in range(12 - 3):
        er, ac1, volx, mdd = _hand_targets(CLOSE_12, t, 3, 0.02)
        assert out["FWD_ER_3"][t] == pytest.approx(er)
        assert out["FWD_AC1_3"][t] == pytest.approx(ac1)
        assert out["FWD_VOLX_3"][t] == pytest.approx(volx)
        assert out["FWD_MDD_3"][t] == pytest.approx(mdd)
        assert out["FWD_MDD_3"][t] <= 0
    # NaN exactly on the last H bars
    for k in ("ER", "AC1", "VOLX", "MDD"):
        a = out[f"FWD_{k}_3"]
        assert np.isfinite(a[:9]).all() and np.isnan(a[9:]).all(), k
    # VR5 needs at least two 5-bar sums, so H = 3 has none
    assert np.isnan(out["FWD_VR5_3"]).all()


def test_variance_ratio_hand_computed_h6():
    out = rs.forward_regime_targets(CLOSE_12, horizons=(6,))
    for t in range(12 - 6):
        r = np.diff(np.log(CLOSE_12[t: t + 7]))
        sums = np.array([r[0:5].sum(), r[1:6].sum()])
        assert out["FWD_VR5_6"][t] == pytest.approx(sums.var(ddof=1) / (5 * r.var(ddof=1)))
    assert np.isnan(out["FWD_VR5_6"][6:]).all()
    assert np.isnan(out["FWD_VOLX_6"]).all()        # no realvol given


def test_nan_inside_window_poisons_only_windows_that_contain_it():
    c = CLOSE_12.copy()
    c[6] = np.nan
    out = rs.forward_regime_targets(c, horizons=(3,), realvol=np.full(12, 0.02))
    ok = rs.forward_regime_targets(CLOSE_12, horizons=(3,), realvol=np.full(12, 0.02))
    for k in ("ER", "AC1", "VOLX", "MDD"):
        a = out[f"FWD_{k}_3"]
        assert np.isnan(a[3:7]).all(), k                      # windows t..t+3 containing bar 6
        np.testing.assert_allclose(a[:3], ok[f"FWD_{k}_3"][:3])
        np.testing.assert_allclose(a[7:9], ok[f"FWD_{k}_3"][7:9])


def test_forward_hurst_is_rolling_hurst_on_the_forward_slice():
    close = _bars(400, seed=3)["close"].to_numpy()
    w = rs.EX_POST_HURST_WINDOW
    out = rs.forward_regime_targets(close)
    fh = out[f"FWD_HURST_{w}"]
    for t in (0, 57, 150, 400 - w - 1):
        ref = R.rolling_hurst(np.log(close[t: t + w + 1]), w)[-1]
        assert fh[t] == pytest.approx(ref, rel=1e-10)
    assert np.isnan(fh[400 - w:]).all() and np.isfinite(fh[: 400 - w]).all()


def test_target_locality():
    """A target at t only depends on closes t..t+H (plus realvol[t])."""
    close = _bars(300, seed=5)["close"].to_numpy()
    rv = np.full(300, 0.015)
    hz, w = (20, 60), 100
    base = rs.forward_regime_targets(close, hz, w, realvol=rv)
    t = 120
    rng = np.random.default_rng(1)
    for stem_h, last in [*(((k, h), t + h) for h in hz for k in ("ER", "AC1", "VR5", "VOLX", "MDD")),
                         (("HURST", w), t + w)]:
        col = f"FWD_{stem_h[0]}_{stem_h[1]}"
        bumped = close.copy()
        bumped[:t] *= np.exp(rng.normal(0, 0.05, t))           # before t
        bumped[last + 1:] *= np.exp(rng.normal(0, 0.05, 300 - last - 1))  # after t+H
        out = rs.forward_regime_targets(bumped, hz, w, realvol=rv)
        assert out[col][t] == pytest.approx(base[col][t], rel=1e-12), col
        # and changing a bar inside the window does move it
        inside = close.copy()
        inside[t + 1] *= 0.5
        assert out[col][t] != rs.forward_regime_targets(inside, hz, w, realvol=rv)[col][t], col


# ──────────────────────────────────────────────────────────
# Features
# ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def bars600():
    return _bars(600, seed=11)


@pytest.fixture(scope="module")
def cond600(bars600):
    return rs.regime_conditioning(bars600)


def test_conditioning_columns_and_span1(cond600):
    expected = [*rs.CONDITIONING_COLUMNS, *(rs.confidence_column(s) for s in rs.CONFIDENCE_SPANS)]
    assert list(cond600) == expected
    np.testing.assert_array_equal(cond600["REGIME_CONFIDENCE_S1"], cond600["REGIME_CONFIDENCE_RAW"])
    span = R.CONFIDENCE_EMA_SPAN
    np.testing.assert_allclose(cond600[f"REGIME_CONFIDENCE_S{span}"], cond600["REGIME_CONFIDENCE"],
                               equal_nan=True)
    assert np.isfinite(cond600["REGIME_CONFIDENCE_S20"][300:]).all()


@pytest.mark.parametrize("cut", [280, 431, 599])
def test_conditioning_truncation_invariance(bars600, cond600, cut):
    trunc = rs.regime_conditioning(bars600.iloc[:cut])
    for k, v in trunc.items():
        np.testing.assert_allclose(v, cond600[k][:cut], rtol=1e-9, atol=1e-12,
                                   equal_nan=True, err_msg=k)


# ──────────────────────────────────────────────────────────
# Shuffled null
# ──────────────────────────────────────────────────────────

def _ac1(x):
    d = x - x.mean()
    return float(d[1:] @ d[:-1] / (d @ d))


def test_shuffled_null_bars():
    bars = _bars(2000, seed=7, phi=0.5)
    a = rs.shuffled_null_bars(bars)
    b = rs.shuffled_null_bars(bars)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(rs.shuffled_null_bars(bars, seed=1))
    assert a.index.equals(bars.index) and list(a.columns) == list(bars.columns)

    r_old = np.diff(np.log(bars["close"].to_numpy()))
    r_new = np.diff(np.log(a["close"].to_numpy()))
    np.testing.assert_allclose(np.sort(r_new), np.sort(r_old), atol=1e-12)
    assert a["close"].iloc[0] == bars["close"].iloc[0]
    np.testing.assert_array_equal(np.sort(a["volume"]), np.sort(bars["volume"]))

    eps = 1e-9
    assert (a["low"] <= a[["open", "close"]].min(axis=1) * (1 + eps)).all()
    assert (a["high"] >= a[["open", "close"]].max(axis=1) * (1 - eps)).all()

    assert _ac1(r_old) > 0.4
    assert abs(_ac1(r_new)) < 0.05


# ──────────────────────────────────────────────────────────
# Conditional-mean test
# ──────────────────────────────────────────────────────────

def _ar1_ac1(phi: float, h: int, rng) -> float:
    e = rng.normal(0, 0.01, h + 50)
    r = np.zeros_like(e)
    for i in range(1, len(e)):
        r[i] = phi * r[i - 1] + e[i]
    path = np.exp(np.cumsum(r[50:]))
    return rs.forward_regime_targets(np.r_[1.0, path], horizons=(h,))[f"FWD_AC1_{h}"][0]


def _synthetic_panel(phi_trend: float, phi_range: float, seed: int, h: int = 20,
                     n_dates: int = 250, n_names: int = 16):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n_dates, name="date")
    labels = rng.integers(0, 2, (n_dates, n_names)).astype(float)       # range/trend
    y = np.empty((n_dates, n_names))
    for i in range(n_dates):
        for j in range(n_names):
            y[i, j] = _ar1_ac1(phi_trend if labels[i, j] == rs.TREND else phi_range, h, rng)
    cols = [f"S{j}" for j in range(n_names)]
    return pd.DataFrame(labels, idx, cols), pd.DataFrame(y, idx, cols)


def test_conditional_mean_detects_persistence():
    labels, y = _synthetic_panel(0.4, -0.4, seed=2)
    d = rs.group_difference(labels, y, (rs.TREND,), (rs.RANGE,))
    s = rs._summary(d, 20)
    assert s["mean"] > 0 and s["t_nw"] > 3


def test_conditional_mean_noise_is_null():
    labels, y = _synthetic_panel(0.0, 0.0, seed=4)
    d = rs.group_difference(labels, y, (rs.TREND,), (rs.RANGE,))
    assert abs(rs._summary(d, 20)["t_nw"]) < 2


def test_group_mean_respects_min_names():
    idx = pd.bdate_range("2024-01-01", periods=2)
    labels = pd.DataFrame([[1, 1, 0, 0], [1, 1, 1, 0]], idx, list("abcd"), dtype=float)
    y = pd.DataFrame([[1, 2, 3, 4], [1, 2, 3, 4]], idx, list("abcd"), dtype=float)
    m = rs.group_mean(labels, y, (rs.TREND,), min_names=3)
    assert np.isnan(m.iloc[0]) and m.iloc[1] == pytest.approx(2.0)


# ──────────────────────────────────────────────────────────
# Calibration helpers, base rates, verdict
# ──────────────────────────────────────────────────────────

def test_hit_matrix_signs_and_range():
    idx = pd.bdate_range("2024-01-01", periods=1)
    labels = pd.DataFrame([[1, 1, 0, 0, 2]], idx, list("abcde"), dtype=float)
    er = pd.DataFrame([[0.9, 0.1, 0.8, 0.2, 0.5]], idx, list("abcde"))
    hit = rs.hit_matrix(labels, er).iloc[0]
    # ranks among a..d: 0.9 -> 4, 0.1 -> 1, 0.8 -> 3, 0.2 -> 2 -> pct +0.5, -0.5, +1/6, -1/6
    assert hit["a"] == pytest.approx(0.5)            # trend, most efficient: right
    assert hit["b"] == pytest.approx(-0.5)           # trend, least efficient: wrong
    assert hit["c"] == pytest.approx(-1 / 6)         # range, fairly efficient: wrong
    assert hit["d"] == pytest.approx(1 / 6)
    assert np.isnan(hit["e"])                        # chaos excluded


def test_base_rates_and_churn():
    idx = pd.bdate_range("2024-01-01", periods=5)
    labels = pd.DataFrame({"a": [0, 1, 1, 0, 0], "b": [1, 1, 1, 1, np.nan]}, idx, dtype=float)
    br = rs.label_base_rates(labels, idx).set_index("ticker")
    assert br.loc["a", "flips_per_1000"] == pytest.approx(1000 * 2 / 4)
    assert br.loc["b", "flips_per_1000"] == 0
    assert br.loc["ALL", "share_trend"] == pytest.approx(6 / 9)
    assert br.loc["ALL", "flips_per_1000"] == pytest.approx(1000 * 2 / 7)

    conf = pd.DataFrame({"a": [0.0, 0.2, 0.05, 0.6, 0.6]}, idx)
    ch = rs.confidence_churn(conf, idx)
    assert ch["n_pairs"] == 4
    assert ch["mean_abs_change"] == pytest.approx((0.2 + 0.15 + 0.55 + 0) / 4)
    assert ch["cross_0.1_per_1000"] == pytest.approx(1000 * 3 / 4)
    assert ch["cross_0.5_per_1000"] == pytest.approx(1000 * 1 / 4)


def test_verdict_rule():
    base = {"mean": 0.1, "q": 0.01, "block_share": 0.8, "expected_sign": 1}
    assert rs.verdict(base) == "supported"
    assert rs.verdict({**base, "block_share": 0.5}) == "not supported"
    assert rs.verdict({**base, "q": 0.5}) == "not supported"
    assert rs.verdict({**base, "mean": -0.1}) == "wrong sign"
    assert rs.verdict({**base, "mean": -0.1, "q": 0.5}) == "not supported"


@pytest.fixture(scope="module")
def synthetic_study():
    """The whole pipeline on a small synthetic universe."""
    real, null = {}, {}
    for k in range(12):
        bars = _bars(900, seed=100 + k, phi=0.1 * (k % 3))
        real[f"S{k}"] = rs.asset_study_frames(bars)
        null[f"S{k}"] = rs.asset_study_frames(rs.shuffled_null_bars(bars))
    feats = rs.to_wide({t: f for t, (f, _) in real.items()})
    tg = rs.to_wide({t: y for t, (_, y) in real.items()})
    nf = rs.to_wide({t: f for t, (f, _) in null.items()})
    ny = rs.to_wide({t: y for t, (_, y) in null.items()})
    return rs.run_study(feats, tg, nf, ny)


def test_run_study_end_to_end_synthetic(synthetic_study):
    res = synthetic_study
    assert len(res.splits) >= 1 and len(res.oos) > 0
    n_cond = 2 * 3 + 1 + 2 + 2
    n_rank = 2 * 3 + 2 * 4
    n_cal = 2 * len(rs.CONFIDENCE_SPANS)
    assert len(res.tests) == n_cond + n_rank + n_cal
    assert len(res.null_conditional) == n_cond and len(res.null_rank) == n_rank
    q = res.tests["q"]
    assert (q.isna() | q.between(0, 1)).all() and q.notna().sum() >= 20
    assert set(res.tests["verdict"]) <= {"supported", "not supported", "wrong sign"}
    assert len(res.churn) == len(rs.CONFIDENCE_SPANS)
    assert res.base_rates_real["ticker"].iloc[-1] == "ALL"


def test_runner_renders_and_never_overwrites(synthetic_study, tmp_path):
    from datetime import date

    from scripts import regime_study as runner

    d = date(2026, 10, 18)
    first = runner.report_paths(tmp_path, tmp_path, d)
    assert first[0].name == "regime_study_20261018.md"
    first[0].write_text("x")
    second = runner.report_paths(tmp_path, tmp_path, d)
    assert second[0].name == "regime_study_20261018_b.md"
    assert second[1].name == "regime_study_tests_20261018_b.csv"

    info = {"n_assets": 12, "skipped": [], "n_rows": 10800,
            "first": synthetic_study.oos[0], "last": synthetic_study.oos[-1]}
    text = runner.render_report(synthetic_study, info, d, elapsed_s=1.0, rss_mb=100.0,
                                git_hash="abc1234", end=None)
    for heading in ("## Setup", "## Label base rates (real vs null)", "## Conditional means",
                    "## Rank association", "## Confidence calibration by span",
                    "## Verdict per pre-registered hypothesis", "## Options for Ankit"):
        assert heading in text
    assert "SCORER_VERSION" in text and "abc1234" in text and "peak RSS" in text
