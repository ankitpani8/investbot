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


def _bars(n: int, seed: int = 0, phi: float = 0.0, start: str = "2020-01-01") -> pd.DataFrame:
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
    idx = pd.bdate_range(start, periods=n, name="date")
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


# ──────────────────────────────────────────────────────────
# Point-in-time universe (task 260): in-memory SQLite fixture
# ──────────────────────────────────────────────────────────

from datetime import date, timedelta  # noqa: E402

PIT_START = "2023-01-02"
PIT_BARS = 620                                   # .. 2025-05-16
PIT_COMPLETE_FROM = date(2023, 9, 1)             # HURST_100 is finite from ~May 2023
J = date(2025, 2, 3)                             # NEW.NS joins
L = date(2025, 3, 3)                             # GONE.NS's first non-member session
BACK_OUT, BACK_IN = date(2025, 3, 17), date(2025, 4, 1)   # BACK.NS exits, re-enters
OLD = date(2019, 1, 1)

#: ticker -> (bars, is_active, NIFTY50 stints)
PIT_ASSETS = {
    **{f"S{i:02d}.NS": (PIT_BARS, True, [(OLD, None)]) for i in range(10)},
    "NEW.NS": (PIT_BARS, True, [(J, None)]),
    "GONE.NS": (PIT_BARS, False, [(OLD, L - timedelta(days=1))]),
    "BACK.NS": (PIT_BARS, True, [(OLD, BACK_OUT - timedelta(days=1)), (BACK_IN, None)]),
    "NOPX.NS": (0, True, [(OLD, None)]),                 # member, no price rows
    "SHORT.NS": (100, False, [(OLD, None)]),             # member, too few bars
    "OUT.NS": (PIT_BARS, True, []),                      # active, never a member
}


def _pit_session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from investbot.data.models import (
        Asset,
        AssetClass,
        Base,
        Exchange,
        IndexMembership,
        PriceDaily,
    )

    engine = create_engine("sqlite://", future=True, poolclass=StaticPool,
                           connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    ac = AssetClass(name="equity")
    ex = Exchange(asset_class=ac, code="NSE", name="NSE", currency="INR")
    s.add_all([ac, ex])
    s.flush()
    for k, (ticker, (n, active, stints)) in enumerate(PIT_ASSETS.items()):
        a = Asset(exchange_id=ex.id, ticker=ticker, name=ticker, asset_type="stock",
                  is_active=active)
        s.add(a)
        s.flush()
        if n:
            b = _bars(n, seed=500 + k, phi=0.1 * (k % 3), start=PIT_START)
            s.add_all([PriceDaily(asset_id=a.id, date=d.date(), open=r.open, high=r.high,
                                  low=r.low, close=r.close, adj_close=r.close,
                                  volume=int(r.volume)) for d, r in b.iterrows()])
        for f, to in stints:
            s.add(IndexMembership(index_code="NIFTY50", asset_id=a.id, effective_from=f,
                                  effective_to=to, source="test", verified=True))
    s.commit()
    return s


def _pit_bars(ticker: str) -> pd.DataFrame:
    k = list(PIT_ASSETS).index(ticker)
    return _bars(PIT_ASSETS[ticker][0], seed=500 + k, phi=0.1 * (k % 3), start=PIT_START)


def _old_build_study_panels(session, end=None):
    """``scripts.regime_study.build_study_panels`` before task 260, verbatim."""
    from investbot.evaluation.loader import active_tickers, load_daily_history
    from scripts.regime_study import MIN_STUDY_BARS

    real, null = {}, {}
    skipped, n_rows = [], 0
    first, last = None, None
    for ticker in active_tickers(session):
        bars = load_daily_history(session, ticker, end)
        if len(bars) < MIN_STUDY_BARS:
            skipped.append(f"{ticker} ({len(bars)} bars)")
            continue
        n_rows += len(bars)
        first = bars.index[0] if first is None else min(first, bars.index[0])
        last = bars.index[-1] if last is None else max(last, bars.index[-1])
        real[ticker] = rs.asset_study_frames(bars)
        null[ticker] = rs.asset_study_frames(rs.shuffled_null_bars(bars))
    info = {"n_assets": len(real), "skipped": skipped, "n_rows": n_rows,
            "first": first, "last": last}
    return (rs.to_wide({t: f for t, (f, _) in real.items()}),
            rs.to_wide({t: y for t, (_, y) in real.items()}),
            rs.to_wide({t: f for t, (f, _) in null.items()}),
            rs.to_wide({t: y for t, (_, y) in null.items()}),
            info)


@pytest.fixture(scope="module")
def pit():
    """One DB; the index panels (with the active comparison) and the old/new active panels."""
    pytest.importorskip("talib")
    from scripts import regime_study as runner

    s = _pit_session()
    try:
        index = runner.build_study_panels(s, universe="index", complete_from=PIT_COMPLETE_FROM,
                                          compare_active=True)
        active = runner.build_study_panels(s, universe="active")
        old = _old_build_study_panels(s)
    finally:
        s.close()
    return {"index": index, "active": active, "old": old}


def _assert_panels_equal(a: dict, b: dict):
    assert list(a) == list(b)
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k], check_exact=True, obj=k)


def test_active_universe_is_the_pre_260_study(pit):
    new, old = pit["active"], pit["old"]
    for a, b in zip(new[:4], old[:4]):
        _assert_panels_equal(a, b)
    assert {k: v for k, v in new[4].items() if k != "universe"} == old[4]
    assert new[4]["universe"] == "active"
    assert sorted(new[0]["REGIME"].columns) == sorted(
        t for t, (n, act, _) in PIT_ASSETS.items() if act and n >= 251)
    r_new, r_old = rs.run_study(*new[:4]), rs.run_study(*old[:4])
    pd.testing.assert_frame_equal(r_new.tests, r_old.tests, check_exact=True)
    pd.testing.assert_frame_equal(r_new.churn, r_old.churn, check_exact=True)


def test_index_comparison_panels_equal_the_active_universe(pit):
    """The active panels built in the index pass (incl. OUT.NS, never a member) are the
    active universe's panels."""
    for a, b in zip(pit["index"][4]["active_panels"], pit["active"][:4]):
        _assert_panels_equal(a, b)
    assert pit["index"][4]["n_active"] == pit["active"][4]["n_assets"]


def _member(ticker: str, dates: pd.DatetimeIndex) -> np.ndarray:
    """Hand-written point-in-time membership with the test's COMPLETE_FROM fallback."""
    n, active, stints = PIT_ASSETS[ticker]
    out = np.zeros(len(dates), dtype=bool)
    for f, to in stints:
        out |= (dates >= pd.Timestamp(f)) & ((dates <= pd.Timestamp(to)) if to else True)
    early = dates < pd.Timestamp(PIT_COMPLETE_FROM)
    out[early] = active
    return out


def test_index_universe_masks_features_on_non_member_dates_only(pit):
    from scripts import regime_study as runner

    feats, tg, nfeats, ntg, info = pit["index"]
    studied = {t for t, (n, _, st) in PIT_ASSETS.items() if st and n >= runner.MIN_STUDY_BARS}
    assert set(feats["REGIME"].columns) == studied              # OUT.NS: never a member
    assert info["n_members"] == 15 and "NOPX.NS" not in feats["REGIME"].columns
    assert "SHORT.NS (100 bars)" in info["skipped"]
    assert info["priced"] == {t for t, (n, _, st) in PIT_ASSETS.items() if st and n}
    for ticker in studied:
        bars = _pit_bars(ticker)
        f_full, y_full = rs.asset_study_frames(bars)
        nf_full, ny_full = rs.asset_study_frames(rs.shuffled_null_bars(bars))
        m = _member(ticker, bars.index)
        for col in rs.STUDY_FEATURES:
            got = feats[col][ticker].reindex(bars.index)
            exp = f_full[col].where(m)
            pd.testing.assert_series_equal(got, exp, check_names=False, check_exact=True)
            assert got[~m].isna().all()
            ngot = nfeats[col][ticker].reindex(bars.index)
            pd.testing.assert_series_equal(ngot, nf_full[col].where(m), check_names=False,
                                           check_exact=True)
        for col in tg:                                          # targets never masked
            pd.testing.assert_series_equal(tg[col][ticker].reindex(bars.index), y_full[col],
                                           check_names=False, check_exact=True)
            pd.testing.assert_series_equal(ntg[col][ticker].reindex(bars.index), ny_full[col],
                                           check_names=False, check_exact=True)
    # the masking is not vacuous: finite values exist on both sides of each change
    h = feats[rs.HURST_COL]
    idx = h.index
    late = idx >= pd.Timestamp(PIT_COMPLETE_FROM)
    assert h.loc[late & (idx < pd.Timestamp(J)), "NEW.NS"].isna().all()
    assert h.loc[idx >= pd.Timestamp(J), "NEW.NS"].notna().all()
    assert h.loc[(idx >= pd.Timestamp(L)), "GONE.NS"].isna().all()
    assert h.loc[late & (idx < pd.Timestamp(L)), "GONE.NS"].notna().all()
    gap = (idx >= pd.Timestamp(BACK_OUT)) & (idx < pd.Timestamp(BACK_IN))
    assert gap.any() and h.loc[gap, "BACK.NS"].isna().all()
    assert h.loc[idx >= pd.Timestamp(BACK_IN), "BACK.NS"].notna().all()
    # a target window of GONE.NS still runs past its exit
    er = tg["FWD_ER_20"]["GONE.NS"]
    assert er[(idx < pd.Timestamp(L)) & (idx >= pd.Timestamp(L) - pd.Timedelta(days=10))].notna().all()


def test_dates_before_complete_from_use_the_active_today_fallback(pit):
    feats = pit["index"][0]
    h = feats[rs.HURST_COL]
    early = h.index < pd.Timestamp(PIT_COMPLETE_FROM)
    warm = early & h["S00.NS"].notna().to_numpy()                 # HURST_100 out of warmup
    assert warm.sum() > 20
    # NEW.NS (active today) counts before its stint starts; GONE.NS (inactive today) does
    # not, although its stint covers those dates
    assert h.loc[warm, "NEW.NS"].notna().all()
    assert h.loc[warm, "GONE.NS"].isna().all()
    m = pit["index"][4]["membership"]
    assert m.loc[m.index < pd.Timestamp(PIT_COMPLETE_FROM), "NEW.NS"].all()
    assert not m.loc[m.index < pd.Timestamp(PIT_COMPLETE_FROM), "GONE.NS"].any()


def test_membership_used_on_t_is_membership_as_of_t_and_truncation_invariant(pit):
    """Truncating the price history (and the membership query) at t leaves every
    feature cell <= t unchanged on the sourced dates (>= COMPLETE_FROM); NEW.NS is NaN
    before J in both. Before COMPLETE_FROM the rule is "active today" by definition, so
    a ticker whose first stint starts after t (NEW.NS) is only loaded by the full run
    (as in ``panel.build_dataset``; 0 such dates on the real data)."""
    from scripts import regime_study as runner

    full = pit["index"][0]
    s = _pit_session()
    try:
        for t in (date(2025, 1, 15), date(2025, 3, 20)):     # before J; inside BACK's gap
            trunc = runner.build_study_panels(s, t, universe="index",
                                              complete_from=PIT_COMPLETE_FROM)[0]
            early = full["REGIME"].index < pd.Timestamp(PIT_COMPLETE_FROM)
            if t < J:
                assert full[rs.HURST_COL].loc[early, "NEW.NS"].notna().any()
            for col, F in full.items():
                ref = F.loc[(F.index <= pd.Timestamp(t)) & ~early]
                got = trunc[col].reindex(index=ref.index, columns=ref.columns)
                # NaN pattern exact; values to the tolerance of the regime truncation test
                # (TA-Lib's recursive smoothers round differently on a shorter series)
                np.testing.assert_array_equal(np.isnan(got.to_numpy()), np.isnan(ref.to_numpy()),
                                              err_msg=f"{col} @ {t}")
                np.testing.assert_allclose(got.to_numpy(), ref.to_numpy(), rtol=1e-9,
                                           atol=1e-12, equal_nan=True, err_msg=f"{col} @ {t}")
            if t < J:
                assert "NEW.NS" not in trunc["REGIME"].columns   # not yet a member
    finally:
        s.close()


def test_exit_then_reentry_is_not_a_flip_or_crossing():
    idx = pd.bdate_range("2024-01-01", periods=7)
    # a: trend, exits (NaN), re-enters as range: no flip across the gap
    labels = pd.DataFrame({"a": [1, 1, np.nan, np.nan, 0, 0, 1],
                           "b": [1, 1, 1, 1, 1, 1, 1]}, idx, dtype=float)
    br = rs.label_base_rates(labels, idx).set_index("ticker")
    assert br.loc["a", "flips_per_1000"] == pytest.approx(1000 * 1 / 3)     # pairs 0-1, 4-5, 5-6
    assert br.loc["a", "n_bars"] == 5
    assert br.loc["ALL", "flips_per_1000"] == pytest.approx(1000 * 1 / 9)
    conf = pd.DataFrame({"a": [0.0, 0.05, np.nan, np.nan, 0.7, 0.6, 0.4]}, idx)
    ch = rs.confidence_churn(conf, idx)
    assert ch["n_pairs"] == 3
    assert ch["cross_0.1_per_1000"] == 0
    assert ch["cross_0.5_per_1000"] == pytest.approx(1000 * 1 / 3)          # 0.6 -> 0.4 only
    assert ch["mean_abs_change"] == pytest.approx((0.05 + 0.1 + 0.2) / 3)


def test_reentry_gap_on_the_index_panel(pit):
    """BACK.NS's flips on the masked panel equal a hand count over adjacent member bars."""
    lab = pit["index"][0]["REGIME"]["BACK.NS"]
    a = lab.to_numpy()
    pair = np.isfinite(a[1:]) & np.isfinite(a[:-1])
    flips = int((pair & (a[1:] != a[:-1])).sum())
    br = rs.label_base_rates(lab.to_frame(), lab.index).set_index("ticker")
    assert br.loc["BACK.NS", "flips_per_1000"] == pytest.approx(1000 * flips / pair.sum())
    gap = (lab.index >= pd.Timestamp(BACK_OUT)) & (lab.index < pd.Timestamp(BACK_IN))
    first_in = int(np.argmax(lab.index >= pd.Timestamp(BACK_IN)))
    assert not pair[first_in - 1] and gap[first_in - 1]          # re-entry pair is broken


def test_coverage_summary_hand_example():
    from scripts import regime_study as runner

    dates = pd.bdate_range("2024-01-01", periods=4)
    M = pd.DataFrame({"A": [True] * 4, "B": [True] * 4, "C": [True] * 4,
                      "D": [False, False, True, True]}, index=dates)
    # C has no prices; D is studied and joins on day 3
    R_ = pd.DataFrame({**{f"X{i}": [1.0] * 4 for i in range(9)},
                       "A": [1.0] * 4, "B": [np.nan, 1, 1, 1], "D": [0, 0, np.nan, 2.0]},
                      index=dates)
    M = pd.concat([M, pd.DataFrame(True, index=dates, columns=[f"X{i}" for i in range(9)])],
                  axis=1)
    out = runner.coverage_summary(M, R_, dates[1:], priced=set(M.columns) - {"C"},
                                  complete_from=date(2024, 1, 2))
    # day 2: 12 members (X0-8, A, B, C), 11 covered; day 3: 13, 11; day 4: 13, 12
    assert out["coverage_mean"] == pytest.approx(np.mean([11 / 12, 11 / 13, 12 / 13]))
    assert out["coverage_min"] == pytest.approx(11 / 13)
    assert out["coverage_min_date"] == dates[2]
    assert out["missing_prices"] == {"C": 3}
    assert out["short_history"] == {}
    assert out["fallback_dates"] == 1 and out["fallback_oos_dates"] == 0


def test_mask_non_members_keeps_rows_and_copies():
    from scripts import regime_study as runner

    f = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    out = runner.mask_non_members(f, np.array([True, False, True]))
    assert out.index.equals(f.index) and out.loc[1].isna().all()
    assert out.loc[[0, 2]].equals(f.loc[[0, 2]]) and f.notna().all().all()


def test_no_dropna_in_the_study():
    import inspect

    from scripts import regime_study as runner

    for mod in (rs, runner):
        assert ".dropna(" not in inspect.getsource(mod)


@pytest.fixture(scope="module")
def pit_reports(tmp_path_factory):
    pytest.importorskip("talib")
    from scripts import regime_study as runner

    root = tmp_path_factory.mktemp("pit_reports")
    s = _pit_session()
    out = {}
    try:
        for u in ("index", "active"):
            report, res = runner.run(universe=u, session=s, report_dir=root, csv_dir=root,
                                     report_date=date(2026, 10, 19),
                                     complete_from=PIT_COMPLETE_FROM)
            out[u] = (report.read_text(), res, report)
    finally:
        s.close()
    return out


def test_index_report_has_coverage_note_and_universe_effect(pit_reports):
    from investbot.evaluation.report import SURVIVORSHIP_CAVEAT

    text, res, report = pit_reports["index"]
    assert report.name == "regime_study_20261019.md"
    assert len(res.oos) > 0
    assert "**Universe: point-in-time NIFTY50.**" in text and SURVIVORSHIP_CAVEAT not in text
    assert "Coverage is below 100%" in text and "NOPX.NS" in text and "SHORT.NS" in text
    assert "## Universe effect" in text and "verdict index" in text and "verdict active" in text
    assert ("No verdict changes between the universes" in text
            or "Verdicts that change between the universes" in text)
    sec = text.split("## Universe effect")[1].split("## Options for Ankit")[0]
    rows = [ln for ln in sec.splitlines() if ln.startswith("| H")]
    assert len(rows) == len(res.tests)
    assert "SCORER_VERSION" in text and "max price date 2025-05-16" in text
    assert "active-universe comparison" in text
    assert "calendar dates" in text and f"before {PIT_COMPLETE_FROM}" in text


def test_active_report_keeps_the_old_layout(pit_reports):
    from investbot.evaluation.report import SURVIVORSHIP_CAVEAT

    text, _, report = pit_reports["active"]
    assert report.name == "regime_study_20261019_b.md"        # never overwrites
    assert text.count(SURVIVORSHIP_CAVEAT) == 2                # header and caveats
    assert "## Universe effect" not in text and "Universe: point-in-time" not in text
    assert "active stocks (today's NIFTY50 constituents" in text
    assert "max price date" in text


def test_active_report_tests_equal_old_study(pit_reports, pit):
    """``run(universe="active")`` gives the same test table as the pre-260 panels."""
    _, res, _ = pit_reports["active"]
    old = rs.run_study(*pit["old"][:4])
    pd.testing.assert_frame_equal(res.tests, old.tests, check_exact=True)
