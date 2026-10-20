"""
Tests for investbot.evaluation (walk-forward forward-return harness).

Synthetic data only; the one DB test uses a throwaway SQLite file.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from investbot.evaluation import stats
from investbot.evaluation.baselines import BASELINE_COLUMNS, compute_baselines
from investbot.evaluation.evaluate import evaluate, headline
from investbot.evaluation.labels import PRICE_STEMS, TR_STEMS, forward_returns, label_columns
from investbot.evaluation.metrics import (
    bucket_returns,
    daily_ic,
    long_only_costs,
    quantile_buckets,
    spread_series,
    top_membership,
    turnover,
)
from investbot.evaluation.walkforward import oos_dates, walk_forward_splits


def _bars(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    vol = rng.integers(100_000, 1_000_000, n).astype(float)
    idx = pd.bdate_range("2020-01-01", periods=n, name="date")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": vol}, index=idx)


# ──────────────────────────────────────────────────────────
# Labels
# ──────────────────────────────────────────────────────────

def test_label_alignment_exact():
    idx = pd.bdate_range("2024-01-01", periods=6)
    bars = pd.DataFrame({"open": [10, 11, 12, 13, 14, 15.0],
                         "close": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5]}, index=idx)
    y = forward_returns(bars, (1, 2))
    # t = 0: entry open[1] = 11; exit close[1] = 11.5 (h=1), close[2] = 12.5 (h=2)
    assert y["FWD_RET_1"][0] == pytest.approx(11.5 / 11 - 1)
    assert y["FWD_RET_2"][0] == pytest.approx(12.5 / 11 - 1)
    assert y["FWD_CC_RET_1"][0] == pytest.approx(11.5 / 10.5 - 1)
    assert y["FWD_CC_RET_2"][3] == pytest.approx(15.5 / 13.5 - 1)
    # the last h bars have no label
    assert np.isnan(y["FWD_RET_1"][-1]) and np.isfinite(y["FWD_RET_1"][-2])
    assert np.isnan(y["FWD_RET_2"][-2:]).all() and np.isfinite(y["FWD_RET_2"][-3])
    assert list(y) == label_columns((1, 2))


@pytest.mark.parametrize("h", [1, 5, 20])
def test_label_uses_only_bars_t_plus_1_to_t_plus_h(h):
    bars = _bars(80)
    t = 30
    base = forward_returns(bars, (h,))[f"FWD_RET_{h}"][t]
    # bars <= t and bars > t+h do not matter
    other = bars.copy()
    other.iloc[: t + 1] *= 1.7
    other.iloc[t + h + 1:] *= 0.3
    assert forward_returns(other, (h,))[f"FWD_RET_{h}"][t] == pytest.approx(base)
    # the entry open (t+1) and the exit close (t+h) do
    moved = bars.copy()
    moved.iloc[t + 1, moved.columns.get_loc("open")] *= 1.1
    assert forward_returns(moved, (h,))[f"FWD_RET_{h}"][t] != pytest.approx(base)
    moved = bars.copy()
    moved.iloc[t + h, moved.columns.get_loc("close")] *= 1.1
    assert forward_returns(moved, (h,))[f"FWD_RET_{h}"][t] != pytest.approx(base)


# ──────────────────────────────────────────────────────────
# Features: no lookahead, separate from labels
# ──────────────────────────────────────────────────────────

def test_baselines_values_and_truncation_invariance():
    bars = _bars(400, seed=3)
    b = compute_baselines(bars)
    c = bars["close"].to_numpy()
    t = 300
    assert b["BASE_MOM_20"][t] == pytest.approx(c[t] / c[t - 20] - 1)
    assert b["BASE_MOM_12_1"][t] == pytest.approx(c[t - 21] / c[t - 252] - 1)
    assert b["BASE_REV_5"][t] == pytest.approx(-(c[t] / c[t - 5] - 1))
    lr = np.log(c[t - 19:t + 1] / c[t - 20:t])
    assert b["BASE_LOWVOL_20"][t] == pytest.approx(-lr.std(ddof=1))
    assert np.isnan(b["BASE_MOM_12_1"][251]) and np.isfinite(b["BASE_MOM_12_1"][252])
    for cut in (260, 333, 399):
        short = compute_baselines(bars.iloc[: cut + 1])
        for k in BASELINE_COLUMNS:
            np.testing.assert_allclose(short[k], b[k][: cut + 1], equal_nan=True)


def _make_db(path, n_bars: dict[str, int]):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from investbot.data.models import Asset, AssetClass, Base, Exchange, PriceDaily

    engine = create_engine(f"sqlite:///{path}", future=True)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    ac = AssetClass(name="equity")
    ex = Exchange(asset_class=ac, code="NSE", name="NSE", currency="INR")
    s.add_all([ac, ex])
    s.flush()
    for k, (ticker, n) in enumerate(n_bars.items()):
        a = Asset(exchange_id=ex.id, ticker=ticker, name=ticker, asset_type="stock", is_active=True)
        s.add(a)
        s.flush()
        bars = _bars(n, seed=k + 1)
        bars.index = pd.bdate_range(end="2026-10-15", periods=n)
        s.add_all([PriceDaily(asset_id=a.id, date=d.date(), open=r.open, high=r.high, low=r.low,
                              close=r.close, adj_close=r.close, volume=int(r.volume))
                   for d, r in bars.iterrows()])
    s.commit()
    return s


@pytest.fixture(scope="module")
def full_features():
    pytest.importorskip("talib")
    from investbot.evaluation.panel import asset_features
    bars = _bars(520, seed=7)
    return bars, asset_features(bars)


def test_asset_features_columns(full_features):
    _, f = full_features
    assert "TECH_SCORE" in f and "REGIME" in f and "REGIME_CONFIDENCE" in f
    assert any(c.startswith("SIG_") for c in f.columns)
    assert set(BASELINE_COLUMNS) <= set(f.columns)
    assert not any(c.startswith("FWD_") for c in f.columns)
    assert f["TECH_SCORE"].notna().sum() > 100
    assert f["TECH_SCORE"].isna().iloc[:50].all()      # warmup kept, not dropped


@pytest.mark.parametrize("cut", [330, 450])
def test_features_unchanged_when_future_deleted(full_features, cut):
    from investbot.evaluation.panel import asset_features
    bars, full = full_features
    trunc = asset_features(bars.iloc[: cut + 1])
    assert trunc.index[-1] == bars.index[cut]
    pd.testing.assert_frame_equal(trunc, full.iloc[: cut + 1], check_exact=False,
                                  rtol=1e-9, atol=1e-12)


def test_build_dataset_separates_features_and_labels(tmp_path):
    pytest.importorskip("talib")
    from investbot.evaluation.panel import build_dataset

    s = _make_db(tmp_path / "e.db", {"AAA.NS": 300, "BBB.NS": 300, "TINY.NS": 5})
    try:
        feats, labels = build_dataset(start=pd.Timestamp("2026-01-01").date(), session=s,
                                      universe="active")
    finally:
        s.close()
    assert feats.index.names == ["date", "ticker"]
    assert feats.index.equals(labels.index)
    assert set(feats.index.get_level_values("ticker")) == {"AAA.NS", "BBB.NS"}
    assert feats.index.get_level_values("date").min() >= pd.Timestamp("2026-01-01")
    assert not set(feats.columns) & set(labels.columns)
    assert list(labels.columns) == label_columns()
    # labels are the asset's own next bars
    a = labels.xs("AAA.NS", level="ticker")
    assert a["FWD_RET_1"].iloc[-1:].isna().all() and a["FWD_RET_20"].iloc[-20:].isna().all()


# ──────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────

def test_newey_west_hand_example():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    # mean 3, d = [-2,-1,0,1,2]; gamma0 = 10/5 = 2; gamma1 = (2+0+0+2)/5 = 0.8
    # lag 1: var = 2 + 2 * (1 - 1/2) * 0.8 = 2.8; se = sqrt(2.8 / 5)
    assert stats.newey_west_se(x, 1) == pytest.approx(math.sqrt(2.8 / 5))
    assert stats.newey_west_tstat(x, 1) == pytest.approx(3 / math.sqrt(0.56))
    # lag 0: plain variance of the mean with 1/n
    assert stats.newey_west_se(x, 0) == pytest.approx(math.sqrt(2 / 5))
    # lag 2: gamma2 = ((0)(-2) + (1)(-1) + (2)(0))/5 = -0.2; weights 2/3, 1/3
    var2 = 2 + 2 * (2 / 3) * 0.8 + 2 * (1 / 3) * (-0.2)
    assert stats.newey_west_se(x, 2) == pytest.approx(math.sqrt(var2 / 5))
    # NaN ignored
    assert stats.newey_west_se([1.0, np.nan, 2, 3, 4, 5], 1) == pytest.approx(math.sqrt(0.56))


def test_newey_west_widens_for_overlapping_sums():
    rng = np.random.default_rng(0)
    e = rng.normal(size=3000)
    x = pd.Series(e).rolling(20).sum().to_numpy()[19:]     # MA(19), like overlapping 20-day returns
    assert stats.newey_west_se(x, 19) > 2.5 * stats.newey_west_se(x, 0)


def test_bh_hand_example():
    p = [0.01, 0.04, 0.03, 0.005]
    # sorted 0.005, 0.01, 0.03, 0.04 → ×4/k = 0.02, 0.02, 0.04, 0.04
    np.testing.assert_allclose(stats.bh_adjust(p), [0.02, 0.04, 0.04, 0.02])
    q = stats.bh_adjust([0.5, np.nan, 0.01])
    assert np.isnan(q[1])
    np.testing.assert_allclose(q[[0, 2]], [0.5, 0.02])
    np.testing.assert_allclose(stats.bh_adjust([0.9, 0.95]), [0.95, 0.95])


def test_bh_matches_statsmodels_if_installed():
    multitest = pytest.importorskip("statsmodels.stats.multitest")
    p = np.random.default_rng(1).uniform(size=40) ** 2
    np.testing.assert_allclose(stats.bh_adjust(p), multitest.multipletests(p, method="fdr_bh")[1])


def test_spearman_and_pvalue():
    x = np.array([1, 2, 2, 4, 5, np.nan])
    y = np.array([5, 6, 7, 8, 7, 1])
    expected = pd.Series(x[:5]).rank().corr(pd.Series(y[:5]).rank())
    assert stats.spearman(x, y) == pytest.approx(expected)
    assert stats.normal_pvalue(1.959963985) == pytest.approx(0.05, abs=1e-6)
    assert math.isnan(stats.normal_pvalue(math.nan))


# ──────────────────────────────────────────────────────────
# Cross-sectional metrics
# ──────────────────────────────────────────────────────────

def _wide(values, n_dates=1):
    idx = pd.bdate_range("2024-01-01", periods=n_dates)
    return pd.DataFrame(np.atleast_2d(values), index=idx,
                        columns=[f"T{i:02d}" for i in range(np.atleast_2d(values).shape[1])])


def test_quintile_assignment_with_ties_and_nans():
    s = _wide([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan, 5]])
    y = _wide([[0.0] * 12])
    y.iloc[0, 9] = np.nan                       # T09 has no label → excluded too
    b = quantile_buckets(s, y, min_names=5).iloc[0]
    # 10 joint names: 1,2,3,4,5,5,6,7,8,9 → average ranks; the two 5s share rank 5.5
    assert np.isnan(b["T10"]) and np.isnan(b["T09"])
    assert b["T04"] == b["T11"] == 3            # tied → same bucket (ceil(5.5 * 5 / 10) = 3)
    assert list(b[["T00", "T01", "T02", "T03", "T05", "T06", "T07", "T08"]]) == \
        [1, 1, 2, 2, 4, 4, 5, 5]
    # too few joint names → the whole date is skipped
    assert quantile_buckets(s, y, min_names=11).isna().all(axis=None)


def test_discrete_signal_top_bottom_and_spread():
    s = _wide([[1, 1, 1, -1, -1, -1, 1, -1, 1, 1]])
    y = _wide([[0.02, 0.02, 0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.02, 0.02]])
    b = quantile_buckets(s, y, min_names=5)
    r = bucket_returns(b, y)
    assert spread_series(b, r).iloc[0] == pytest.approx(0.02)
    assert top_membership(b).iloc[0].sum() == 6
    # all tied: one bucket, no spread
    flat = quantile_buckets(_wide([[1.0] * 10]), y, min_names=5)
    assert np.isnan(spread_series(flat, bucket_returns(flat, y)).iloc[0])


def test_turnover_and_costs():
    m = pd.DataFrame([[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 1, 0]], dtype=bool,
                     index=pd.bdate_range("2024-01-01", periods=3))
    np.testing.assert_allclose(turnover(m), [1.0, 0.5, 0.0])
    y = pd.DataFrame([[0.01, 0.03, 0.0, 0.0]] * 3, index=m.index, columns=m.columns)
    out = long_only_costs(m, y, horizon=1, cost=0.01)
    gross = np.array([0.02, 0.005, 0.005])
    net = gross - np.array([1.0, 0.5, 0.0]) * 0.01
    assert out["gross_ann"] == pytest.approx(gross.mean() * 252)
    assert out["net_ann"] == pytest.approx(net.mean() * 252)
    assert out["universe_ann"] == pytest.approx(0.01 * 252)
    assert out["turnover"] == pytest.approx(0.25)


def test_daily_ic_matches_pandas_spearman():
    rng = np.random.default_rng(5)
    s = _wide(rng.normal(size=(4, 15)), 4)
    y = _wide(rng.normal(size=(4, 15)), 4)
    s.iloc[1, :3] = np.nan
    ic = daily_ic(s, y, min_names=10)
    for i in range(4):
        ok = s.iloc[i].notna() & y.iloc[i].notna()
        exp = s.iloc[i][ok].rank().corr(y.iloc[i][ok].rank())
        assert ic.iloc[i] == pytest.approx(exp)
    assert daily_ic(s, y, min_names=14).isna().iloc[1]


# ──────────────────────────────────────────────────────────
# Walk-forward splits
# ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("embargo", [0, 5, 20])
def test_walk_forward_purge_embargo(embargo):
    cal = pd.bdate_range("2019-03-01", "2024-08-15")
    splits = walk_forward_splits(cal, embargo=embargo)
    assert len(splits) >= 8
    first_train = splits[0].train[0]
    prev_test_end = None
    for sp in splits:
        assert sp.train[0] == first_train                       # expanding
        pos_last_train = cal.get_loc(sp.train[-1])
        pos_first_test = cal.get_loc(sp.test[0])
        assert pos_first_test - pos_last_train - 1 == embargo   # exactly the gap
        forbidden = set(sp.train) | set(cal[pos_last_train + 1: pos_last_train + 1 + embargo])
        assert not forbidden & set(sp.test)                     # no test date in train + embargo
        assert sp.test[0].month in (1, 7) and sp.test[0].day <= 7
        assert (sp.test[-1] - sp.test[0]).days < 186
        if prev_test_end is not None:
            assert sp.test[0] > prev_test_end
        prev_test_end = sp.test[-1]
        assert len(sp.train) >= 252
    assert oos_dates(splits).is_monotonic_increasing


def test_walk_forward_too_short():
    assert walk_forward_splits(pd.bdate_range("2024-01-01", periods=200), embargo=20) == []


# ──────────────────────────────────────────────────────────
# End to end on a synthetic panel: oracle and random signals
# ──────────────────────────────────────────────────────────

N_RANDOM = 10


@pytest.fixture(scope="module")
def synthetic_eval():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2021-01-01", "2024-06-30")
    tickers = [f"S{i:02d}" for i in range(30)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    n = len(idx)
    labels = pd.DataFrame({c: rng.normal(0, 0.02, n)
                           for c in label_columns((1, 5), PRICE_STEMS)}, index=idx)
    # total return = price without dividends (task 200); same random stream as before
    labels = labels.assign(**{c.replace(p, t, 1): labels[c] for p, t in zip(PRICE_STEMS, TR_STEMS)
                              for c in label_columns((1, 5), (p,))})
    features = pd.DataFrame({
        "TECH_SCORE": rng.normal(size=n),
        "ORACLE": labels["FWD_TR_RET_5"].to_numpy(),      # the primary (total-return) label
        **{f"RANDOM_{k}": rng.normal(size=n) for k in range(N_RANDOM)},
        "REGIME": rng.integers(0, 2, n).astype(float),
        "REGIME_CONFIDENCE": rng.uniform(size=n),
    }, index=idx)
    table, splits = evaluate(features, labels, (1, 5))
    return table, splits


def test_oracle_signal_ic_is_one(synthetic_eval):
    table, splits = synthetic_eval
    h = headline(table).set_index(["signal", "horizon"])
    assert len(splits) >= 3
    assert h.loc[("ORACLE", 5), "ic_mean"] == pytest.approx(1.0)
    assert h.loc[("ORACLE", 5), "ic_hit"] == 1.0
    assert bool(h.loc[("ORACLE", 5), "promising"])
    assert h.loc[("ORACLE", 5), "spread_mean"] > 0.03
    assert h.loc[("ORACLE", 5), "monotonicity"] == pytest.approx(1.0)


def test_random_signal_ic_is_zero(synthetic_eval):
    table, _ = synthetic_eval
    h = headline(table).set_index(["signal", "horizon"])
    for hz in (1, 5):
        r = h.loc[("RANDOM_5", hz)]
        assert abs(r["ic_mean"]) < 0.02
        assert abs(r["ic_t_nw"]) < 2
    # all ten nulls: ~5% of the 20 tests are expected to reach |t| >= 2 by chance
    rand = h[h.index.get_level_values("signal").str.startswith("RANDOM_")]
    assert len(rand) == 2 * N_RANDOM
    assert (rand["ic_mean"].abs() < 0.03).all()
    assert rand["ic_t_nw"].abs().median() < 1.5
    assert (rand["ic_t_nw"].abs() >= 2).sum() <= 2
    assert not rand["promising"].any()                      # BH keeps the nulls out


def test_evaluate_rows_and_report(synthetic_eval):
    from investbot.evaluation.report import SURVIVORSHIP_CAVEAT, RunInfo, render_report
    table, splits = synthetic_eval
    subsets = set(table["subset"])
    assert {"oos", "all", "regime:range", "regime:trend"} <= subsets
    assert any(s.startswith("block:") for s in subsets)
    assert any(s.startswith("year:") for s in subsets)
    assert set(table["label"]) == {"open", "close", "price"}
    head = headline(table)
    assert head["ic_q"].notna().all() and (head["ic_q"] >= head["ic_p"] - 1e-12).all()
    # OOS rows only use test-block dates
    oos = headline(table)
    assert (oos["n_days"] <= len(oos_dates(splits))).all()
    md = render_report(table, splits, RunInfo(30, pd.Timestamp("2021-01-01"),
                                              pd.Timestamp("2024-06-30"), (1, 5), None),
                       pd.Timestamp("2024-07-01").date())
    assert md.count(SURVIVORSHIP_CAVEAT) >= 2
    assert "## Verdict" in md and "Benjamini" in md


def test_round_trip_cost_comes_from_config():
    from investbot.evaluation import metrics as m
    assert m.ROUND_TRIP_COST == pytest.approx(m.statutory_round_trip_cost() + 2 * m.SLIPPAGE_PER_SIDE)
    assert 0.0025 <= m.ROUND_TRIP_COST <= 0.0035          # master plan §2.8: ~0.25-0.35%


# ──────────────────────────────────────────────────────────
# Task 420: data-driven history caveat and the descriptive survivorship split
# ──────────────────────────────────────────────────────────

OLD_HISTORY_SENTENCE = "Five years of daily history leave about three years out of sample"


def _splits_between(start: str, end: str):
    cal = pd.bdate_range(start, end)
    return walk_forward_splits(cal, embargo=5)


def _run_info(**kw):
    from investbot.evaluation.report import RunInfo
    return RunInfo(12, pd.Timestamp("2000-01-03"), pd.Timestamp("2024-06-28"), (1, 5), None, **kw)


def test_history_caveat_is_built_from_the_splits():
    from investbot.evaluation.report import history_caveat
    short = _splits_between("2021-01-01", "2023-12-29")
    long = _splits_between("1995-01-02", "2024-06-28")
    for splits in (short, long):
        oos = oos_dates(splits)
        text = history_caveat(splits, _run_info(names_median=48.0))
        years = (oos.max() - oos.min()).days / 365.25
        assert f"{oos.min():%Y-%m-%d} → {oos.max():%Y-%m-%d}" in text
        assert f"{len(oos)} dates" in text and f"({years:.1f} years)" in text
        assert "median 48 names per OOS date" in text
    assert history_caveat(short, _run_info()) != history_caveat(long, _run_info())
    assert "names per OOS date" not in history_caveat(short, _run_info())      # unknown median
    assert "no out-of-sample dates" in history_caveat([], _run_info())


def test_report_caveat_replaces_the_fixed_history_sentence(synthetic_eval):
    from investbot.evaluation.report import history_caveat, render_report
    table, splits = synthetic_eval
    md = render_report(table, splits, _run_info(names_median=30.0), pd.Timestamp("2024-07-01").date())
    assert OLD_HISTORY_SENTENCE not in md
    assert history_caveat(splits, _run_info(names_median=30.0)) in md
    assert "## Caveats" in md


def test_oos_names_median_counts_finite_scores_per_oos_date():
    from investbot.evaluation.report import oos_names_median
    dates = pd.bdate_range("2024-01-01", periods=4, name="date")
    tickers = [f"T{i:02d}" for i in range(14)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    score = np.ones(len(idx))
    score[idx.get_level_values("date") == dates[1]] = np.nan
    score[(idx.get_level_values("date") == dates[1]) & (idx.get_level_values("ticker") < "T11")] = 1.0
    # per-date finite counts: 14, 11, 14, 14; dates[0] is not OOS
    f = pd.DataFrame({"TECH_SCORE": score}, index=idx)
    assert oos_names_median(f, dates[1:]) == 14.0
    assert oos_names_median(f, dates[1:2]) == 11.0
    f["TECH_SCORE"] = np.nan
    assert math.isnan(oos_names_median(f, dates))


def test_ic_complete_split_hand_values_around_the_cut():
    from investbot.evaluation.metrics import SURVIVORSHIP_KEYS, ic_complete_split
    idx = pd.to_datetime(["2019-03-26", "2019-03-27", "2019-03-28", "2019-03-29", "2019-04-01",
                          "2019-04-02", "2019-04-03"])
    ic = pd.Series([0.10, 0.20, np.nan, -0.30, 0.50, np.nan, 0.10], index=idx)
    out = ic_complete_split(ic, pd.Timestamp("2019-03-29").date())
    assert tuple(out) == SURVIVORSHIP_KEYS
    assert out["ic_mean_pre_complete"] == pytest.approx(0.15)       # (0.10 + 0.20) / 2, NaN skipped
    assert out["n_dates_pre_complete"] == 2
    assert out["ic_mean_post_complete"] == pytest.approx((-0.30 + 0.50 + 0.10) / 3)
    assert out["n_dates_post_complete"] == 3                        # the cut date itself is "post"


def test_ic_complete_split_edges_are_nan_with_zero_dates():
    from investbot.evaluation.metrics import ic_complete_split
    idx = pd.bdate_range("2020-01-01", periods=5)
    ic = pd.Series([0.1, 0.2, 0.3, np.nan, 0.4], index=idx)
    all_post = ic_complete_split(ic, pd.Timestamp("2019-01-01").date())
    assert math.isnan(all_post["ic_mean_pre_complete"]) and all_post["n_dates_pre_complete"] == 0
    assert all_post["ic_mean_post_complete"] == pytest.approx(0.25)
    assert all_post["n_dates_post_complete"] == 4
    all_pre = ic_complete_split(ic, pd.Timestamp("2021-01-01").date())
    assert math.isnan(all_pre["ic_mean_post_complete"]) and all_pre["n_dates_post_complete"] == 0
    assert all_pre["ic_mean_pre_complete"] == pytest.approx(0.25) and all_pre["n_dates_pre_complete"] == 4
    empty = ic_complete_split(ic.iloc[:0], pd.Timestamp("2020-01-03").date())
    assert math.isnan(empty["ic_mean_pre_complete"]) and empty["n_dates_post_complete"] == 0


@pytest.fixture(scope="module")
def split_tables():
    from investbot.evaluation.metrics import SURVIVORSHIP_KEYS
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2021-01-01", "2023-06-30")
    idx = pd.MultiIndex.from_product([dates, [f"S{i:02d}" for i in range(14)]],
                                     names=["date", "ticker"])
    n = len(idx)
    labels = pd.DataFrame({c: rng.normal(0, 0.02, n)
                           for c in label_columns((1, 5), PRICE_STEMS)}, index=idx)
    labels = labels.assign(**{c.replace(p, t, 1): labels[c] for p, t in zip(PRICE_STEMS, TR_STEMS)
                              for c in label_columns((1, 5), (p,))})
    features = pd.DataFrame({"TECH_SCORE": rng.normal(size=n),
                             "ORACLE": labels["FWD_TR_RET_5"].to_numpy(),
                             "REGIME": rng.integers(0, 2, n).astype(float),
                             "REGIME_CONFIDENCE": rng.uniform(size=n)}, index=idx)
    cut = pd.Timestamp("2022-10-03").date()
    kw = dict(signals=["TECH_SCORE", "ORACLE"], robust=False)
    table, splits = evaluate(features, labels, (1, 5), complete_from=cut, **kw)
    plain, _ = evaluate(features, labels, (1, 5), complete_from=None, splits=splits, **kw)
    return table, plain, splits, cut, SURVIVORSHIP_KEYS


def test_split_columns_come_last_and_only_on_headline_rows(split_tables):
    table, plain, _, _, keys = split_tables
    assert list(table.columns[-4:]) == list(keys)
    assert list(table.columns[:-4]) == list(plain.columns[:-4])
    head = table[(table["label"] == "open") & (table["subset"] == "oos")]
    rest = table.drop(head.index)
    assert head[list(keys)].notna().all(axis=None)
    assert rest[list(keys)].isna().all(axis=None)
    assert plain[list(keys)].isna().all(axis=None)


def test_existing_columns_are_bit_identical_with_the_split(split_tables):
    table, plain, _, _, keys = split_tables
    old = [c for c in table.columns if c not in keys]
    pd.testing.assert_frame_equal(table[old], plain[old], check_exact=True)


def test_split_means_match_the_row_ic_series(split_tables):
    table, _, splits, cut, _ = split_tables
    head = headline(table).set_index(["signal", "horizon"])
    assert (head["n_dates_pre_complete"] + head["n_dates_post_complete"] == head["n_days"]).all()
    assert (head["n_dates_pre_complete"] > 0).all() and (head["n_dates_post_complete"] > 0).all()
    pooled = ((head["ic_mean_pre_complete"] * head["n_dates_pre_complete"]
               + head["ic_mean_post_complete"] * head["n_dates_post_complete"])
              / head["n_days"])
    np.testing.assert_allclose(pooled, head["ic_mean"], atol=1e-12)
    assert splits[0].test[0] < pd.Timestamp(cut) <= splits[-1].test[-1]


def test_split_with_all_oos_dates_on_one_side(split_tables):
    table, _, splits, _, _ = split_tables
    pre = pd.Timestamp("2030-01-01").date()
    post = pd.Timestamp("2000-01-01").date()
    # re-use the stored OOS ICs through the public function instead of re-running evaluate
    from investbot.evaluation.metrics import ic_complete_split
    ic = pd.Series(np.linspace(-0.1, 0.1, len(oos_dates(splits))), index=oos_dates(splits))
    a, b = ic_complete_split(ic, pre), ic_complete_split(ic, post)
    assert a["n_dates_post_complete"] == 0 and math.isnan(a["ic_mean_post_complete"])
    assert b["n_dates_pre_complete"] == 0 and math.isnan(b["ic_mean_pre_complete"])


def test_survivorship_section_in_the_report(split_tables):
    from investbot.evaluation.report import render_report, survivorship_rows
    table, _, splits, cut, _ = split_tables
    md = render_report(table, splits, _run_info(complete_from=cut, names_median=14.0),
                       pd.Timestamp("2023-07-01").date())
    assert "## Survivorship split (descriptive)" in md
    assert md.index("## IC inference robustness") < md.index("## Survivorship split") \
        < md.index("## Top ")
    sec = md.split("## Survivorship split (descriptive)")[1].split("\n## ")[0]
    assert "survivorship bias" in sec and "no test is run" in sec and str(cut) in sec
    rows = survivorship_rows(headline(table))
    assert {"TECH_SCORE"} <= set(rows["signal"])
    assert (rows[rows["signal"] == "TECH_SCORE"]["horizon"].tolist() == [1, 5])
    for _, r in rows.iterrows():
        same = np.sign(r["ic_mean_pre_complete"]) == np.sign(r["ic_mean_post_complete"])
        assert r["_same_sign"] == ("yes" if same else "no")
