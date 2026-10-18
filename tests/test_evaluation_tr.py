"""
Total-return labels, split-safe periods and the run's reproducibility setup (task 200).

Synthetic bars and a throwaway SQLite file; no network.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from investbot.evaluation.labels import (
    HORIZONS,
    PRICE_STEMS,
    TR_STEMS,
    forward_returns,
    label_columns,
)

# ──────────────────────────────────────────────────────────
# Hand-computed labels: 10 bars, a dividend of 2.0 on bar 4
# ──────────────────────────────────────────────────────────

OPEN = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], dtype=float)
CLOSE = OPEN + 0.5
DIV_BAR, DIV = 4, 2.0


def _ten_bars() -> tuple[pd.DataFrame, np.ndarray]:
    idx = pd.bdate_range("2026-01-05", periods=10, name="date")
    d = np.zeros(10)
    d[DIV_BAR] = DIV
    return pd.DataFrame({"open": OPEN, "close": CLOSE}, index=idx), d


def _expected(h: int) -> dict[str, np.ndarray]:
    """Straight from the definitions, one t at a time."""
    o, c, n = OPEN, CLOSE, 10

    def D(a, b):
        return DIV if a <= DIV_BAR <= b else 0.0

    out = {k: np.full(n, np.nan) for k in ("tr", "tr_cc", "tr_oo")}
    for t in range(n):
        if t + h < n:
            out["tr"][t] = (c[t + h] + D(t + 2, t + h)) / o[t + 1] - 1
            out["tr_cc"][t] = (c[t + h] + D(t + 1, t + h)) / c[t] - 1
        if t + 1 + h < n:
            out["tr_oo"][t] = (o[t + 1 + h] + D(t + 2, t + 1 + h)) / o[t + 1] - 1
    return out


@pytest.mark.parametrize("h", [1, 3])
def test_tr_labels_by_hand(h):
    bars, d = _ten_bars()
    y = forward_returns(bars, (h,), dividends=d)
    exp = _expected(h)
    np.testing.assert_allclose(y[f"FWD_TR_RET_{h}"], exp["tr"], rtol=0, atol=1e-15)
    np.testing.assert_allclose(y[f"FWD_TR_CC_RET_{h}"], exp["tr_cc"], rtol=0, atol=1e-15)
    np.testing.assert_allclose(y[f"FWD_TR_OO_RET_{h}"], exp["tr_oo"], rtol=0, atol=1e-15)


def test_tr_timing_entry_and_exit_on_the_ex_date():
    bars, d = _ten_bars()
    y = forward_returns(bars, (1, 3), dividends=d)
    o, c = OPEN, CLOSE
    # FWD_TR_RET: entry at the ex-date's open (t + 1 = 4) does not earn the dividend
    assert y["FWD_TR_RET_1"][3] == pytest.approx(c[4] / o[4] - 1)
    assert y["FWD_TR_RET_3"][3] == pytest.approx(c[6] / o[4] - 1)
    # ... entry the bar before (t + 1 = 3) does, for h = 3 (exit close[5]); h = 1 exits at
    # close[3], before the ex-date
    assert y["FWD_TR_RET_3"][2] == pytest.approx((c[5] + DIV) / o[3] - 1)
    assert y["FWD_TR_RET_1"][2] == pytest.approx(c[3] / o[3] - 1)
    # exit at the ex-date's close (t + h = 4) earns it
    assert y["FWD_TR_RET_3"][1] == pytest.approx((c[4] + DIV) / o[2] - 1)
    # FWD_TR_CC_RET: held from close[t]; ex-date t + 1 = 4 is earned, t = 4 is not
    assert y["FWD_TR_CC_RET_1"][3] == pytest.approx((c[4] + DIV) / c[3] - 1)
    assert y["FWD_TR_CC_RET_1"][4] == pytest.approx(c[5] / c[4] - 1)
    assert y["FWD_TR_CC_RET_3"][1] == pytest.approx((c[4] + DIV) / c[1] - 1)
    # FWD_TR_OO_RET: exit at the ex-date's open (t + 1 + h = 4) earns it, entry there does not
    assert y["FWD_TR_OO_RET_1"][2] == pytest.approx((o[4] + DIV) / o[3] - 1)
    assert y["FWD_TR_OO_RET_3"][0] == pytest.approx((o[4] + DIV) / o[1] - 1)
    assert y["FWD_TR_OO_RET_1"][3] == pytest.approx(o[5] / o[4] - 1)
    assert y["FWD_TR_OO_RET_3"][3] == pytest.approx(o[7] / o[4] - 1)
    # a period ending before the ex-date never sees it
    assert y["FWD_TR_OO_RET_1"][1] == pytest.approx(o[3] / o[2] - 1)
    # the dividend never touches the price columns
    for k in ("FWD_RET_3", "FWD_CC_RET_3", "FWD_OO_RET_3"):
        assert np.array_equal(y[k], forward_returns(bars, (3,))[k], equal_nan=True)


# ──────────────────────────────────────────────────────────
# No dividends: TR == price, pre-existing columns bit-identical
# ──────────────────────────────────────────────────────────

def _random_bars(n: int = 300, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    vol = rng.integers(100_000, 1_000_000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol},
                        index=pd.bdate_range("2023-01-02", periods=n, name="date"))


def _pre_200_forward_returns(bars: pd.DataFrame, horizons) -> dict[str, np.ndarray]:
    """The labels.py implementation before task 200, verbatim."""
    def lead(x, k):
        out = np.full(x.shape, np.nan)
        if k < x.size:
            out[: x.size - k] = x[k:]
        return out
    open_ = bars["open"].to_numpy(dtype=np.float64)
    close = bars["close"].to_numpy(dtype=np.float64)
    entry = lead(open_, 1)
    out = {}
    for h in horizons:
        out[f"FWD_RET_{h}"] = lead(close, h) / entry - 1.0
    for h in horizons:
        out[f"FWD_CC_RET_{h}"] = lead(close, h) / close - 1.0
    for h in horizons:
        out[f"FWD_OO_RET_{h}"] = lead(open_, 1 + h) / entry - 1.0
    return out


@pytest.mark.parametrize("dividends", [None, "zeros"])
def test_no_dividends_tr_equals_price_and_old_columns_are_bit_identical(dividends):
    bars = _random_bars()
    bars.iloc[100, bars.columns.get_loc("close")] = np.nan      # a NaN stays a NaN
    d = None if dividends is None else np.zeros(len(bars))
    y = forward_returns(bars, HORIZONS, dividends=d)
    assert list(y) == label_columns(HORIZONS)
    old = _pre_200_forward_returns(bars, HORIZONS)
    for k, v in old.items():
        assert np.array_equal(y[k], v, equal_nan=True), k
    for p, t in zip(PRICE_STEMS, TR_STEMS):
        for h in HORIZONS:
            assert np.array_equal(y[f"{t}_{h}"], y[f"{p}_{h}"], equal_nan=True)


# ──────────────────────────────────────────────────────────
# Split safety: a flagged bar NaNs exactly the spanning periods
# ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("h", [1, 3])
def test_split_flag_nans_exactly_the_spanning_periods(h):
    bars, d = _ten_bars()
    i = 6
    flag = np.zeros(10, dtype=bool)
    flag[i] = True
    clean = forward_returns(bars, (h,), dividends=d)
    y = forward_returns(bars, (h,), dividends=d, invalid_from=flag)
    # the period contains the close[i-1] -> open[i] boundary iff i lies in its bar window
    windows = {"RET": (2, h), "CC_RET": (1, h), "OO_RET": (2, 1 + h)}
    for stem, (a, b) in windows.items():
        spans = np.array([t + a <= i <= t + b for t in range(10)])
        for col in (f"FWD_{stem}_{h}", f"FWD_TR_{stem}_{h}"):
            assert np.isnan(y[col][spans]).all(), col
            assert np.array_equal(y[col][~spans], clean[col][~spans], equal_nan=True), col
            assert np.isfinite(clean[col][spans]).all()       # it really was a value before


def test_split_flag_by_hand_h1():
    bars, _ = _ten_bars()
    flag = np.zeros(10, dtype=bool)
    flag[6] = True
    y = forward_returns(bars, (1,), invalid_from=flag)
    # FWD_RET_1 at t = 5 is open[6] -> close[6]: same side of the boundary, kept
    assert np.isfinite(y["FWD_RET_1"]).sum() == 9
    # close-to-close over close[5] -> close[6] spans it
    assert np.isnan(y["FWD_CC_RET_1"][5]) and np.isfinite(y["FWD_CC_RET_1"][[4, 6]]).all()
    # open[5] -> open[6] (t = 4) spans it
    assert np.isnan(y["FWD_OO_RET_1"][4]) and np.isfinite(y["FWD_OO_RET_1"][[3, 5]]).all()


def test_split_flags_from_detect_unadjusted_splits():
    from investbot.data.quality import detect_unadjusted_splits
    from investbot.evaluation.panel import label_inputs, nan_periods, split_flags

    bars = _random_bars(60)
    bars.iloc[:30, :4] *= 2.0                       # unadjusted 1:2 split on bar 30
    actions = pd.DataFrame({"action_type": ["split", "dividend"], "value": [2.0, 1.5]},
                           index=pd.DatetimeIndex([bars.index[30], bars.index[40]],
                                                  name="ex_date"))
    issues = detect_unadjusted_splits(bars, actions, ticker="X")
    assert len(issues) == 1
    flag = split_flags(bars, issues)
    assert flag.sum() == 1 and flag[30]
    d, inv, iss = label_inputs(bars, actions, "X")
    assert np.array_equal(inv, flag) and iss == issues and d[40] == 1.5 and d.sum() == 1.5
    keep = np.ones(len(bars), dtype=bool)
    # FWD_TR_RET_h spans bar 30 for t in 28..29 (h=1: none), so h=1,5,20 give 0 + 4 + 19
    assert nan_periods(bars, (1, 5, 20), keep, issues[0], d) == 0 + 4 + 19
    y = forward_returns(bars, (1,), invalid_from=flag)
    assert np.isnan(y["FWD_CC_RET_1"][29])          # the fake -50% day is gone
    assert np.nanmin(y["FWD_CC_RET_1"]) > -0.2


# ──────────────────────────────────────────────────────────
# Separation and truncation invariance with dividends in the DB
# ──────────────────────────────────────────────────────────

def _db_with_actions(path, n: int = 330):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from investbot.data.models import (
        Asset,
        AssetClass,
        Base,
        CorporateAction,
        Exchange,
        PriceDaily,
    )

    engine = create_engine(f"sqlite:///{path}", future=True)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    ac = AssetClass(name="equity")
    ex = Exchange(asset_class=ac, code="NSE", name="NSE", currency="INR")
    s.add_all([ac, ex])
    s.flush()
    out = {}
    for k, ticker in enumerate(["AAA.NS", "BBB.NS"]):
        a = Asset(exchange_id=ex.id, ticker=ticker, name=ticker, asset_type="stock",
                  is_active=True)
        s.add(a)
        s.flush()
        bars = _random_bars(n, seed=k + 1)
        if ticker == "BBB.NS":
            bars.iloc[:250, :4] *= 2.0              # unadjusted split on bar 250
        s.add_all([PriceDaily(asset_id=a.id, date=d.date(), open=r.open, high=r.high,
                              low=r.low, close=r.close, adj_close=r.close,
                              volume=int(r.volume)) for d, r in bars.iterrows()])
        out[ticker] = bars
    ids = {t: i + 1 for i, t in enumerate(out)}
    s.add_all([
        CorporateAction(asset_id=ids["AAA.NS"], ex_date=out["AAA.NS"].index[300].date(),
                        action_type="dividend", value=3.0, source="test"),
        CorporateAction(asset_id=ids["AAA.NS"], ex_date=out["AAA.NS"].index[50].date(),
                        action_type="dividend", value=1.0, source="test"),
        CorporateAction(asset_id=ids["BBB.NS"], ex_date=out["BBB.NS"].index[250].date(),
                        action_type="split", value=2.0, source="test"),
    ])
    s.commit()
    return s, out


@pytest.fixture(scope="module")
def tr_panel(tmp_path_factory):
    pytest.importorskip("talib")
    from investbot.evaluation.panel import build_dataset

    s, bars = _db_with_actions(tmp_path_factory.mktemp("tr") / "tr.db")
    info: dict = {}
    try:
        start = bars["AAA.NS"].index[100].date()
        feats, labels = build_dataset(start=start, session=s, universe="active",
                                      label_info=info)
    finally:
        s.close()
    return feats, labels, bars, info


def test_dividends_reach_labels_not_features(tr_panel):
    from investbot.evaluation.panel import asset_features
    feats, labels, bars, info = tr_panel
    assert not set(feats.columns) & set(labels.columns)
    assert not any(c.startswith("FWD_") for c in feats.columns)
    a = labels.xs("AAA.NS", level="ticker")
    ex = bars["AAA.NS"].index[300]
    # CC h=1 on the bar before the ex-date earns the 3.0 dividend
    t = a.index.get_loc(ex) - 1
    c = bars["AAA.NS"]["close"]
    assert a["FWD_TR_CC_RET_1"].iloc[t] == pytest.approx((c.loc[ex] + 3.0) / c.iloc[300 - 1] - 1)
    assert a["FWD_CC_RET_1"].iloc[t] == pytest.approx(c.loc[ex] / c.iloc[300 - 1] - 1)
    # features are those of the price bars alone (the dividend changes nothing)
    f = feats.xs("AAA.NS", level="ticker")
    ref = asset_features(bars["AAA.NS"]).loc[f.index]
    pd.testing.assert_frame_equal(f, ref)
    # label_info: the 1.0 dividend (bar 50) is before the panel start, the 3.0 inside
    assert info["AAA.NS"] == {"dividend_rows": 1, "splits": []}


def test_split_in_panel_nans_spanning_labels(tr_panel):
    _, labels, bars, info = tr_panel
    (issue, n_nan), = info["BBB.NS"]["splits"]
    assert issue.ex_date == bars["BBB.NS"].index[250].date() and issue.factor == 2.0
    assert n_nan == 0 + 4 + 19                     # FWD_TR_RET_h over h = 1, 5, 20
    b = labels.xs("BBB.NS", level="ticker")
    pos = b.index.get_loc(bars["BBB.NS"].index[250])
    assert np.isnan(b["FWD_CC_RET_1"].iloc[pos - 1])
    assert np.isnan(b["FWD_TR_RET_20"].iloc[pos - 20: pos - 1]).all()   # t = 230 .. 248
    assert np.isfinite(b["FWD_TR_RET_20"].iloc[pos - 1])      # entry at open[250]: same scale
    assert np.isfinite(b["FWD_TR_RET_20"].iloc[pos - 21])     # exit close[249]: same scale
    assert np.nanmin(b["FWD_CC_RET_1"].to_numpy()) > -0.2


@pytest.mark.parametrize("cut", [200, 280])
def test_tr_labels_only_use_bars_and_dividends_up_to_their_window(cut):
    """A TR label at t uses bars <= t+h+1 and dividends on bars <= t+h+1: truncating the
    series (and its dividends) after that changes nothing; features stay truncation-invariant
    (covered in test_evaluation.py) because dividends never enter them."""
    bars = _random_bars(330)
    d = np.zeros(330)
    d[[150, 205, 285]] = [2.0, 1.0, 4.0]
    full = forward_returns(bars, HORIZONS, dividends=d)
    trunc = forward_returns(bars.iloc[: cut + 1], HORIZONS, dividends=d[: cut + 1])
    for h in HORIZONS:
        t_ok = cut - h - 1                      # last t whose OO label fits in the truncation
        for col in (f"FWD_TR_RET_{h}", f"FWD_TR_CC_RET_{h}", f"FWD_TR_OO_RET_{h}"):
            assert np.array_equal(trunc[col][: t_ok + 1], full[col][: t_ok + 1]), col


# ──────────────────────────────────────────────────────────
# Report: coverage, split issues, reproducibility, dividend effect
# ──────────────────────────────────────────────────────────

def test_db_fingerprint(tmp_path):
    from investbot.evaluation.loader import db_fingerprint
    s, bars = _db_with_actions(tmp_path / "f.db", n=310)
    try:
        fp = db_fingerprint(s)
    finally:
        s.close()
    assert fp == {"db_max_date": bars["AAA.NS"].index[-1].date(), "db_price_rows": 620,
                  "db_action_rows": 3}


def test_git_commit_falls_back_to_unknown(tmp_path):
    from investbot.evaluation.report import git_commit
    assert git_commit(tmp_path) == "unknown"            # not a repository
    head = git_commit()
    assert head == "unknown" or (4 <= len(head) <= 40 and head.isalnum())


@pytest.fixture(scope="module")
def e2e_tr(tmp_path_factory):
    """12 priced names, dividends on some, one unadjusted split; a real run."""
    pytest.importorskip("talib")
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from investbot.data.models import (
        Asset,
        AssetClass,
        Base,
        CorporateAction,
        Exchange,
        PriceDaily,
    )
    from investbot.evaluation.report import run_evaluation

    root = tmp_path_factory.mktemp("e2e_tr")
    engine = create_engine(f"sqlite:///{root / 'e.db'}", future=True)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    ac = AssetClass(name="equity")
    ex = Exchange(asset_class=ac, code="NSE", name="NSE", currency="INR")
    s.add_all([ac, ex])
    s.flush()
    n = 620
    for k in range(12):
        a = Asset(exchange_id=ex.id, ticker=f"S{k:02d}.NS", name=str(k), asset_type="stock",
                  is_active=True)
        s.add(a)
        s.flush()
        bars = _random_bars(n, seed=k + 1)
        if k == 0:
            bars.iloc[:500, :4] *= 2.0
            s.add(CorporateAction(asset_id=a.id, ex_date=bars.index[500].date(),
                                  action_type="split", value=2.0, source="test"))
        if k < 6:
            for i in (300, 550):
                s.add(CorporateAction(asset_id=a.id, ex_date=bars.index[i].date(),
                                      action_type="dividend", value=2.0, source="test"))
        s.add_all([PriceDaily(asset_id=a.id, date=d.date(), open=r.open, high=r.high,
                              low=r.low, close=r.close, adj_close=r.close,
                              volume=int(r.volume)) for d, r in bars.iterrows()])
    s.commit()
    try:
        out = run_evaluation((1, 5), session=s, universe="active", report_dir=root,
                             csv_dir=root, report_date=date(2026, 10, 19))
    finally:
        s.close()
    return out


def test_run_evaluation_records_tr_setup(e2e_tr):
    from investbot.scoring.technical.scorer import SCORER_VERSION
    report, csv, table, info = e2e_tr
    assert info.n_dividend_names == 6 and info.n_dividend_rows == 12
    ex_date = pd.bdate_range("2023-01-02", periods=620)[500].date()
    assert [(t, d, f) for t, d, f, _ in info.split_issues] == [("S00.NS", ex_date, 2.0)]
    assert info.split_issues[0][3] == 0 + 4                       # h = 1, 5
    assert info.scorer_version == SCORER_VERSION
    assert info.db_price_rows == 12 * 620 and info.db_action_rows == 13
    md = report.read_text()
    assert "## Dividend effect" in md
    assert "FWD_TR_RET_h" in md and "FWD_TR_OO_RET_h" in md
    assert f"`SCORER_VERSION` {SCORER_VERSION}" in md
    assert "S00.NS" in md and "corporate_actions` is empty" not in md
    assert set(table["label"]) == {"open", "close", "price"}
    out = pd.read_csv(csv)
    assert {"open", "price"} <= set(out["label"])


def test_dividend_effect_table_and_empty_coverage_warning(e2e_tr):
    from investbot.evaluation.report import RunInfo, dividend_effect, render_report
    _, _, table, info = e2e_tr
    eff = dividend_effect(table, ["TECH_SCORE"])
    assert set(eff["horizon"]) == {1, 5}
    assert {"ic_mean_price", "ic_mean_tr", "held_cagr_net_price", "held_cagr_net_tr"} <= set(eff)
    # dividends only add to returns: the universe benchmark is higher under TR
    assert (eff["held_cagr_universe_tr"] >= eff["held_cagr_universe_price"]).all()
    empty = RunInfo(info.n_assets, info.first_date, info.last_date, info.horizons, None)
    md = render_report(table, [], empty, date(2026, 10, 19))
    assert "corporate_actions` is empty" in md
    assert md.count("total return equals price") >= 2
