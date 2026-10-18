"""
Tests for the point-in-time index universe in the evaluation harness (task 170).

Synthetic panels plus a throwaway SQLite file; no network.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from investbot.evaluation.evaluate import evaluate, headline
from investbot.evaluation.labels import HORIZONS, PRICE_STEMS, TR_STEMS, label_columns

START = "2023-01-02"
N_BARS = 330


def _bars(n: int, seed: int, start: str = START) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    vol = rng.integers(100_000, 1_000_000, n).astype(float)
    idx = pd.bdate_range(start, periods=n, name="date")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": vol}, index=idx)


def _make_db(path, assets: dict[str, tuple[int, bool]],
             stints: list[tuple[str, date, date | None]] = ()):
    """assets: ticker -> (n_bars, is_active); stints: NIFTY50 (ticker, from, to)."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from investbot.data.models import (
        Asset,
        AssetClass,
        Base,
        Exchange,
        IndexMembership,
        PriceDaily,
    )

    engine = create_engine(f"sqlite:///{path}", future=True)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    ac = AssetClass(name="equity")
    ex = Exchange(asset_class=ac, code="NSE", name="NSE", currency="INR")
    s.add_all([ac, ex])
    s.flush()
    ids = {}
    for k, (ticker, (n, active)) in enumerate(assets.items()):
        a = Asset(exchange_id=ex.id, ticker=ticker, name=ticker, asset_type="stock",
                  is_active=active)
        s.add(a)
        s.flush()
        ids[ticker] = a.id
        bars = _bars(n, seed=k + 1) if n else None
        if n:
            s.add_all([PriceDaily(asset_id=a.id, date=d.date(), open=r.open, high=r.high,
                                  low=r.low, close=r.close, adj_close=r.close,
                                  volume=int(r.volume)) for d, r in bars.iterrows()])
    for t, f, to in stints:
        s.add(IndexMembership(index_code="NIFTY50", asset_id=ids[t], effective_from=f,
                              effective_to=to, source="test", verified=True))
    s.commit()
    return s


def synthetic_panel(n_dates: int = 700, n_names: int = 15, seed: int = 3):
    """A fixed (features, labels) panel with a weak true signal, horizons (1, 5)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates, name="date")
    tickers = [f"T{i:02d}.NS" for i in range(n_names)]
    shape = (n_dates, n_names)
    noise = rng.normal(0, 0.01, shape)
    alpha = rng.normal(0, 1, shape)
    r1 = 0.002 * alpha + noise
    feats = {"TECH_SCORE": alpha + rng.normal(0, 3, shape),
             "BASE_MOM_20": rng.normal(0, 1, shape),
             "REGIME": rng.integers(0, 3, shape).astype(float),
             "REGIME_CONFIDENCE": rng.uniform(0, 1, shape)}
    feats["TECH_SCORE"][:30] = np.nan                       # warmup
    labels = {}
    for h in (1, 5):
        labels[f"FWD_RET_{h}"] = r1 * np.sqrt(h)
        labels[f"FWD_CC_RET_{h}"] = r1 * np.sqrt(h) + rng.normal(0, 0.002, shape)
        labels[f"FWD_OO_RET_{h}"] = r1 * np.sqrt(h) + rng.normal(0, 0.002, shape)
    for stem in ("RET", "CC_RET", "OO_RET"):      # total return = price without dividends
        for h in (1, 5):
            labels[f"FWD_TR_{stem}_{h}"] = labels[f"FWD_{stem}_{h}"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    f = pd.DataFrame({k: v.reshape(-1) for k, v in feats.items()}, index=idx)
    y = pd.DataFrame({k: v.reshape(-1) for k, v in labels.items()}, index=idx)
    return f, y


# ──────────────────────────────────────────────────────────
# Regression: universe="active" is the pre-170 behaviour
# ──────────────────────────────────────────────────────────

# Fingerprints computed with the code at d49d9ac (before task 170) on the same inputs.
GOLDEN_EVAL = {
    "n_rows": 44,
    "sum": 36851.852018090554,
    ("TECH_SCORE", 1): (0.03920480813684698, 0.14954880158006711, -0.3630883772882971,
                        0.01925441773838968),
    ("TECH_SCORE", 5): (0.03920480813684698, 0.14954880158006711, 0.0247696994194202,
                        0.01925441773838968),
}
GOLDEN_PANEL = {"shape": [580, 44], "feat_sum": -2967.6614279902037,
                "label_sum": -35.21085586486308}


def eval_fingerprint(table: pd.DataFrame) -> dict:
    num = table.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    h = headline(table).set_index(["signal", "horizon"])
    out = {"n_rows": len(table), "sum": float(np.nansum(np.where(np.isfinite(num), num, 0.0)))}
    for hz in (1, 5):
        r = h.loc[("TECH_SCORE", hz)]
        out[("TECH_SCORE", hz)] = (float(r["ic_mean"]), float(r["icir"]),
                                   float(r["held_cagr_net"]), float(r["ic_q"]))
    return out


def panel_fingerprint(feats: pd.DataFrame, labels: pd.DataFrame) -> dict:
    f = feats.to_numpy(dtype=np.float64)
    # the pre-task-200 (price) label columns; the TR columns are checked separately
    y = labels[label_columns(HORIZONS, PRICE_STEMS)].to_numpy(dtype=np.float64)
    return {"shape": list(feats.shape),
            "feat_sum": float(np.nansum(np.where(np.isfinite(f), f, 0.0))),
            "label_sum": float(np.nansum(np.where(np.isfinite(y), y, 0.0)))}


def test_active_evaluation_is_unchanged():
    f, y = synthetic_panel()
    # primary="price" is the pre-task-200 table exactly (no "price" comparison rows)
    table, _ = evaluate(f, y, (1, 5), primary="price")
    got = eval_fingerprint(table)
    assert got["n_rows"] == GOLDEN_EVAL["n_rows"]
    assert got["sum"] == pytest.approx(GOLDEN_EVAL["sum"], rel=1e-10)
    for hz in (1, 5):
        assert got[("TECH_SCORE", hz)] == pytest.approx(GOLDEN_EVAL[("TECH_SCORE", hz)],
                                                       rel=1e-10)
    # the default (total return) has TR == price here, so its headline is the same numbers
    tr = eval_fingerprint(evaluate(f, y, (1, 5))[0])
    for hz in (1, 5):
        assert tr[("TECH_SCORE", hz)] == pytest.approx(GOLDEN_EVAL[("TECH_SCORE", hz)],
                                                      rel=1e-10)


def test_active_build_dataset_is_unchanged(tmp_path):
    pytest.importorskip("talib")
    from investbot.evaluation.panel import build_dataset

    s = _make_db(tmp_path / "a.db", {"AAA.NS": (300, True), "BBB.NS": (280, True),
                                      "OLD.NS": (300, False), "TINY.NS": (5, True)},
                 [("OLD.NS", date(2023, 1, 2), None)])
    try:
        feats, labels = build_dataset(session=s, universe="active")
    finally:
        s.close()
    assert "IN_UNIVERSE" not in feats.columns
    got = panel_fingerprint(feats, labels)
    assert got["shape"] == GOLDEN_PANEL["shape"]
    assert got["feat_sum"] == pytest.approx(GOLDEN_PANEL["feat_sum"], rel=1e-10)
    assert got["label_sum"] == pytest.approx(GOLDEN_PANEL["label_sum"], rel=1e-10)
    # no corporate_actions rows: every TR column equals its price column exactly
    for p, t in zip(PRICE_STEMS, TR_STEMS):
        for h in HORIZONS:
            np.testing.assert_array_equal(labels[f"{t}_{h}"].to_numpy(),
                                          labels[f"{p}_{h}"].to_numpy())


# ──────────────────────────────────────────────────────────
# Joining and leaving the index
# ──────────────────────────────────────────────────────────

J = date(2023, 9, 1)      # NEW.NS joins (first member session)
L = date(2023, 10, 2)     # GONE.NS's first session outside the index


@pytest.fixture(scope="module")
def join_leave(tmp_path_factory):
    pytest.importorskip("talib")
    from investbot.evaluation.panel import build_dataset

    s = _make_db(tmp_path_factory.mktemp("jl") / "jl.db",
                 {"AAA.NS": (N_BARS, True), "NEW.NS": (N_BARS, True),
                  "GONE.NS": (N_BARS, False), "NOPX.NS": (0, False)},
                 [("AAA.NS", date(2019, 1, 1), None), ("NEW.NS", J, None),
                  ("GONE.NS", date(2019, 1, 1), L - timedelta(days=1)),
                  ("NOPX.NS", date(2019, 1, 1), None)])
    try:
        feats, labels = build_dataset(session=s, universe="index", horizons=(1, 20))
        bars = {t: _bars(N_BARS, seed=k + 1) for k, t in enumerate(["AAA.NS", "NEW.NS",
                                                                     "GONE.NS"])}
    finally:
        s.close()
    return feats, labels, bars


def test_index_universe_tickers_and_flag(join_leave):
    feats, labels, _ = join_leave
    # past members are included whether active or not; a member without prices is skipped
    assert set(feats.index.get_level_values("ticker")) == {"AAA.NS", "NEW.NS", "GONE.NS"}
    assert feats["IN_UNIVERSE"].dtype == bool
    assert feats.index.equals(labels.index)
    new = feats.xs("NEW.NS", level="ticker")["IN_UNIVERSE"]
    assert not new[new.index < pd.Timestamp(J)].any() and new[new.index >= pd.Timestamp(J)].all()
    gone = feats.xs("GONE.NS", level="ticker")["IN_UNIVERSE"]
    assert gone[gone.index < pd.Timestamp(L)].all() and not gone[gone.index >= pd.Timestamp(L)].any()


def test_joiner_is_masked_before_join_and_keeps_full_history_warmup(join_leave):
    from investbot.evaluation.evaluate import _wide, apply_universe, universe_frame
    from investbot.evaluation.panel import asset_features

    feats, _, bars = join_leave
    dates = pd.DatetimeIndex(feats.index.get_level_values("date").unique()).sort_values()
    tickers = sorted(feats.index.get_level_values("ticker").unique())
    U = universe_frame(feats, dates, tickers)
    full = asset_features(bars["NEW.NS"])           # unmasked, full history
    for col in ("TECH_SCORE", "SIG_RVOL_CONFIRM", "BASE_MOM_20", "BASE_MOM_12_1"):
        S = apply_universe(_wide(feats[col], dates, tickers), U)["NEW.NS"]
        before = S.index < pd.Timestamp(J)
        assert S[before].isna().all()
        # from J on: exactly the full-history values (warmup not restarted)
        exp = full[col].reindex(S.index[~before])
        pd.testing.assert_series_equal(S[~before], exp, check_names=False, check_exact=True)
        assert np.isfinite(S[~before].to_numpy()).any()
    # the value on J itself is real (BASE_MOM_20 needs only 20 bars, the name has ~170 before J)
    j = pd.Timestamp(J)
    assert np.isfinite(apply_universe(_wide(feats["BASE_MOM_20"], dates, tickers), U).loc[j, "NEW.NS"])


def test_leaver_keeps_labels_that_reach_past_its_exit(join_leave):
    from investbot.evaluation.evaluate import _wide, apply_universe, universe_frame

    feats, labels, bars = join_leave
    dates = pd.DatetimeIndex(feats.index.get_level_values("date").unique()).sort_values()
    tickers = sorted(feats.index.get_level_values("ticker").unique())
    U = universe_frame(feats, dates, tickers)
    b = bars["GONE.NS"]
    Y = apply_universe(_wide(labels["FWD_RET_20"], dates, tickers), U)["GONE.NS"]
    last_in = b.index[b.index < pd.Timestamp(L)]
    for t in last_in[-5:]:                          # positions opened in the last member week
        i = b.index.get_loc(t)
        exp = b["close"].iloc[i + 20] / b["open"].iloc[i + 1] - 1
        assert b.index[i + 20] >= pd.Timestamp(L)   # the label reaches past the exit
        assert Y[t] == pytest.approx(exp)
    assert Y[Y.index >= pd.Timestamp(L)].isna().all()    # no new positions after the exit
    # the label panel itself is never masked
    raw = labels.xs("GONE.NS", level="ticker")["FWD_RET_20"]
    assert raw[raw.index >= pd.Timestamp(L)].iloc[:20].notna().all()


def test_evaluate_masks_signals_outside_the_universe():
    """A signal that is pure noise outside the universe and perfect inside it."""
    f, y = synthetic_panel(n_dates=700, n_names=14, seed=5)
    rng = np.random.default_rng(1)
    inside = pd.Series(rng.uniform(size=len(f)) < 0.8, index=f.index)
    f = f.assign(ORACLE=np.where(inside, y["FWD_RET_1"], rng.normal(size=len(f))),
                 IN_UNIVERSE=inside)
    t_index, splits = evaluate(f, y, (1,), signals=["ORACLE"])
    t_active, _ = evaluate(f.drop(columns="IN_UNIVERSE"), y, (1,), signals=["ORACLE"],
                           splits=splits)
    ic_i = headline(t_index).set_index("signal").loc["ORACLE", "ic_mean"]
    ic_a = headline(t_active).set_index("signal").loc["ORACLE", "ic_mean"]
    assert ic_i == pytest.approx(1.0)
    assert ic_a < 0.9
    # IN_UNIVERSE is a conditioning column, never a signal
    t_default, _ = evaluate(f, y, (1,))
    assert "IN_UNIVERSE" not in set(t_default["signal"])


# ──────────────────────────────────────────────────────────
# Membership is point in time
# ──────────────────────────────────────────────────────────

def _changes(rows):
    return pd.DataFrame(rows, columns=["effective_date", "ticker", "action", "source"])


TOY_CURRENT = ["A.NS", "B.NS", "D.NS", "E.NS"]
TOY_CHANGES = [
    (date(2020, 3, 2), "C.NS", "remove", "s1"), (date(2020, 3, 2), "D.NS", "add", "s1"),
    (date(2021, 6, 1), "B.NS", "remove", "s2"), (date(2021, 6, 1), "F.NS", "add", "s2"),
    (date(2022, 1, 3), "F.NS", "remove", "s3"), (date(2022, 1, 3), "B.NS", "add", "s3"),
    (date(2022, 9, 30), "G.NS", "remove", "s4"), (date(2022, 9, 30), "E.NS", "add", "s4"),
]
TOY_START = ["A.NS", "B.NS", "C.NS", "G.NS"]


def _toy_db(path, current, changes):
    from investbot.data.index_membership import reconstruct_intervals
    iv = reconstruct_intervals(current, _changes(changes), date(2023, 1, 1),
                               history_start=date(2019, 1, 1))
    tickers = sorted(set(iv["ticker"]))
    stints = [(r.ticker, r.effective_from,
               r.effective_to if pd.notna(r.effective_to) else None) for r in iv.itertuples()]
    return _make_db(path, {t: (0, t in current) for t in tickers}, stints), tickers


def _replay(start: list[str], changes, d: date) -> set[str]:
    members = set(start)
    for eff, t, action, _ in sorted(changes):
        if eff <= d:
            (members.add if action == "add" else members.discard)(t)
    return members


def test_membership_for_equals_forward_replay_of_past_changes(tmp_path):
    from investbot.evaluation.loader import membership_for, universe_tickers

    s, tickers = _toy_db(tmp_path / "m.db", TOY_CURRENT, TOY_CHANGES)
    try:
        dates = pd.bdate_range("2019-01-01", "2022-12-30")
        M = membership_for(s, "NIFTY50", tickers, dates)
        assert list(M.columns) == tickers and M.dtypes.eq(bool).all()
        for d in dates:
            assert set(M.columns[M.loc[d].to_numpy()]) == _replay(TOY_START, TOY_CHANGES,
                                                                   d.date()), d
        # any member in a window, active or not
        assert universe_tickers(s, "NIFTY50", date(2021, 7, 1), date(2021, 12, 31)) == \
            ["A.NS", "D.NS", "F.NS", "G.NS"]
        assert universe_tickers(s, "NIFTY50", date(2022, 10, 1), None) == TOY_CURRENT
        assert universe_tickers(s, "OTHER", None, None) == []
        assert not membership_for(s, "NIFTY50", ["ZZZ.NS"], dates[:3]).any().any()
    finally:
        s.close()


def test_membership_is_truncation_invariant(tmp_path):
    """A change effective after t never alters membership on t."""
    from investbot.evaluation.loader import membership_for

    later = date(2022, 6, 1)
    changes_b = [*TOY_CHANGES, (later, "A.NS", "remove", "s5"), (later, "H.NS", "add", "s5")]
    current_b = ["B.NS", "D.NS", "E.NS", "H.NS"]
    sa, ta = _toy_db(tmp_path / "a.db", TOY_CURRENT, TOY_CHANGES)
    sb, tb = _toy_db(tmp_path / "b.db", current_b, changes_b)
    try:
        cols = sorted(set(ta) | set(tb))
        dates = pd.bdate_range("2019-01-01", "2022-12-30")
        Ma = membership_for(sa, "NIFTY50", cols, dates)
        Mb = membership_for(sb, "NIFTY50", cols, dates)
        before = dates < pd.Timestamp(later)
        pd.testing.assert_frame_equal(Ma[before], Mb[before])
        assert not Ma[~before].equals(Mb[~before])
        assert Mb.loc[pd.Timestamp(later), "H.NS"] and not Mb.loc[pd.Timestamp(later), "A.NS"]
    finally:
        sa.close()
        sb.close()


def test_dates_before_complete_from_fall_back_to_active():
    from investbot.evaluation.panel import point_in_time_membership

    dates = pd.bdate_range("2019-03-25", periods=6)       # 25..29 Mar, 1 Apr
    M = pd.DataFrame({"A.NS": [False] * 6, "B.NS": [True] * 6, "C.NS": [True] * 6}, index=dates)
    out = point_in_time_membership(M, {"A.NS", "B.NS"}, complete_from=date(2019, 3, 29))
    early = dates < pd.Timestamp("2019-03-29")
    assert out.loc[early, "A.NS"].all() and out.loc[early, "B.NS"].all()
    assert not out.loc[early, "C.NS"].any()
    pd.testing.assert_frame_equal(out[~early], M[~early])


# ──────────────────────────────────────────────────────────
# Coverage
# ──────────────────────────────────────────────────────────

def test_coverage_hand_example():
    from investbot.evaluation.panel import universe_coverage

    dates = pd.bdate_range("2024-01-01", periods=4)
    # members: A always, B always, C always (no prices), D from day 3
    M = pd.DataFrame({"A": [True] * 4, "B": [True] * 4, "C": [True] * 4,
                      "D": [False, False, True, True]}, index=dates)
    # values: C missing entirely (no price rows), B in warmup on day 1, D has a value before joining
    V = pd.DataFrame({"A": [0.1, 0.2, 0.3, 0.4], "B": [np.nan, 0.1, 0.1, 0.1],
                      "D": [0.5, 0.5, np.nan, 0.2]}, index=dates)
    cov = universe_coverage(M, V)
    assert cov["members"].tolist() == [3, 3, 4, 4]
    assert cov["covered"].tolist() == [1, 2, 2, 3]
    assert cov["coverage"].tolist() == pytest.approx([1 / 3, 2 / 3, 2 / 4, 3 / 4])
    assert np.isnan(universe_coverage(M.iloc[:, :0], V)["coverage"]).all()


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    """12 priced names (one joins, one leaves) + one member with no prices; real run."""
    pytest.importorskip("talib")
    from investbot.evaluation.report import run_evaluation

    root = tmp_path_factory.mktemp("e2e")
    n = 620
    assets = {f"S{i:02d}.NS": (n, True) for i in range(10)}
    assets.update({"JOIN.NS": (n, True), "LEFT.NS": (n, False), "NOPX.NS": (0, True)})
    stints = [(t, date(2019, 1, 1), None) for t in assets if t not in ("JOIN.NS", "LEFT.NS")]
    stints += [("JOIN.NS", date(2025, 3, 3), None),
               ("LEFT.NS", date(2019, 1, 1), date(2025, 2, 2))]
    s = _make_db(root / "e.db", assets, stints)
    try:
        report, csv, table, info = run_evaluation(
            (1, 5), session=s, universe="index", report_dir=root, csv_dir=root,
            report_date=date(2026, 10, 18))
    finally:
        s.close()
    return report, csv, table, info


def test_run_evaluation_index_universe_report(e2e):
    report, csv, table, info = e2e
    assert info.universe == "index" and info.n_members == 13 and info.n_assets == 12
    assert info.missing_prices and set(info.missing_prices) == {"NOPX.NS"}
    assert info.short_history == {}
    md = report.read_text()
    assert "## Universe effect" in md
    assert "NOPX.NS" in md and "Coverage is below 100%" in md
    assert "peak RSS" in md
    out = pd.read_csv(csv)
    assert set(out["universe"]) == {"index", "active"}
    # both universes on the same OOS blocks
    blocks = out[out["subset"].str.startswith("block:")]
    assert set(blocks.loc[blocks["universe"] == "index", "subset"]) == \
        set(blocks.loc[blocks["universe"] == "active", "subset"])


def test_run_evaluation_coverage_numbers(e2e):
    """Recompute coverage by hand: members per OOS date (NOPX always, JOIN from its date,
    LEFT until its exit) vs names with a TECH_SCORE."""
    _, _, _, info = e2e
    # every priced name has TECH_SCORE throughout the OOS window (warmup ends ~bar 200),
    # so coverage = priced members / members: NOPX is the only gap.
    # members: 10 S + NOPX always, LEFT until 2025-02-02, JOIN from 2025-03-03
    # -> 12 members before 2025-02-03 (11 covered), 11 between (10), 12 from 2025-03-03 (11)
    assert info.coverage_skipped_dates == 0
    assert info.coverage_min == pytest.approx(10 / 11)
    assert info.coverage_min_date >= pd.Timestamp("2025-02-03")
    assert info.coverage_min_date < pd.Timestamp("2025-03-03")
    assert 10 / 11 < info.coverage_mean < 11 / 12 + 1e-12
    assert info.fallback_oos_dates == 0
