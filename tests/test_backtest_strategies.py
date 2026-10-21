"""
Strategy-to-weights adapter (investbot/backtest/strategies.py). Synthetic fixtures only.
"""

import inspect
import logging

import numpy as np
import pandas as pd
import pytest

from investbot.backtest import strategies
from investbot.backtest.simulator import simulate
from investbot.backtest.strategies import (
    DEFAULT_QUANTILE,
    equal_weight_universe,
    rebalance_sessions,
    signal_panel,
    top_quantile_weights,
)
from investbot.evaluation import metrics
from investbot.evaluation.panel import _stack

FREQS = ["D", "W", "M", "5D"]


def ts(*days: str) -> list[pd.Timestamp]:
    return [pd.Timestamp(d) for d in days]


def one_row(values, members=None, date="2025-03-03") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single-session signal and membership frames over assets 1..n."""
    cols = pd.Index(range(1, len(values) + 1), name="asset_id")
    idx = pd.DatetimeIndex([date], name="date")
    signal = pd.DataFrame([values], index=idx, columns=cols, dtype=float)
    m = [True] * len(values) if members is None else members
    return signal, pd.DataFrame([m], index=idx, columns=cols)


def selected(w: pd.DataFrame) -> list[int]:
    row = w.iloc[0]
    return sorted(int(a) for a in row.index[row > 0])


def random_panel(seed: int, n_assets: int = 40, start: str = "2023-11-01", periods: int = 320):
    """Random signal (integer-valued, so ties occur), membership and closes with holidays."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=periods)
    days = days[rng.random(len(days)) > 0.05]  # random holidays, month/week ends included
    cols = pd.Index(range(1, n_assets + 1), name="asset_id")
    signal = pd.DataFrame(rng.integers(0, 20, (len(days), n_assets)).astype(float),
                          index=days, columns=cols)
    signal = signal.mask(rng.random(signal.shape) < 0.1)
    membership = pd.DataFrame(rng.random(signal.shape) < 0.8, index=days, columns=cols)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, signal.shape), axis=0)),
                         index=days, columns=cols)
    close = close.mask(rng.random(close.shape) < 0.05)
    return signal, membership, close


# --- rebalance_sessions -----------------------------------------------------------------


def test_month_end_holiday_uses_last_present_session():
    # 2025-03-31 (Monday) was an NSE holiday: March's last session is Friday 2025-03-28.
    days = pd.bdate_range("2025-02-20", "2025-04-03").drop(pd.Timestamp("2025-03-31"))
    assert list(rebalance_sessions(days, "M")) == ts("2025-02-28", "2025-03-28")


def test_iso_week_across_year_boundary():
    # 2024-12-30 .. 2025-01-03 is ISO week 2025-W01; December's month end is still 12-31.
    days = pd.bdate_range("2024-12-23", "2025-01-10")
    assert list(rebalance_sessions(days, "W")) == ts("2024-12-27", "2025-01-03")
    days = pd.bdate_range("2024-12-02", "2025-01-10")
    assert list(rebalance_sessions(days, "M")) == ts("2024-12-31")


def test_every_n_sessions():
    days = pd.bdate_range("2025-01-01", periods=23)
    out = rebalance_sessions(days, "5D")
    assert list(out) == [days[i] for i in (0, 5, 10, 15, 20)]
    assert list(rebalance_sessions(days, "1D")) == list(days)
    assert list(rebalance_sessions(days, "D")) == list(days)


def test_trailing_period_excluded():
    # The index stops on Wednesday 2025-01-15, mid-week and mid-month.
    days = pd.bdate_range("2024-12-16", "2025-01-15")
    assert list(rebalance_sessions(days, "W")) == ts("2024-12-20", "2024-12-27", "2025-01-03", "2025-01-10")
    assert list(rebalance_sessions(days, "M")) == ts("2024-12-31")
    # Even when the index ends on a true week / month end, that period is not closed yet.
    days = pd.bdate_range("2025-01-01", "2025-01-31")  # ends on Friday 31 January
    assert pd.Timestamp("2025-01-31") not in rebalance_sessions(days, "W")
    assert len(rebalance_sessions(days, "M")) == 0
    # ... until the next session arrives.
    days = pd.bdate_range("2025-01-01", "2025-02-03")
    assert pd.Timestamp("2025-01-31") in rebalance_sessions(days, "W")
    assert list(rebalance_sessions(days, "M")) == ts("2025-01-31")


@pytest.mark.parametrize("freq", ["Q", "w", "0D", "2W", "D5", "", " M", None, 5])
def test_unknown_freq_raises(freq):
    with pytest.raises(ValueError):
        rebalance_sessions(pd.bdate_range("2025-01-01", periods=10), freq)


def test_unsorted_sessions_raise():
    days = pd.bdate_range("2025-01-01", periods=5)
    with pytest.raises(ValueError):
        rebalance_sessions(days[::-1], "D")


def test_freq_has_no_default():
    for fn in (rebalance_sessions, top_quantile_weights, equal_weight_universe):
        p = inspect.signature(fn).parameters["freq"]
        assert p.default is inspect.Parameter.empty


# --- top_quantile_weights ---------------------------------------------------------------


def test_hand_checked_ten_names():
    signal, members = one_row([5, 3, 9, 1, 7, 2, 8, 4, 6, 0])
    w = top_quantile_weights(signal, members, freq="D")
    assert selected(w) == [3, 7]  # values 9 and 8: ceil(0.2 x 10) = 2
    assert w.loc[:, [3, 7]].to_numpy().tolist() == [[0.5, 0.5]]
    assert w.iloc[0].sum() == pytest.approx(1.0, abs=1e-12)
    assert list(w.columns) == list(signal.columns) and list(w.index) == list(signal.index)


def test_bottom_quantile():
    signal, members = one_row([5, 3, 9, 1, 7, 2, 8, 4, 6, 0])
    w = top_quantile_weights(signal, members, freq="D", higher_is_better=False)
    assert selected(w) == [4, 10]  # values 1 and 0
    assert w.iloc[0][[4, 10]].tolist() == [0.5, 0.5]


def test_ties_at_cutoff_all_included():
    signal, members = one_row([5, 3, 9, 1, 7, 2, 8, 4, 6, 8])
    w = top_quantile_weights(signal, members, freq="D")
    assert selected(w) == [3, 7, 10]  # 9, then 8 tied at the cutoff
    assert w.iloc[0][[3, 7, 10]].tolist() == pytest.approx([1 / 3] * 3, abs=1e-15)
    signal, members = one_row([5, 1, 9, 1, 7, 2, 8, 4, 6, 0])
    w = top_quantile_weights(signal, members, freq="D", higher_is_better=False)
    assert selected(w) == [2, 4, 10]  # 0, then 1 tied at the cutoff


def test_fewer_than_min_names_is_cash(caplog):
    values = [5, 3, 9, 1, 7, 2, 8, 4, 6, 0]
    signal, members = one_row(values, [True] * 9 + [False])
    with caplog.at_level(logging.INFO, logger="investbot.backtest.strategies"):
        w = top_quantile_weights(signal, members, freq="D")
    assert (w.to_numpy() == 0).all() and len(w) == 1
    assert "1 of 1 rebalance rows" in caplog.text
    signal, members = one_row(values[:9] + [np.nan])
    assert (top_quantile_weights(signal, members, freq="D").to_numpy() == 0).all()
    w = top_quantile_weights(signal, members, freq="D", min_names=9)
    assert selected(w) == [3, 7]  # ceil(0.2 x 9) = 2


def test_non_member_with_best_signal_never_selected():
    signal, members = one_row([5, 3, 9, 1, 7, 2, 8, 4, 6, 0, 100, np.inf], [True] * 10 + [False, True])
    w = top_quantile_weights(signal, members, freq="D")
    assert selected(w) == [3, 7]  # 11 is not a member, 12 has no finite signal
    # a membership frame that lacks an asset or a session counts as not a member
    w = top_quantile_weights(signal, members.drop(columns=[11]), freq="D")
    assert selected(w) == [3, 7]
    w = top_quantile_weights(signal, members.iloc[:0], freq="D")
    assert (w.to_numpy() == 0).all()


def test_float_quantile_rounding():
    signal, members = one_row(list(range(30)))
    w = top_quantile_weights(signal, members, freq="D", quantile=0.1)
    assert selected(w) == [28, 29, 30]  # 0.1 x 30 = 3.0000000000000004 -> 3, not 4


def test_bad_quantile_or_min_names_raise():
    signal, members = one_row(list(range(10)))
    for q in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            top_quantile_weights(signal, members, freq="D", quantile=q)
    with pytest.raises(ValueError):
        top_quantile_weights(signal, members, freq="D", min_names=0)


def test_row_sums_are_one_or_zero_and_inputs_untouched():
    signal, membership, _ = random_panel(1)
    s0, m0 = signal.copy(), membership.copy()
    for freq in FREQS:
        w = top_quantile_weights(signal, membership, freq=freq, min_names=29)
        sums = w.sum(axis=1).to_numpy()
        assert np.all((np.abs(sums - 1) <= 1e-12) | (sums == 0))
        assert (sums == 0).any() and (sums > 0).any()  # both cases occur in the fixture
        assert (w.to_numpy() >= 0).all() and np.isfinite(w.to_numpy()).all()
        assert list(w.index) == list(rebalance_sessions(signal.index, freq))
        assert list(w.columns) == list(signal.columns)
        # nothing outside the eligible set is ever weighted
        sub_s, sub_m = signal.loc[w.index], membership.loc[w.index]
        assert not ((w > 0) & ~(sub_m & np.isfinite(sub_s))).to_numpy().any()
    pd.testing.assert_frame_equal(signal, s0)
    pd.testing.assert_frame_equal(membership, m0)


def test_name_leaving_index_gets_zero_and_no_forward_fill():
    cols = pd.Index(range(1, 11), name="asset_id")
    days = pd.bdate_range("2025-01-27", "2025-03-04")
    signal = pd.DataFrame(np.tile(np.arange(10, dtype=float), (len(days), 1)), index=days, columns=cols)
    membership = pd.DataFrame(True, index=days, columns=cols)
    membership.loc["2025-02-10":, 10] = False  # the top name leaves mid-February
    w = top_quantile_weights(signal, membership, freq="M", min_names=5)
    assert list(w.index) == ts("2025-01-31", "2025-02-28")
    assert selected(w.iloc[[0]]) == [9, 10]
    assert selected(w.iloc[[1]]) == [8, 9] and w.loc["2025-02-28", 10] == 0.0


def test_defaults():
    assert DEFAULT_QUANTILE == 0.2
    assert strategies.MIN_NAMES is metrics.MIN_NAMES
    p = inspect.signature(top_quantile_weights).parameters
    assert p["quantile"].default == DEFAULT_QUANTILE and p["min_names"].default == metrics.MIN_NAMES


def test_no_dropna_in_module():
    assert "dropna(" not in inspect.getsource(strategies)


# --- truncation invariance (no lookahead) ------------------------------------------------


def _check_truncation(fn, frames_full, s, freq, w_full):
    trunc = fn(*(f.loc[:s] for f in frames_full), freq=freq)
    # Rows before s are all present; the row dated s only for D / nD (a week or month
    # ending on s is not known to have ended until the next session exists).
    expect = w_full.index[(w_full.index < s) | ((w_full.index == s) & freq.endswith("D"))]
    assert list(trunc.index) == list(expect)
    pd.testing.assert_frame_equal(trunc, w_full.loc[trunc.index])


@pytest.mark.parametrize("seed", [2, 3])
@pytest.mark.parametrize("freq", FREQS)
def test_truncation_invariance(seed, freq):
    signal, membership, close = random_panel(seed)
    rng = np.random.default_rng(100 + seed)
    tq = lambda sig, mem, freq: top_quantile_weights(sig, mem, freq=freq, min_names=29)  # noqa: E731
    w_full = tq(signal, membership, freq)
    month_ends = rebalance_sessions(signal.index, "M")
    cuts = list(rng.choice(signal.index[30:], 8, replace=False)) + list(month_ends[:3]) + [signal.index[-1]]
    for s in cuts:
        s = pd.Timestamp(s)
        _check_truncation(tq, (signal, membership), s, freq, w_full)

        # changing any signal / membership value after s leaves rows <= s unchanged
        later = signal.index > s
        sig2, mem2 = signal.copy(), membership.copy()
        sig2.loc[later] = rng.normal(0, 10, (int(later.sum()), signal.shape[1]))
        mem2.loc[later] = rng.random((int(later.sum()), signal.shape[1])) < 0.5
        w2 = tq(sig2, mem2, freq)
        pd.testing.assert_frame_equal(w2.loc[:s], w_full.loc[:s])

    ew = lambda mem, cl, freq: equal_weight_universe(mem, cl, freq=freq, min_names=25)  # noqa: E731
    ew_full = ew(membership, close, freq)
    for s in cuts:
        _check_truncation(ew, (membership, close), pd.Timestamp(s), freq, ew_full)


# --- equal_weight_universe ---------------------------------------------------------------


def test_equal_weight_universe_hand_checked():
    days = pd.bdate_range("2025-01-27", "2025-02-04")
    cols = pd.Index([1, 2, 3, 4], name="asset_id")
    close = pd.DataFrame(100.0, index=days, columns=cols)
    close.loc["2025-01-31", 2] = np.nan  # no bar on the month end
    membership = pd.DataFrame(True, index=days, columns=cols)
    membership[4] = False
    w = equal_weight_universe(membership, close, freq="M", min_names=2)
    assert list(w.index) == ts("2025-01-31")
    assert w.iloc[0].tolist() == pytest.approx([0.5, 0.0, 0.5, 0.0], abs=1e-15)
    w = equal_weight_universe(membership, close, freq="D", min_names=3)
    assert (w.loc["2025-01-31"] == 0).all()  # only 2 eligible -> cash
    assert w.loc["2025-02-03"].tolist() == pytest.approx([1 / 3] * 3 + [0.0], abs=1e-15)


# --- signal_panel --------------------------------------------------------------------------


def stacked_features() -> pd.DataFrame:
    """A (date, ticker) frame laid out as evaluation.panel builds it."""
    days = pd.bdate_range("2025-01-01", periods=4)
    frames = {
        "BBB.NS": pd.DataFrame({"TECH_SCORE": [0.1, 0.2, np.nan, 0.4], "BASE_X": 1.0}, index=days),
        "AAA.NS": pd.DataFrame({"TECH_SCORE": [0.5, 0.6, 0.7], "BASE_X": 2.0}, index=days[1:]),
    }
    return _stack(frames)


def test_signal_panel_pivot():
    feats = stacked_features()
    f0 = feats.copy()
    wide = signal_panel(feats, "TECH_SCORE")
    assert list(wide.columns) == ["AAA.NS", "BBB.NS"]
    assert list(wide.index) == list(pd.bdate_range("2025-01-01", periods=4))
    assert np.isnan(wide.iloc[0]["AAA.NS"]) and np.isnan(wide.iloc[2]["BBB.NS"])  # kept, not dropped
    assert wide.iloc[3].tolist() == [0.7, 0.4]
    by_id = signal_panel(feats, "TECH_SCORE", ids={"AAA.NS": 7, "BBB.NS": 3, "CCC.NS": 9})
    assert list(by_id.columns) == [3, 7] and by_id.columns.name == "asset_id"
    assert by_id[7].equals(wide["AAA.NS"].rename(7))
    pd.testing.assert_frame_equal(feats, f0)


def test_signal_panel_errors():
    feats = stacked_features()
    with pytest.raises(KeyError):
        signal_panel(feats, "NOPE")
    with pytest.raises(ValueError):
        signal_panel(feats, "TECH_SCORE", ids={"AAA.NS": 1})
    with pytest.raises(ValueError):
        signal_panel(feats.reset_index().set_index(["ticker", "date"]), "TECH_SCORE")


# --- end to end ------------------------------------------------------------------------------


def test_end_to_end_simulate_daily_trades_more_than_monthly():
    rng = np.random.default_rng(7)
    days = pd.bdate_range("2021-01-01", periods=200)
    cols = pd.Index(range(1, 16), name="asset_id")
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, (len(days), 15)), axis=0)),
                         index=days, columns=cols)
    open_ = close.shift(1).fillna(close) * np.exp(rng.normal(0, 0.003, close.shape))
    momentum = close / close.shift(20) - 1  # causal: uses closes up to t
    membership = pd.DataFrame(True, index=days, columns=cols)
    membership.iloc[100:, 14] = False

    monthly = top_quantile_weights(momentum, membership, freq="M")
    daily = top_quantile_weights(momentum, membership, freq="D")
    bench = equal_weight_universe(membership, close, freq="M")
    res_m = simulate(monthly, open_, close)
    res_d = simulate(daily, open_, close)
    res_b = simulate(bench, open_, close)
    for res in (res_m, res_d, res_b):
        assert np.isfinite(res.daily["nav"]).all() and len(res.trades) > 0
    assert len(res_d.trades) > len(res_m.trades)
