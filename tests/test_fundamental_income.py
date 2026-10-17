"""Tests for investbot.scoring.fundamental.income_statement (synthetic frames only)."""
import numpy as np
import pandas as pd
import pytest

from investbot.scoring.fundamental import income_statement as m
from investbot.scoring.fundamental.income_statement import compute_income_statement_scores

P = m.PREFIX


def _index(n: int, start: int = 2010) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(f"{start + i}-03-31") for i in range(n)])


def _frame(n: int = 12, **overrides) -> pd.DataFrame:
    """A healthy, steadily growing company. Overrides replace whole columns."""
    years = np.arange(n, dtype=float)
    revenue = 1000.0 * 1.1 ** years
    data = {
        "revenue": revenue,
        "cogs": revenue * 0.5,
        "gross_profit": revenue * 0.5,
        "sga_expense": revenue * 0.1,
        "rnd_expense": revenue * 0.02,
        "depreciation": revenue * 0.01,
        "amortization": revenue * 0.005,
        "interest_expense": revenue * 0.02,
        "ebit": revenue * 0.3,
        "pretax_income": revenue * 0.28,
        "tax_paid": revenue * 0.28 * 0.25,
        "net_income": revenue * 0.21,
        "eps": 10.0 * 1.1 ** years,
    }
    data.update({k: np.asarray(v, dtype=float) for k, v in overrides.items()})
    return pd.DataFrame(data, index=_index(n))


def _ratio_frame(**ratios) -> pd.DataFrame:
    """Single-row frame with revenue=100, gross_profit=100 and ebit/pretax=100."""
    base = {"revenue": [100.0], "gross_profit": [100.0], "ebit": [100.0], "pretax_income": [100.0]}
    base.update({k: [float(v)] for k, v in ratios.items()})
    return pd.DataFrame(base, index=_index(1))


# ── Output shape and contract ──────────────────────────────────────────────

def test_output_columns_and_index():
    fin = _frame()
    out = compute_income_statement_scores(fin)
    assert out.index.equals(fin.index)
    expected = []
    for name in m.METRICS:
        expected.append(f"{P}{name}")
        if name in m.SCORED_METRICS:
            expected.append(f"{P}{name}_SCORE")
    expected += [m.TAX_FLAG_COL, m.SCORE_COL]
    assert list(out.columns) == expected
    assert f"{P}TAX_RATE_SCORE" not in out.columns
    assert out[m.TAX_FLAG_COL].dtype == bool


def test_healthy_company_values():
    out = compute_income_statement_scores(_frame())
    last = out.iloc[-1]
    assert last[f"{P}GROSS_MARGIN"] == pytest.approx(0.5)
    assert last[f"{P}GROSS_MARGIN_STD"] == pytest.approx(0.0, abs=1e-12)
    assert last[f"{P}SGA_RATIO"] == pytest.approx(0.2)
    assert last[f"{P}RND_BURDEN"] == pytest.approx(0.04)
    assert last[f"{P}DEPRECIATION_BURDEN"] == pytest.approx(0.01 / 0.3)
    assert last[f"{P}INTEREST_BURDEN"] == pytest.approx(0.02 / 0.3)
    assert last[f"{P}NET_MARGIN"] == pytest.approx(0.21)
    assert last[f"{P}NET_MARGIN_SLOPE"] == pytest.approx(0.0, abs=1e-9)
    assert last[f"{P}EPS_CONSISTENCY"] == pytest.approx(1.0)
    assert last[f"{P}TAX_RATE"] == pytest.approx(0.25)
    assert not last[m.TAX_FLAG_COL]
    # every sub-score is 100 except the flat net-margin slope (60): (9*100 + 60) / 10
    assert last[m.SCORE_COL] == pytest.approx(96.0)


# ── Hand-computed ratios ───────────────────────────────────────────────────

def test_window_mean_and_std_hand_computed():
    revenue = [100.0, 100.0, 100.0]
    gp = [30.0, 40.0, 50.0]
    sga = [6.0, 12.0, 25.0]           # ratios 0.2, 0.3, 0.5
    out = compute_income_statement_scores(_frame(3, revenue=revenue, gross_profit=gp,
                                                 sga_expense=sga))
    assert out[f"{P}GROSS_MARGIN"].tolist() == pytest.approx([0.30, 0.35, 0.40])
    assert np.isnan(out[f"{P}GROSS_MARGIN_STD"].iloc[1])          # needs 3 years
    assert out[f"{P}GROSS_MARGIN_STD"].iloc[2] == pytest.approx(0.1)
    assert out[f"{P}SGA_RATIO"].iloc[2] == pytest.approx((0.2 + 0.3 + 0.5) / 3)
    assert out[f"{P}SGA_RATIO_STD"].iloc[2] == pytest.approx(np.std([0.2, 0.3, 0.5], ddof=1))
    assert out[f"{P}GROSS_MARGIN_STD_SCORE"].iloc[2] == m.SCORE_INCONSISTENT
    assert out[f"{P}SGA_RATIO_STD_SCORE"].iloc[2] == m.SCORE_INCONSISTENT


def test_gross_profit_falls_back_to_revenue_minus_cogs():
    fin = _frame(3, gross_profit=[np.nan, np.nan, np.nan], cogs=[70.0, 70.0, 70.0],
                 revenue=[100.0, 100.0, 100.0], sga_expense=[15.0, 15.0, 15.0])
    out = compute_income_statement_scores(fin)
    assert out[f"{P}GROSS_MARGIN"].iloc[-1] == pytest.approx(0.3)
    assert out[f"{P}SGA_RATIO"].iloc[-1] == pytest.approx(0.5)
    fin = fin.drop(columns="gross_profit")
    assert compute_income_statement_scores(fin)[f"{P}GROSS_MARGIN"].iloc[-1] == pytest.approx(0.3)


def test_net_margin_slope_in_pp_per_year():
    revenue = [100.0] * 4
    ni = [10.0, 12.0, 14.0, 16.0]      # +2 pp per year
    out = compute_income_statement_scores(_frame(4, revenue=revenue, net_income=ni))
    slope = out[f"{P}NET_MARGIN_SLOPE"]
    assert np.isnan(slope.iloc[1])                                  # needs 3 years
    # period ends are real dates, so a leap year makes a year 366/365.25 long
    assert slope.iloc[3] == pytest.approx(2.0, rel=1e-3)
    assert out[f"{P}NET_MARGIN_SLOPE_SCORE"].iloc[3] == m.SCORE_TOP

    out = compute_income_statement_scores(_frame(4, revenue=revenue, net_income=ni[::-1]))
    assert out[f"{P}NET_MARGIN_SLOPE"].iloc[3] == pytest.approx(-2.0, rel=1e-3)
    assert out[f"{P}NET_MARGIN_SLOPE_SCORE"].iloc[3] == m.SCORE_LOW


def test_net_margin_slope_uses_dates_not_row_positions():
    fin = _frame(3, revenue=[100.0] * 3, net_income=[10.0, 12.0, 14.0])
    fin.index = pd.DatetimeIndex(["2010-03-31", "2012-03-31", "2014-03-31"])
    out = compute_income_statement_scores(fin)
    assert out[f"{P}NET_MARGIN_SLOPE"].iloc[-1] == pytest.approx(1.0, rel=1e-3)


def test_eps_consistency_hand_computed():
    eps = [1.0, 2.0, 1.5, 3.0, 4.0, 5.0]   # changes: up, down, up, up, up
    out = compute_income_statement_scores(_frame(6, eps=eps))
    c = out[f"{P}EPS_CONSISTENCY"]
    assert np.isnan(c.iloc[0]) and np.isnan(c.iloc[1])             # needs 3 years
    assert c.tolist()[2:] == pytest.approx([0.5, 2 / 3, 0.75, 0.8])
    assert out[f"{P}EPS_CONSISTENCY_SCORE"].tolist()[2:] == [60.0, 60.0, 60.0, 100.0]


def test_eps_consistency_skips_nan_pairs():
    eps = [1.0, 2.0, np.nan, 3.0, 2.0]     # valid pairs: (1,2) up, (3,2) down
    out = compute_income_statement_scores(_frame(5, eps=eps))
    assert np.isnan(out[f"{P}EPS_CONSISTENCY"].iloc[3])            # only 1 valid pair
    assert out[f"{P}EPS_CONSISTENCY"].iloc[4] == pytest.approx(0.5)


def test_tax_rate_and_flag():
    fin = _frame(3, pretax_income=[100.0, 100.0, 100.0], tax_paid=[25.17, 10.0, 40.0])
    out = compute_income_statement_scores(fin)
    assert out[f"{P}TAX_RATE"].tolist() == pytest.approx([0.2517, 0.10, 0.40])
    assert out[m.TAX_FLAG_COL].tolist() == [False, True, True]


# ── Score band edges ───────────────────────────────────────────────────────

@pytest.mark.parametrize("gp, expected", [
    (40.01, 100.0), (40.0, 60.0), (20.0, 60.0), (19.99, 20.0),
])
def test_gross_margin_bands(gp, expected):
    out = compute_income_statement_scores(_ratio_frame(gross_profit=gp))
    assert out[f"{P}GROSS_MARGIN_SCORE"].iloc[0] == expected


@pytest.mark.parametrize("col, metric, values", [
    ("sga_expense", "SGA_RATIO", [(29.99, 100.0), (30.0, 60.0), (50.0, 60.0), (50.01, 20.0)]),
    ("rnd_expense", "RND_BURDEN", [(9.99, 100.0), (10.0, 60.0), (30.0, 60.0), (30.01, 20.0)]),
    ("depreciation", "DEPRECIATION_BURDEN",
     [(9.99, 100.0), (10.0, 70.0), (15.0, 70.0), (15.01, 40.0)]),
    ("interest_expense", "INTEREST_BURDEN",
     [(14.99, 100.0), (15.0, 60.0), (25.0, 60.0), (25.01, 20.0)]),
    ("net_income", "NET_MARGIN", [(20.01, 100.0), (20.0, 60.0), (10.0, 60.0), (9.99, 20.0)]),
])
def test_ratio_bands(col, metric, values):
    for value, expected in values:
        out = compute_income_statement_scores(_ratio_frame(**{col: value}))
        assert out[f"{P}{metric}"].iloc[0] == pytest.approx(value / 100.0)
        assert out[f"{P}{metric}_SCORE"].iloc[0] == expected, (metric, value)


@pytest.mark.parametrize("spread, expected", [(0.0299, 100.0), (0.0301, 40.0)])
def test_gross_margin_std_band(spread, expected):
    # margins 0.5 - d, 0.5, 0.5 + d have sample std exactly d
    gp = [100.0 * (0.5 - spread), 50.0, 100.0 * (0.5 + spread)]
    out = compute_income_statement_scores(_frame(3, revenue=[100.0] * 3, gross_profit=gp))
    assert out[f"{P}GROSS_MARGIN_STD"].iloc[-1] == pytest.approx(spread)
    assert out[f"{P}GROSS_MARGIN_STD_SCORE"].iloc[-1] == expected


@pytest.mark.parametrize("step, expected", [(0.51, 100.0), (0.49, 60.0), (-0.49, 60.0),
                                            (-0.51, 20.0)])
def test_net_margin_slope_band(step, expected):
    ni = [10.0, 10.0 + step, 10.0 + 2 * step]
    fin = _frame(3, revenue=[100.0] * 3, net_income=ni)
    fin.index = pd.Index([2020, 2021, 2022])   # numeric index → fiscal years
    out = compute_income_statement_scores(fin)
    assert out[f"{P}NET_MARGIN_SLOPE"].iloc[-1] == pytest.approx(step)
    assert out[f"{P}NET_MARGIN_SLOPE_SCORE"].iloc[-1] == expected


def test_composite_needs_min_subscores():
    # One row with only revenue, gross_profit and net_income: GROSS_MARGIN, SGA (NaN),
    # RND (0 → 100), NET_MARGIN → 3 sub-scores, below MIN_SUBSCORES.
    fin = pd.DataFrame({"revenue": [100.0], "gross_profit": [50.0], "net_income": [25.0]},
                       index=_index(1))
    out = compute_income_statement_scores(fin)
    assert np.isnan(out[m.SCORE_COL].iloc[0])
    fin["ebit"] = [40.0]
    fin["interest_expense"] = [2.0]
    out = compute_income_statement_scores(fin)
    assert out[m.SCORE_COL].iloc[0] == pytest.approx(100.0)


# ── Missing data and bad denominators ──────────────────────────────────────

def test_rnd_missing_counts_as_zero():
    fin = _frame(3).drop(columns="rnd_expense")
    out = compute_income_statement_scores(fin)
    assert out[f"{P}RND_BURDEN"].tolist() == [0.0, 0.0, 0.0]
    assert out[f"{P}RND_BURDEN_SCORE"].tolist() == [100.0, 100.0, 100.0]
    fin = _frame(3, rnd_expense=[np.nan, 5.0, np.nan])
    out = compute_income_statement_scores(fin)
    assert out[f"{P}RND_BURDEN"].iloc[0] == 0.0
    assert out[f"{P}RND_BURDEN"].iloc[1] > 0.0


def test_empty_frame_gives_all_nan_without_errors():
    fin = pd.DataFrame(index=_index(4))
    out = compute_income_statement_scores(fin)
    assert len(out) == 4
    numeric = out.drop(columns=m.TAX_FLAG_COL)
    assert numeric.isna().all().all()
    assert not out[m.TAX_FLAG_COL].any()


def test_zero_rows():
    out = compute_income_statement_scores(pd.DataFrame(index=pd.DatetimeIndex([])))
    assert len(out) == 0
    assert m.SCORE_COL in out.columns


def test_nan_rows_give_nan_not_exceptions():
    fin = _frame(5)
    fin.iloc[2] = np.nan
    out = compute_income_statement_scores(fin)
    for name in ("DEPRECIATION_BURDEN", "INTEREST_BURDEN", "NET_MARGIN", "TAX_RATE"):
        assert np.isnan(out[f"{P}{name}"].iloc[2]), name
    # window metrics skip the NaN year
    assert out[f"{P}GROSS_MARGIN"].iloc[2] == pytest.approx(0.5)
    assert out[f"{P}GROSS_MARGIN"].iloc[4] == pytest.approx(0.5)


def test_negative_ebit_gives_nan_burdens():
    fin = _frame(3, ebit=[-50.0, 0.0, 300.0])
    out = compute_income_statement_scores(fin)
    for name in ("DEPRECIATION_BURDEN", "INTEREST_BURDEN"):
        assert np.isnan(out[f"{P}{name}"].iloc[0])
        assert np.isnan(out[f"{P}{name}_SCORE"].iloc[0])
        assert np.isnan(out[f"{P}{name}"].iloc[1])
        assert not np.isnan(out[f"{P}{name}"].iloc[2])


def test_negative_pretax_gives_nan_tax_rate_and_no_flag():
    fin = _frame(2, pretax_income=[-100.0, 0.0], tax_paid=[10.0, 5.0])
    out = compute_income_statement_scores(fin)
    assert out[f"{P}TAX_RATE"].isna().all()
    assert not out[m.TAX_FLAG_COL].any()


def test_negative_gross_profit_gives_nan_sga_and_rnd():
    fin = _frame(1, gross_profit=[-10.0])
    out = compute_income_statement_scores(fin)
    for name in ("SGA_RATIO", "RND_BURDEN"):
        assert np.isnan(out[f"{P}{name}"].iloc[0])
        assert np.isnan(out[f"{P}{name}_SCORE"].iloc[0])
    # a negative gross margin itself is real and scores in the bottom band
    assert out[f"{P}GROSS_MARGIN_SCORE"].iloc[0] == m.SCORE_LOW


def test_negative_expense_sign_cannot_flip_into_good_ratio():
    fin = _ratio_frame(interest_expense=-40.0)
    out = compute_income_statement_scores(fin)
    assert out[f"{P}INTEREST_BURDEN"].iloc[0] == pytest.approx(0.4)
    assert out[f"{P}INTEREST_BURDEN_SCORE"].iloc[0] == m.SCORE_LOW


def test_loss_year_net_margin_is_negative_and_scores_low():
    out = compute_income_statement_scores(_frame(1, net_income=[-50.0]))
    assert out[f"{P}NET_MARGIN"].iloc[0] < 0
    assert out[f"{P}NET_MARGIN_SCORE"].iloc[0] == m.SCORE_LOW


def test_unsorted_index_raises():
    fin = _frame(3).iloc[::-1]
    with pytest.raises(ValueError):
        compute_income_statement_scores(fin)


# ── No lookahead ───────────────────────────────────────────────────────────

def _noisy_frame(n: int = 15, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    revenue = 1000.0 * np.cumprod(1 + rng.normal(0.08, 0.1, n))
    fin = pd.DataFrame({
        "revenue": revenue,
        "cogs": revenue * rng.uniform(0.4, 0.8, n),
        "gross_profit": np.nan,
        "sga_expense": revenue * rng.uniform(0.05, 0.2, n),
        "rnd_expense": revenue * rng.uniform(0.0, 0.05, n),
        "depreciation": revenue * rng.uniform(0.01, 0.05, n),
        "interest_expense": revenue * rng.uniform(0.0, 0.05, n),
        "ebit": revenue * rng.uniform(-0.05, 0.3, n),
        "pretax_income": revenue * rng.uniform(-0.05, 0.28, n),
        "tax_paid": revenue * rng.uniform(0.0, 0.08, n),
        "net_income": revenue * rng.uniform(-0.05, 0.25, n),
        "eps": 10.0 * np.cumprod(1 + rng.normal(0.05, 0.2, n)),
    }, index=_index(n))
    fin.iloc[4, 0] = np.nan
    fin.iloc[9, -1] = np.nan
    return fin


@pytest.mark.parametrize("lookback", [3, 5, m.LOOKBACK_YEARS])
def test_truncation_invariance(lookback):
    fin = _noisy_frame()
    full = compute_income_statement_scores(fin, lookback_years=lookback)
    for k in range(1, len(fin) + 1):
        part = compute_income_statement_scores(fin.iloc[:k], lookback_years=lookback)
        pd.testing.assert_frame_equal(part, full.iloc[:k])


def test_future_rows_do_not_change_past_outputs():
    fin = _noisy_frame()
    base = compute_income_statement_scores(fin)
    shocked = fin.copy()
    shocked.iloc[10:] = shocked.iloc[10:] * -3.0
    out = compute_income_statement_scores(shocked)
    pd.testing.assert_frame_equal(out.iloc[:10], base.iloc[:10])


def test_lookback_window_cap():
    # 12 years: the first two have a 10% gross margin, the rest 50%.
    revenue = [100.0] * 12
    gp = [10.0, 10.0] + [50.0] * 10
    fin = _frame(12, revenue=revenue, gross_profit=gp)
    out = compute_income_statement_scores(fin)
    gm = out[f"{P}GROSS_MARGIN"]
    assert gm.iloc[9] == pytest.approx((2 * 0.1 + 8 * 0.5) / 10)    # window rows 0..9
    assert gm.iloc[10] == pytest.approx((0.1 + 9 * 0.5) / 10)       # rows 1..10
    assert gm.iloc[11] == pytest.approx(0.5)                        # rows 2..11
    assert out[f"{P}GROSS_MARGIN_STD"].iloc[11] == pytest.approx(0.0, abs=1e-12)

    out3 = compute_income_statement_scores(fin, lookback_years=3)
    assert out3[f"{P}GROSS_MARGIN"].iloc[4] == pytest.approx(0.5)   # rows 2..4
    assert out3[f"{P}GROSS_MARGIN"].iloc[3] == pytest.approx((0.1 + 2 * 0.5) / 3)


def test_lookback_cap_applies_to_eps_and_slope():
    # EPS falls for the first 5 years, then rises: a 3-year window only sees the rise.
    eps = [5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0]
    ni = [20.0, 18.0, 16.0, 14.0, 12.0, 14.0, 16.0, 18.0]
    fin = _frame(8, eps=eps, revenue=[100.0] * 8, net_income=ni)
    out = compute_income_statement_scores(fin, lookback_years=3)
    assert out[f"{P}EPS_CONSISTENCY"].iloc[-1] == pytest.approx(1.0)
    assert out[f"{P}NET_MARGIN_SLOPE"].iloc[-1] == pytest.approx(2.0, rel=1e-3)
    full = compute_income_statement_scores(fin)
    assert full[f"{P}EPS_CONSISTENCY"].iloc[-1] == pytest.approx(3 / 7)


def test_lookback_must_be_positive():
    with pytest.raises(ValueError):
        compute_income_statement_scores(_frame(2), lookback_years=0)


def test_no_dropna_in_module():
    import inspect
    assert "dropna(" not in inspect.getsource(m)
