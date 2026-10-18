import numpy as np
import pandas as pd
import pytest

from investbot.config.settings import MAX_POSITION_WEIGHT, MAX_SECTOR_WEIGHT
from investbot.trading.risk import (
    COV_MIN_OBS,
    COV_WINDOW,
    cap_weights,
    estimate_covariance,
    inverse_volatility_weights,
    log_returns,
    portfolio_volatility,
    risk_contributions,
    risk_parity_weights,
)


def _cov(sigma: np.ndarray) -> pd.DataFrame:
    names = [f"A{i:02d}" for i in range(len(sigma))]
    return pd.DataFrame(sigma, index=names, columns=names)


def _random_spd(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    vols = rng.uniform(0.1, 0.5, size=n)
    corr_raw = a @ a.T + n * 0.1 * np.eye(n)
    d = np.sqrt(np.diag(corr_raw))
    corr = corr_raw / np.outer(d, d)
    return _cov(corr * np.outer(vols, vols))


def _equicorr(vols: np.ndarray, rho: float) -> np.ndarray:
    n = len(vols)
    corr = np.full((n, n), rho)
    np.fill_diagonal(corr, 1.0)
    return corr * np.outer(vols, vols)


def _returns(n_rows: int, sigma: np.ndarray, seed: int = 0, start="2020-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.multivariate_normal(np.zeros(len(sigma)), sigma, size=n_rows)
    idx = pd.bdate_range(start, periods=n_rows)
    return pd.DataFrame(x, index=idx, columns=[f"T{i:02d}" for i in range(len(sigma))])


def _daily_sigma(n: int, seed: int = 1) -> np.ndarray:
    """A daily covariance with heterogeneous correlations (not constant-correlation)."""
    return _random_spd(n, seed).to_numpy() / 252


# ── log_returns ────────────────────────────────────────────


def test_log_returns_shape_values_and_nan_not_filled():
    idx = pd.bdate_range("2024-01-01", periods=5)
    close = pd.DataFrame({"A": [100, 110, np.nan, 121, 133.1], "B": [50, 50, 50, 25, 50]}, index=idx)
    before = close.copy()
    r = log_returns(close)
    assert r.shape == close.shape
    assert list(r.columns) == ["A", "B"] and r.index.equals(idx)
    assert np.isnan(r["A"].iloc[0]) and np.isnan(r["B"].iloc[0])
    assert r["A"].iloc[1] == pytest.approx(np.log(1.1))
    assert np.isnan(r["A"].iloc[2]) and np.isnan(r["A"].iloc[3])  # no forward fill across the gap
    assert r["A"].iloc[4] == pytest.approx(np.log(1.1))
    assert r["B"].iloc[3] == pytest.approx(np.log(0.5))
    pd.testing.assert_frame_equal(close, before)


def test_log_returns_truncation_invariant():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2023-01-02", periods=60)
    close = pd.DataFrame(100 * np.exp(rng.normal(0, 0.01, (60, 3)).cumsum(axis=0)), index=idx, columns=list("XYZ"))
    full = log_returns(close)
    for t in idx[[1, 20, 45]]:
        pd.testing.assert_frame_equal(log_returns(close.loc[:t]), full.loc[:t])


def test_log_returns_rejects_bad_input():
    idx = pd.bdate_range("2024-01-01", periods=3)
    with pytest.raises(ValueError):
        log_returns(pd.DataFrame({"A": [1.0, 0.0, 2.0]}, index=idx))
    with pytest.raises(ValueError):
        log_returns(pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx[::-1]))


# ── estimate_covariance ────────────────────────────────────


def test_covariance_spd_symmetric_annualised_and_intensity_in_range():
    sigma = _daily_sigma(10)
    rets = _returns(400, sigma)
    est = estimate_covariance(rets)
    c = est.cov.to_numpy()
    assert list(est.cov.index) == list(rets.columns)
    assert np.allclose(c, c.T)
    assert np.linalg.eigvalsh(c).min() > 0
    assert 0.0 <= est.shrinkage <= 1.0
    assert est.n_obs == COV_WINDOW
    assert est.excluded == {}
    # Diagonal of the target equals the sample variances, so shrinkage keeps them.
    sample_var = rets.iloc[-COV_WINDOW:].var(ddof=0).to_numpy() * 252
    assert np.allclose(np.diag(c), sample_var)


def test_covariance_spd_when_assets_close_to_observations():
    sigma = _daily_sigma(40, seed=7)
    rets = _returns(45, sigma, seed=2)
    est = estimate_covariance(rets, min_obs=40)
    assert est.cov.shape == (40, 40)
    assert est.n_obs == 45
    assert 0.0 < est.shrinkage <= 1.0
    assert np.linalg.eigvalsh(est.cov.to_numpy()).min() > 0
    # The raw sample matrix of 40 assets from 45 rows is near-singular; shrinkage fixes the conditioning.
    sample = np.cov(rets.to_numpy(), rowvar=False)
    assert np.linalg.cond(est.cov.to_numpy()) < np.linalg.cond(sample)


def test_shrinkage_tends_to_zero_with_many_observations():
    sigma = _daily_sigma(8, seed=11)
    deltas = []
    for n_rows in (100, 1_000, 20_000):
        rets = _returns(n_rows, sigma, seed=5)
        deltas.append(estimate_covariance(rets, window=n_rows, min_obs=50).shrinkage)
    assert deltas[0] > deltas[1] > deltas[2]
    assert deltas[2] < 0.01
    # The large-sample estimate recovers the true matrix.
    rets = _returns(20_000, sigma, seed=5)
    est = estimate_covariance(rets, window=20_000, min_obs=50, periods_per_year=1)
    assert np.allclose(est.cov.to_numpy(), sigma, atol=0.05 * np.abs(sigma).max())


def test_shrinkage_matches_reference_formula():
    """Loop-based implementation of Ledoit-Wolf (2004) constant correlation."""
    sigma = _daily_sigma(4, seed=13)
    rets = _returns(150, sigma, seed=9)
    est = estimate_covariance(rets, window=150, min_obs=100, periods_per_year=1)

    x = rets.to_numpy()
    t, n = x.shape
    y = x - x.mean(axis=0)
    s = y.T @ y / t
    sd = np.sqrt(np.diag(s))
    r_bar = np.mean([s[i, j] / (sd[i] * sd[j]) for i in range(n) for j in range(n) if i != j])
    f = np.array([[s[i, i] if i == j else r_bar * sd[i] * sd[j] for j in range(n)] for i in range(n)])
    pi = sum(np.mean((y[:, i] * y[:, j] - s[i, j]) ** 2) for i in range(n) for j in range(n))

    def theta(k, i, j):  # AsyCov(s_kk, s_ij)
        return np.mean((y[:, k] ** 2 - s[k, k]) * (y[:, i] * y[:, j] - s[i, j]))

    rho = sum(np.mean((y[:, i] ** 2 - s[i, i]) ** 2) for i in range(n))
    rho += sum(
        r_bar / 2 * (sd[j] / sd[i] * theta(i, i, j) + sd[i] / sd[j] * theta(j, i, j))
        for i in range(n)
        for j in range(n)
        if i != j
    )
    gamma = ((f - s) ** 2).sum()
    delta = min(1.0, max(0.0, (pi - rho) / gamma / t))
    assert est.shrinkage == pytest.approx(delta, rel=1e-10)
    assert np.allclose(est.cov.to_numpy(), delta * f + (1 - delta) * s, rtol=1e-12)


def test_covariance_excludes_short_histories_without_dropna():
    sigma = _daily_sigma(4, seed=17)
    rets = _returns(300, sigma, seed=4)
    rets.iloc[:-60, 0] = np.nan  # T00: only 60 returns in the window (newly listed)
    rets.iloc[100:200, 1] = np.nan  # T01: 150 non-NaN in the window, fine on its own
    before = rets.copy()
    est = estimate_covariance(rets)
    assert "T00" in est.excluded and "min_obs" in est.excluded["T00"]
    assert list(est.cov.index) == ["T01", "T02", "T03"]
    # T01 has 150 non-NaN returns in the 250-row window, so it stays and only its complete rows are used.
    window = rets.iloc[-COV_WINDOW:]
    complete = window[["T01", "T02", "T03"]].notna().all(axis=1)
    assert est.n_obs == int(complete.sum())
    pd.testing.assert_frame_equal(rets, before)


def test_covariance_drops_most_nan_asset_when_too_few_complete_rows():
    sigma = _daily_sigma(3, seed=19)
    rets = _returns(250, sigma, seed=6)
    rets.iloc[0:110, 0] = np.nan  # T00: 140 non-NaN
    rets.iloc[120:220, 1] = np.nan  # T01: 150 non-NaN; complete rows with T00 = 40
    est = estimate_covariance(rets)
    assert list(est.excluded) == ["T00"]
    assert "complete rows" in est.excluded["T00"]
    assert list(est.cov.index) == ["T01", "T02"]
    assert est.n_obs == 150


def test_covariance_zero_variance_and_empty_result():
    sigma = _daily_sigma(2, seed=23)
    rets = _returns(200, sigma)
    rets["FLAT"] = 0.0
    est = estimate_covariance(rets)
    assert est.excluded == {"FLAT": "zero variance"}
    assert list(est.cov.index) == ["T00", "T01"]

    empty = estimate_covariance(rets.iloc[:50])
    assert empty.cov.empty and empty.n_obs == 0
    assert set(empty.excluded) == {"T00", "T01", "FLAT"}


def test_covariance_single_asset():
    rets = _returns(200, np.array([[1e-4]]))
    est = estimate_covariance(rets)
    assert est.shrinkage == 0.0
    assert est.cov.iloc[0, 0] == pytest.approx(rets.iloc[-COV_WINDOW:, 0].var(ddof=0) * 252)


def test_covariance_truncation_invariance():
    sigma = _daily_sigma(6, seed=29)
    rets = _returns(600, sigma, seed=8)
    rets.iloc[:420, 5] = np.nan  # an asset that appears late
    rets.iloc[300:310, 2] = np.nan
    before = rets.copy()
    for t in rets.index[[200, 350, 500, 599]]:
        a = estimate_covariance(rets, as_of=t)
        b = estimate_covariance(rets.loc[:t])
        pd.testing.assert_frame_equal(a.cov, b.cov)
        assert (a.shrinkage, a.n_obs, a.excluded) == (b.shrinkage, b.n_obs, b.excluded)

    # Appending future rows (even wild ones, or a new asset's history) changes nothing.
    t = rets.index[450]
    base = estimate_covariance(rets.loc[:t], as_of=t)
    future_idx = pd.bdate_range(rets.index[-1] + pd.offsets.BDay(), periods=100)
    future = pd.DataFrame(np.random.default_rng(1).normal(0, 0.5, (100, 6)), index=future_idx, columns=rets.columns)
    extended = pd.concat([rets, future])
    extended["NEW"] = np.where(extended.index > t, 0.01, np.nan)
    later = estimate_covariance(extended, as_of=t)
    pd.testing.assert_frame_equal(later.cov, base.cov)
    assert later.shrinkage == base.shrinkage and later.n_obs == base.n_obs
    assert later.excluded == {**base.excluded, "NEW": later.excluded["NEW"]}
    pd.testing.assert_frame_equal(rets, before)


def test_covariance_as_of_accepts_string_and_rejects_unsorted():
    rets = _returns(300, _daily_sigma(3))
    t = rets.index[260]
    a = estimate_covariance(rets, as_of=str(t.date()))
    b = estimate_covariance(rets, as_of=t)
    pd.testing.assert_frame_equal(a.cov, b.cov)
    with pytest.raises(ValueError):
        estimate_covariance(rets.iloc[::-1])


# ── risk parity ───────────────────────────────────────────


def test_inverse_volatility_weights():
    vols = np.array([0.1, 0.2, 0.4])
    w = inverse_volatility_weights(_cov(np.diag(vols**2)))
    expected = (1 / vols) / (1 / vols).sum()
    assert np.allclose(w.to_numpy(), expected)


def test_erc_diagonal_is_inverse_vol():
    vols = np.array([0.1, 0.15, 0.3, 0.45])
    cov = _cov(np.diag(vols**2))
    w = risk_parity_weights(cov)
    assert np.allclose(w.to_numpy(), inverse_volatility_weights(cov).to_numpy(), atol=1e-10)


@pytest.mark.parametrize("rho", [-0.15, 0.0, 0.3, 0.9])
def test_erc_equal_correlation_is_inverse_vol(rho):
    vols = np.array([0.12, 0.2, 0.25, 0.33, 0.5])
    cov = _cov(_equicorr(vols, rho))
    w = risk_parity_weights(cov)
    assert np.allclose(w.to_numpy(), inverse_volatility_weights(cov).to_numpy(), atol=1e-10)


@pytest.mark.parametrize("rho", [-0.8, 0.0, 0.6, 0.99])
def test_erc_two_asset_closed_form(rho):
    s1, s2 = 0.18, 0.42
    cov = _cov(np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]]))
    w = risk_parity_weights(cov)
    assert w.iloc[0] == pytest.approx(s2 / (s1 + s2), abs=1e-12)
    assert w.iloc[1] == pytest.approx(s1 / (s1 + s2), abs=1e-12)


@pytest.mark.parametrize("n,seed", [(5, 0), (20, 1), (50, 2), (50, 3)])
def test_erc_random_spd_equal_contributions(n, seed):
    cov = _random_spd(n, seed)
    before = cov.copy()
    w = risk_parity_weights(cov)
    rc = risk_contributions(w, cov)
    assert (w > 0).all()
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert rc.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.abs(rc.to_numpy() - 1.0 / n).max() < 1e-8
    assert list(w.index) == list(cov.index)
    pd.testing.assert_frame_equal(cov, before)


def test_erc_on_estimated_covariance():
    rets = _returns(300, _daily_sigma(12, seed=31), seed=12)
    est = estimate_covariance(rets)
    w = risk_parity_weights(est.cov)
    rc = risk_contributions(w, est.cov)
    assert np.abs(rc - 1 / 12).max() < 1e-8


def test_erc_raises_when_not_converged_and_on_bad_input():
    cov = _random_spd(10, 4)
    with pytest.raises(RuntimeError):
        risk_parity_weights(cov, max_iter=1)
    with pytest.raises(ValueError):
        risk_parity_weights(_cov(np.array([[0.04, 0.01], [0.02, 0.09]])))  # asymmetric
    with pytest.raises(ValueError):
        risk_parity_weights(_cov(np.array([[0.0, 0.0], [0.0, 0.09]])))
    with pytest.raises(ValueError):
        risk_parity_weights(pd.DataFrame(dtype=float))


def test_risk_contributions_and_portfolio_volatility():
    vols = np.array([0.2, 0.3])
    rho = 0.5
    cov = _cov(_equicorr(vols, rho))
    w = pd.Series([0.6, 0.4], index=cov.index)
    var = 0.36 * 0.04 + 0.16 * 0.09 + 2 * 0.6 * 0.4 * rho * 0.2 * 0.3
    assert portfolio_volatility(w, cov) == pytest.approx(np.sqrt(var))
    rc = risk_contributions(w, cov)
    assert rc.iloc[0] == pytest.approx((0.36 * 0.04 + 0.6 * 0.4 * rho * 0.06) / var)
    assert rc.sum() == pytest.approx(1.0)
    # Weights given in a different order are aligned by label.
    assert portfolio_volatility(w.iloc[::-1], cov) == pytest.approx(np.sqrt(var))
    with pytest.raises(ValueError):
        risk_contributions(pd.Series([1.0], index=["A00"]), cov)


# ── cap_weights ───────────────────────────────────────────


def test_cap_defaults_come_from_settings():
    assert MAX_POSITION_WEIGHT == 0.15
    assert MAX_SECTOR_WEIGHT == 0.35


def test_cap_feasible_input_unchanged():
    w = pd.Series(np.full(10, 0.1), index=[f"S{i}" for i in range(10)])
    sectors = {t: f"sec{i % 4}" for i, t in enumerate(w.index)}
    before = w.copy()
    out = cap_weights(w, sectors=sectors)
    pd.testing.assert_series_equal(out, w)
    pd.testing.assert_series_equal(w, before)


def test_cap_position_redistributes_pro_rata():
    w = pd.Series([0.40, 0.20, 0.10, 0.10, 0.08, 0.06, 0.06], index=list("ABCDEFG"))
    before = w.copy()
    out = cap_weights(w, max_weight=0.25)
    assert out.sum() == pytest.approx(1.0)
    assert out.max() <= 0.25 + 1e-12
    assert out["A"] == pytest.approx(0.25)
    # The uncapped names keep their relative proportions.
    rest = out.drop("A")
    assert np.allclose(rest / rest.sum(), w.drop("A") / w.drop("A").sum())
    pd.testing.assert_series_equal(w, before)


def test_cap_cascades_until_all_within_cap():
    w = pd.Series([0.5, 0.2, 0.14, 0.06, 0.05, 0.05], index=list("ABCDEF"))
    out = cap_weights(w, max_weight=0.2)
    assert out.sum() == pytest.approx(1.0)
    assert out.max() <= 0.2 + 1e-12
    assert (out >= 0).all()


def test_cap_sector_and_position_together():
    rng = np.random.default_rng(42)
    tickers = [f"T{i:02d}" for i in range(20)]
    raw = rng.dirichlet(np.full(20, 0.5))
    w = pd.Series(raw, index=tickers)
    sectors = {t: ("BANK" if i < 8 else "IT" if i < 12 else f"OTHER{i % 4}") for i, t in enumerate(tickers)}
    out = cap_weights(w, sectors=sectors)
    assert out.sum() == pytest.approx(1.0, abs=1e-12)
    assert out.max() <= MAX_POSITION_WEIGHT + 1e-12
    by_sector = out.groupby(pd.Series(sectors)).sum()
    assert by_sector.max() <= MAX_SECTOR_WEIGHT + 1e-12
    assert (out >= 0).all()


def test_cap_sector_overweight_scaled_and_excess_moved():
    w = pd.Series([0.25, 0.25, 0.1, 0.1, 0.1, 0.1, 0.1], index=list("ABCDEFG"))
    sectors = {"A": "BANK", "B": "BANK", "C": "IT", "D": "IT", "E": "AUTO", "F": "FMCG", "G": "PHARMA"}
    out = cap_weights(w, max_weight=0.3, sectors=sectors, max_sector_weight=0.35)
    assert out["A"] + out["B"] == pytest.approx(0.35)
    assert out["A"] == pytest.approx(out["B"])
    assert out.sum() == pytest.approx(1.0)
    assert np.allclose(out[list("CDEFG")], 0.13)


def test_cap_infeasible_raises():
    w = pd.Series(np.full(5, 0.2), index=list("ABCDE"))
    with pytest.raises(ValueError, match="infeasible"):
        cap_weights(w, max_weight=0.15)  # 0.15 x 5 < 1
    w10 = pd.Series(np.full(10, 0.1), index=list("ABCDEFGHIJ"))
    with pytest.raises(ValueError, match="infeasible"):
        cap_weights(w10, sectors={t: ("X" if t < "E" else "Y") for t in w10.index}, max_sector_weight=0.35)
    with pytest.raises(ValueError, match="no sector"):
        cap_weights(w10, sectors={"A": "X"})
    with pytest.raises(ValueError, match="sum to 1"):
        cap_weights(w10 * 0.5)
    with pytest.raises(ValueError):
        cap_weights(pd.Series([1.2, -0.2], index=["A", "B"]), max_weight=1.0)


def test_cap_erc_weights_end_to_end():
    cov = _random_spd(30, 9)
    w = risk_parity_weights(cov)
    sectors = {t: f"sec{i % 5}" for i, t in enumerate(cov.index)}
    out = cap_weights(w, max_weight=0.05, sectors=sectors, max_sector_weight=0.22)
    assert out.sum() == pytest.approx(1.0)
    assert out.max() <= 0.05 + 1e-12
    assert out.groupby(pd.Series(sectors)).sum().max() <= 0.22 + 1e-12


def test_no_dropna_in_module():
    import inspect

    import investbot.trading.risk as risk

    assert "dropna" not in inspect.getsource(risk)
    assert COV_MIN_OBS == 120 and COV_WINDOW == 250
