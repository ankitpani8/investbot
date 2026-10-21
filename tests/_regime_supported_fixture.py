"""Hand-built StudyResult for the supported-set report tests (no pipeline run)."""
from __future__ import annotations

from datetime import date

import pandas as pd

from investbot.config.scoring_weights import REGIME_NAMES
from investbot.evaluation import regime_study as rs

#: (group, name, target, section, horizon, expected_sign) in table order.
BASE_ROWS = [
    ("H1", "trend − range", "FWD_ER_20", "conditional", 20, 1),
    ("H2", "chaos − rest", "FWD_VOLX_20", "conditional", 20, 1),
    ("H3", "chaos − rest", "FWD_MDD_20", "conditional", 20, -1),
    ("H4", "REGIME_SCORE_CHAOS", "FWD_VOLX_20", "rank", 20, 1),
    ("H4", "REGIME_SCORE_CHAOS", "FWD_VOLX_60", "rank", 60, 1),
    ("H4", "REGIME_SCORE_CHAOS", "FWD_MDD_20", "rank", 20, -1),
    ("H4", "VOL_RATIO", "FWD_VOLX_20", "rank", 20, 1),
    ("H4", "VOL_RATIO", "FWD_VOLX_60", "rank", 60, 1),
    ("H4", "VOL_RATIO", "FWD_MDD_20", "rank", 20, -1),
    ("H5", "CONF_EMA_10", "HIT_20", "calibration", 20, 1),
]
H4_VOLX = {("REGIME_SCORE_CHAOS", "FWD_VOLX_20"), ("REGIME_SCORE_CHAOS", "FWD_VOLX_60"),
           ("VOL_RATIO", "FWD_VOLX_20"), ("VOL_RATIO", "FWD_VOLX_60")}


def rows_frame(supported: set[tuple[str, str]] | set[tuple[str, str, str]]) -> pd.DataFrame:
    """Rows of ``BASE_ROWS``; a row is "supported" (all three rules) when ``(name, target)``
    is in ``supported`` (or ``(group, name, target)``)."""
    rows = []
    for i, (g, n, t, sec, h, sign) in enumerate(BASE_ROWS):
        ok = (n, t) in supported or (g, n, t) in supported
        v = "supported" if ok else "not supported"
        rows.append({
            "group": g, "name": n, "target": t, "section": sec, "horizon": h,
            "expected_sign": sign, "n_days": 690 - i, "mean": 0.05 * sign * (1 if ok else 0.1),
            "t_nw": 3.0 if ok else 0.4, "q": 0.01 if ok else 0.6, "block_share": 0.8,
            "verdict": v, "acf1": 0.9, "lag_auto": 40 + i, "t_auto": 2.5 if ok else 0.3,
            "p_auto": 0.01, "q_auto": 0.02 if ok else 0.7, "boot_block": 41 + i,
            "p_boot": 0.002, "q_boot": 0.01 if ok else 0.7,
            "verdict_auto": v, "verdict_boot": v})
    return pd.DataFrame(rows)


def study(supported, h6: pd.DataFrame | None = None) -> rs.StudyResult:
    names = list(REGIME_NAMES.values())
    base = {"ticker": "ALL", "n_bars": 1000, "flips_per_1000": 12.0,
            **{f"share_{n}": 1 / len(names) for n in names}}
    hurst = {"n": 1000, "mean": 0.53, "p05": 0.4, "median": 0.53, "p95": 0.65}
    tests = rows_frame(supported)
    t0 = pd.Timestamp("2022-01-03")
    oos = pd.bdate_range(t0, periods=5)
    return rs.StudyResult(
        splits=[], oos=oos,
        base_rates_real=pd.DataFrame([base]), base_rates_null=pd.DataFrame([base]),
        hurst_real=hurst, hurst_null=hurst, tests=tests,
        group_means=pd.DataFrame([{"target": "FWD_ER_20", "horizon": 20,
                                   **{n: 0.01 for n in names}, **{f"{n}_days": 600 for n in names}}]),
        null_conditional=pd.DataFrame([{"name": "trend − range", "target": "FWD_ER_20",
                                        "mean": 0.0, "t_nw": 0.1}]),
        null_rank=pd.DataFrame([{"name": "VOL_RATIO", "target": "FWD_VOLX_20",
                                 "mean": 0.01, "t_nw": 0.5}]),
        hit_quintiles=pd.DataFrame([{"span": 10, "horizon": 20, **{f"Q{k}": 0.0 for k in range(1, 6)}}]),
        churn=pd.DataFrame([{"span": 10, "n_pairs": 900, "mean_abs_change": 0.01,
                             **{f"cross_{lv:g}_per_1000": 5.0 for lv in rs.CROSSING_LEVELS}}]),
        h6=h6 if h6 is not None else pd.DataFrame())


INFO = {"n_assets": 12, "skipped": [], "n_rows": 10800, "first": pd.Timestamp("2022-01-03"),
        "last": pd.Timestamp("2022-01-07")}


def render(res: rs.StudyResult) -> str:
    from scripts import regime_study as runner
    return runner.render_report(res, INFO, date(2026, 10, 21), elapsed_s=1.0, rss_mb=100.0,
                                git_hash="abc1234", end=None)
