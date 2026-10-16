"""
Smoke test for scoring/technical/indicators.py (TA-Lib backend).

Picks the asset with the most daily history in the DB, runs the full
indicator pipeline on it, and prints a diagnostic summary. Saves the
full indicator DataFrame to a parquet file for manual inspection.

Run:
    python -m scripts.smoke_indicators
or:
    python scripts/smoke_indicators.py

Exit codes:
    0 — at least MIN_EXPECTED_INDICATORS columns populated on latest bar
    1 — DB not populated, or coverage below threshold
"""
from __future__ import annotations

import logging
import sys

import pandas as pd
from sqlalchemy import func, select

from investbot.data.db import SessionLocal
from investbot.data.models import Asset, PriceDaily, PriceIntraday
from investbot.scoring.technical.indicators import (
    add_all_indicators,
    list_indicator_columns,
)

MIN_EXPECTED_INDICATORS = 90   # sanity floor — we target ~115-120

# Representative columns from each family — handy for eyeballing sanity.
SAMPLE_COLS = [
    # Momentum
    "RSI_14", "MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9",
    "ADX_14", "CCI_20", "STOCHk_14_3_3", "STOCHd_14_3_3",
    "MFI_14", "WILLR_14", "ULTOSC_7_14_28",
    # Overlap / MAs
    "SMA_20", "SMA_50", "SMA_200", "EMA_20", "EMA_50", "EMA_200",
    "BBU_20_2.0", "BBM_20_2.0", "BBL_20_2.0", "BBP_20_2.0", "BBB_20_2.0",
    "HT_TRENDLINE", "MAMA",
    # Volatility
    "ATR_14", "NATR_14",
    # Volume
    "OBV", "AD", "ADOSC_3_10",
    # Cycle
    "HT_DCPERIOD", "HT_TRENDMODE",
    # Extensions
    "ICHI_TENKAN_9", "ICHI_KIJUN_26", "ICHI_SPAN_A_26", "ICHI_SPAN_B_26",
    "SUPERT_10_3.0", "SUPERTd_10_3.0",
    "KCM_20_2.0", "KCU_20_2.0", "KCL_20_2.0",
    "SQUEEZE_ON", "SQUEEZE_INTENSITY",
    "HA_OPEN", "HA_HIGH", "HA_LOW", "HA_CLOSE",
    "VTXP_14", "VTXM_14",
    "RVOL_20", "RVOL_50",
    "LOGRET_1", "PCTRET_1", "CUMLOGRET_1",
]


def _pick_best_asset(session) -> tuple[Asset | None, int]:
    """Return the active stock asset with the most daily price rows."""
    stmt = (
        select(Asset, func.count(PriceDaily.id).label("n"))
        .join(PriceDaily, PriceDaily.asset_id == Asset.id)
        .where(Asset.is_active.is_(True), Asset.asset_type == "stock")
        .group_by(Asset.id)
        .order_by(func.count(PriceDaily.id).desc())
        .limit(1)
    )
    row = session.execute(stmt).first()
    if row is None:
        return None, 0
    return row[0], row[1]


def _load_daily_df(session, asset_id: int) -> pd.DataFrame:
    rows = session.execute(
        select(PriceDaily)
        .where(PriceDaily.asset_id == asset_id)
        .order_by(PriceDaily.date)
    ).scalars().all()
    df = pd.DataFrame([{
        "date": r.date,
        "open": r.open, "high": r.high, "low": r.low,
        "close": r.close, "adj_close": r.adj_close,
        "volume": r.volume,
    } for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def _load_intraday_df(session, asset_id: int, interval: str = "15m") -> pd.DataFrame:
    rows = session.execute(
        select(PriceIntraday)
        .where(PriceIntraday.asset_id == asset_id, PriceIntraday.interval == interval)
        .order_by(PriceIntraday.timestamp)
    ).scalars().all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "timestamp": r.timestamp,
        "open": r.open, "high": r.high, "low": r.low,
        "close": r.close, "volume": r.volume,
    } for r in rows])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp").sort_index()


def _summarize(df: pd.DataFrame, label: str) -> int:
    """Print diagnostic summary and return count of columns populated on last bar."""
    added = list_indicator_columns(df)
    latest = df.iloc[-1]
    latest_filled = latest.notna().sum()
    latest_total = len(latest)
    coverage_pct = 100.0 * latest_filled / latest_total if latest_total else 0.0

    # Warmup map: first-valid-index per column.
    first_valid = {c: df[c].first_valid_index() for c in added}
    warmup_bars = [
        df.index.get_loc(first_valid[c])
        for c in added
        if first_valid[c] is not None
    ]
    all_nan = [c for c in added if first_valid[c] is None]

    print(f"\n=== {label} ===")
    print(f"Rows                : {len(df)}")
    print(f"Total columns       : {df.shape[1]}")
    print(f"Indicator columns   : {len(added)}")
    print(f"All-NaN indicators  : {len(all_nan)}  {all_nan[:5] if all_nan else ''}")
    if warmup_bars:
        print(
            f"Warmup (bars to 1st value)  min={min(warmup_bars)}  "
            f"median={int(pd.Series(warmup_bars).median())}  "
            f"max={max(warmup_bars)}"
        )
    print(f"Latest-bar coverage : {latest_filled}/{latest_total} ({coverage_pct:.1f}%)")

    available_samples = [c for c in SAMPLE_COLS if c in df.columns]
    missing_samples = [c for c in SAMPLE_COLS if c not in df.columns]
    if available_samples:
        print(f"\n Sample values on latest bar ({df.index[-1]}):")
        for col in available_samples:
            val = df.iloc[-1][col]
            print(f"   {col:<22} = {val}")
    if missing_samples:
        print(f"\n Missing sample columns: {missing_samples}")

    # Sanity checks: RSI and ADX should be in [0, 100].
    for col, lo, hi in [("RSI_14", 0, 100), ("ADX_14", 0, 100),
                        ("STOCHk_14_3_3", 0, 100), ("MFI_14", 0, 100)]:
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) and (vals.min() < lo - 0.01 or vals.max() > hi + 0.01):
                print(f" ⚠️  SANITY: {col} out of [{lo}, {hi}]: "
                      f"min={vals.min():.3f} max={vals.max():.3f}")

    return latest_filled


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    log = logging.getLogger("smoke_indicators")

    session = SessionLocal()
    try:
        asset, n_daily = _pick_best_asset(session)
        if asset is None:
            log.error("No daily price data in DB. Run Phase 1 ingestion first.")
            return 1

        log.info("Selected %s (asset_id=%d) with %d daily rows",
                 asset.ticker, asset.id, n_daily)

        # ── Daily run ──────────────────────────────────────
        df_daily = _load_daily_df(session, asset.id)
        log.info("Daily DataFrame: shape=%s, range=%s..%s",
                 df_daily.shape, df_daily.index[0].date(), df_daily.index[-1].date())
        df_daily = add_all_indicators(df_daily)
        daily_latest_filled = _summarize(df_daily, f"{asset.ticker} — DAILY")

        out_daily = f"/tmp/{asset.ticker.replace('.', '_')}_daily_indicators.parquet"
        df_daily.to_parquet(out_daily)
        log.info("Saved daily indicator frame to %s", out_daily)

        # ── Intraday run (optional, best-effort) ───────────
        df_intra = _load_intraday_df(session, asset.id)
        if not df_intra.empty:
            log.info("Intraday DataFrame: shape=%s, range=%s..%s",
                     df_intra.shape, df_intra.index[0], df_intra.index[-1])
            df_intra = add_all_indicators(df_intra)
            _summarize(df_intra, f"{asset.ticker} — INTRADAY 15m")
            out_intra = f"/tmp/{asset.ticker.replace('.', '_')}_intraday_indicators.parquet"
            df_intra.to_parquet(out_intra)
            log.info("Saved intraday indicator frame to %s", out_intra)
        else:
            log.info("No intraday data for %s — skipping intraday run", asset.ticker)

        # ── Pass/fail gate ─────────────────────────────────
        if daily_latest_filled < MIN_EXPECTED_INDICATORS:
            log.error(
                "FAIL: only %d columns populated on latest daily bar (expected ≥ %d)",
                daily_latest_filled, MIN_EXPECTED_INDICATORS,
            )
            return 1
        log.info("PASS: %d columns populated on latest daily bar", daily_latest_filled)
        return 0

    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
