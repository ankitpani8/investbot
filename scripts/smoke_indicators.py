"""
Smoke test for scoring/technical/indicators.py (TA-Lib backend).

Runs the full indicator pipeline on one asset (DEFAULT_SMOKE_TICKER, or the
ticker given on the command line), and prints a diagnostic summary. Saves the
full indicator DataFrame to a CSV file in ./tmp_output/ for manual
inspection.

Run from project root:
    python -m scripts.smoke_indicators [TICKER]

If the default ticker has no daily rows, the asset with the most rows is used
instead, with a printed warning. An explicit TICKER is never substituted.

Exit codes:
    0 — at least MIN_EXPECTED_INDICATORS columns populated on latest bar
    1 — DB not populated, coverage below threshold, or unknown/empty explicit ticker
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import func, select

from investbot.data.db import SessionLocal
from investbot.data.models import Asset, PriceDaily, PriceIntraday
from investbot.scoring.technical.indicators import (
    add_all_indicators,
    list_indicator_columns,
)

MIN_EXPECTED_INDICATORS = 90   # sanity floor — we target ~115

# Fixed asset under test, so purging bad rows elsewhere cannot change it.
DEFAULT_SMOKE_TICKER = "RELIANCE.NS"

# Project-root-relative output directory. Works on Windows, macOS, Linux.
# __file__ = scripts/smoke_indicators.py; parent.parent = project root.
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "tmp_output"

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
        .order_by(func.count(PriceDaily.id).desc(), Asset.ticker)
        .limit(1)
    )
    row = session.execute(stmt).first()
    if row is None:
        return None, 0
    return row[0], row[1]


class SmokeAssetError(Exception):
    """The requested asset cannot be smoke-tested."""


def select_smoke_asset(session, ticker: str | None = None) -> tuple[Asset, int, str | None]:
    """
    Choose the asset under test. Returns (asset, daily row count, warning or None).

    ``ticker=None`` means DEFAULT_SMOKE_TICKER, with a fallback to the asset with the
    most rows (and a warning naming both) when it has no daily rows. An explicit
    ticker is never substituted: unknown or empty raises SmokeAssetError.
    """
    explicit = ticker is not None
    wanted = ticker if explicit else DEFAULT_SMOKE_TICKER

    asset = session.execute(select(Asset).where(Asset.ticker == wanted)).scalar_one_or_none()
    if asset is None and explicit:
        raise SmokeAssetError(f"Unknown ticker {wanted!r}: no such asset in the DB")
    n = 0
    if asset is not None:
        n = session.execute(
            select(func.count(PriceDaily.id)).where(PriceDaily.asset_id == asset.id)
        ).scalar_one()
    if n:
        return asset, n, None
    if explicit:
        raise SmokeAssetError(f"{wanted} has no daily price rows")

    best, best_n = _pick_best_asset(session)
    if best is None:
        raise SmokeAssetError("No daily price data in DB. Run Phase 1 ingestion first.")
    warning = (f"{DEFAULT_SMOKE_TICKER} has no daily rows; falling back to {best.ticker} "
               f"({best_n} rows). The asset under test is NOT the default.")
    return best, best_n, warning


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


def _save(df: pd.DataFrame, filename: str) -> Path:
    """Save DataFrame to CSV under ./tmp_output/, return full path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=True)
    return path


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) > 1:
        print("usage: python -m scripts.smoke_indicators [TICKER]", file=sys.stderr)
        return 2
    ticker = argv[0] if argv else None

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    log = logging.getLogger("smoke_indicators")

    session = SessionLocal()
    try:
        try:
            asset, n_daily, warning = select_smoke_asset(session, ticker)
        except SmokeAssetError as e:
            log.error("%s", e)
            return 1
        if warning:
            print(f"WARNING: {warning}", file=sys.stderr)
            log.warning("%s", warning)

        log.info("Selected %s (asset_id=%d) with %d daily rows",
                 asset.ticker, asset.id, n_daily)

        # ── Daily run ──────────────────────────────────────
        df_daily = _load_daily_df(session, asset.id)
        log.info("Daily DataFrame: shape=%s, range=%s..%s",
                 df_daily.shape, df_daily.index[0].date(), df_daily.index[-1].date())
        df_daily = add_all_indicators(df_daily)
        daily_latest_filled = _summarize(df_daily, f"{asset.ticker} — DAILY")

        out_daily = _save(df_daily, f"{asset.ticker.replace('.', '_')}_daily_indicators.csv")
        log.info("Saved daily indicator frame to %s", out_daily)

        # ── Intraday run (optional, best-effort) ───────────
        df_intra = _load_intraday_df(session, asset.id)
        if not df_intra.empty:
            log.info("Intraday DataFrame: shape=%s, range=%s..%s",
                     df_intra.shape, df_intra.index[0], df_intra.index[-1])
            df_intra = add_all_indicators(df_intra)
            _summarize(df_intra, f"{asset.ticker} — INTRADAY 15m")
            out_intra = _save(df_intra, f"{asset.ticker.replace('.', '_')}_intraday_indicators.csv")
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
