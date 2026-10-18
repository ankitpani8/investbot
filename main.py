"""
InvestBot — Main entry point.

Usage:
    python main.py setup      # Create tables + seed reference data
    python main.py ingest     # Run full initial data ingestion (prices + financials)
    python main.py run        # Start scheduler (background jobs)
    python main.py status     # Show database statistics
    python main.py gaps       # Detect price data gaps
    python main.py dq         # Audit stored price rows (exit 1 if any are bad)
    python main.py dq --purge # ...and delete the bad rows (next ingest re-fetches them)
    python main.py score [TICKER] [--timeframe 1d|15m] [--as-of YYYY-MM-DD] [--no-cache]
                              # Technical scores, ranked (one asset in detail with TICKER)
    python main.py evaluate [--horizons 1,5,20] [--start YYYY-MM-DD] [--end YYYY-MM-DD]
                            [--universe index|active]
                              # index (default when index_membership has NIFTY50 rows): point-in-time
                              # members, plus the active universe on the same splits for comparison
                              # Walk-forward forward-return evaluation → docs/reports/eval_<date>.md
"""
import sys
import logging
import signal
from datetime import datetime

from investbot.config.settings import LOG_LEVEL
from investbot.data.db import init_db, SessionLocal
from investbot.data.index_membership import membership_summary
from investbot.data.seed import run_seed
from investbot.data.models import (
    AssetClass, Exchange, Asset, PriceDaily, PriceIntraday,
    Financial, FinancialDataAvailability,
)

# ── Logging setup ─────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("investbot")


def cmd_setup():
    """Create all database tables and seed reference data."""
    logger.info("Creating database tables...")
    init_db()
    logger.info("All tables created.")

    logger.info("Seeding reference data...")
    session = SessionLocal()
    try:
        run_seed(session)
    finally:
        session.close()
    logger.info("Setup complete.")


def cmd_ingest():
    """Run initial full data ingestion — prices + financials for all NIFTY50."""
    from investbot.data.ingestion.prices import ingest_all_prices
    from investbot.data.ingestion.financials import ingest_all_financials

    session = SessionLocal()
    try:
        logger.info("=" * 60)
        logger.info("STARTING FULL DATA INGESTION")
        logger.info("=" * 60)

        start = datetime.now()

        logger.info("\n--- Phase 1/2: Price Data ---")
        price_stats = ingest_all_prices(session, asset_type="stock")

        logger.info("\n--- Phase 2/2: Financial Statements ---")
        fin_stats = ingest_all_financials(session, asset_type="stock")

        elapsed = (datetime.now() - start).total_seconds()
        logger.info("=" * 60)
        logger.info(f"INGESTION COMPLETE in {elapsed:.0f}s")
        for label, key in (("Daily prices:   ", "daily"), ("Intraday prices:", "intraday")):
            logger.info(
                f"  {label} inserted {price_stats[key + '_inserted']}, "
                f"updated {price_stats[key + '_updated']}, "
                f"rejected {price_stats[key + '_rejected']}"
            )
        logger.info(
            f"  Corporate actions: inserted {price_stats['actions_inserted']}, "
            f"updated {price_stats['actions_updated']}"
        )
        logger.info(
            f"  Price downloads: {price_stats['full_downloads']} full, "
            f"{price_stats['incremental_downloads']} incremental"
        )
        logger.info(f"  Annual financials:    {fin_stats['annual']} records")
        logger.info(f"  Quarterly financials: {fin_stats['quarterly']} records")
        if price_stats['errors'] or fin_stats['errors']:
            logger.warning(f"  Errors: {price_stats['errors'] + fin_stats['errors']}")
        logger.info("=" * 60)
    finally:
        session.close()


def cmd_run():
    """Start the scheduler for continuous data refresh."""
    from investbot.scheduler.jobs import create_scheduler

    scheduler = create_scheduler()
    scheduler.start()
    logger.info("Scheduler started. Press Ctrl+C to stop.")

    # Print upcoming jobs
    for job in scheduler.get_jobs():
        logger.info(f"  Job: {job.name} | Next run: {job.next_run_time}")

    # Keep main thread alive, handle graceful shutdown
    def shutdown(signum, frame):
        logger.info("Shutting down scheduler...")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        while True:
            import time
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped.")


def cmd_status():
    """Print database statistics."""
    session = SessionLocal()
    try:
        n_classes = session.query(AssetClass).count()
        n_exchanges = session.query(Exchange).count()
        n_assets = session.query(Asset).count()
        n_active = session.query(Asset).filter_by(is_active=True).count()
        n_daily = session.query(PriceDaily).count()
        n_intraday = session.query(PriceIntraday).count()
        n_financials = session.query(Financial).count()
        n_availability = session.query(FinancialDataAvailability).count()

        print("\n" + "=" * 50)
        print("  InvestBot — Database Status")
        print("=" * 50)
        print(f"  Asset classes:       {n_classes}")
        print(f"  Exchanges:           {n_exchanges}")
        print(f"  Assets (total):      {n_assets}")
        print(f"  Assets (active):     {n_active}")
        print(f"  Daily price rows:    {n_daily:,}")
        print(f"  Intraday price rows: {n_intraday:,}")
        print(f"  Financial records:   {n_financials}")
        print(f"  Data availability:   {n_availability}")
        ms = membership_summary(session)
        print(f"  Index membership:    {ms['rows']} rows, {ms['members_without_prices']} members without prices")
        print("=" * 50 + "\n")

        # Show per-asset breakdown for first 5
        assets = session.query(Asset).filter_by(is_active=True, asset_type="stock").limit(5).all()
        if assets:
            print("  Sample asset data coverage:")
            for a in assets:
                daily_count = session.query(PriceDaily).filter_by(asset_id=a.id).count()
                intra_count = session.query(PriceIntraday).filter_by(asset_id=a.id).count()
                fin_count = session.query(Financial).filter_by(asset_id=a.id).count()
                print(f"    {a.ticker:<20s}  daily={daily_count:<6}  intraday={intra_count:<6}  financials={fin_count}")
            print()
    finally:
        session.close()


def cmd_gaps():
    """Detect and report price data gaps."""
    from investbot.data.ingestion.prices import detect_gaps

    session = SessionLocal()
    try:
        assets = session.query(Asset).filter_by(is_active=True, asset_type="stock").all()
        any_gaps = False
        for asset in assets:
            gaps = detect_gaps(session, asset)
            if gaps:
                any_gaps = True
                print(f"\n  {asset.ticker} — {len(gaps)} gap(s):")
                for start, end, bdays in gaps:
                    print(f"    {start} → {end} ({bdays} business days)")
        if not any_gaps:
            print("\n  No significant gaps detected in any asset.\n")
    finally:
        session.close()


def cmd_dq():
    """Audit stored price rows against the ingestion rules (optionally delete the bad ones) and check splits."""
    from collections import Counter, defaultdict
    from investbot.data.models import Base
    from investbot.data.quality import audit_prices, audit_splits, purge_bad_rows

    purge = "--purge" in sys.argv[2:]
    session = SessionLocal()
    try:
        Base.metadata.create_all(session.get_bind())   # corporate_actions may be missing on an existing DB
        bad = audit_prices(session)
        split_issues = audit_splits(session)
        failed = False

        if not bad:
            print("\n  Data quality OK: no bad price rows.")
        else:
            by_key = defaultdict(list)
            for b in bad:
                by_key[(b.table, b.ticker)].append(b)

            print(f"\n  {len(bad)} bad price row(s) in {len(by_key)} (table, asset) group(s):")
            for (table, ticker), rows in sorted(by_key.items()):
                reasons = dict(Counter(r.reason for r in rows))
                first, last = rows[0].when, rows[-1].when
                print(f"    {table:<15s} {ticker:<16s} {len(rows):>3}  {reasons}  {first} → {last}")

            print("\n  By table and reason:")
            for (table, reason), n in sorted(Counter((b.table, b.reason) for b in bad).items()):
                dates = sorted({str(b.when) for b in bad if b.table == table and b.reason == reason})
                shown = ", ".join(dates[:8]) + (f" (+{len(dates) - 8} more)" if len(dates) > 8 else "")
                print(f"    {table:<15s} {reason:<20s} {n:>4} row(s) on {len(dates)} date(s): {shown}")

            if purge:
                deleted = purge_bad_rows(session, bad)
                print(f"\n  Purged {deleted} row(s). Run `python main.py ingest` to re-fetch them.")
            else:
                print("\n  Re-run with --purge to delete these rows.")
                failed = True

        if not split_issues:
            print("\n  Split check OK: no unadjusted splits.\n")
        else:
            print(f"\n  {len(split_issues)} unadjusted split(s): history before the ex-date is not re-adjusted:")
            for i in split_issues:
                print(f"    {i.ticker:<16s} ex {i.ex_date}  factor {i.factor:g}  prev close / ex open = {i.observed_ratio:.3f}")
            print("\n  --purge does not touch these. Re-adjusting history is a human decision.\n")
            failed = True

        if failed:
            sys.exit(1)
    finally:
        session.close()


def cmd_score():
    """Score one asset (detail) or every active stock (ranked table) and persist the results."""
    import pandas as pd
    from investbot.scoring.technical.scorer import score, score_universe

    args = sys.argv[2:]
    opts = {"--timeframe": "1d", "--as-of": None}
    positional = []
    i = 0
    while i < len(args):
        if args[i] in opts and i + 1 < len(args):
            opts[args[i]] = args[i + 1]
            i += 2
        elif args[i] == "--no-cache":
            opts["--no-cache"] = True
            i += 1
        else:
            positional.append(args[i])
            i += 1
    timeframe = opts["--timeframe"]
    as_of = datetime.strptime(opts["--as-of"], "%Y-%m-%d").date() if opts["--as-of"] else None
    use_cache = not opts.get("--no-cache", False)

    init_db()   # creates technical_scores on an existing DB
    logging.getLogger("investbot.scoring").setLevel(logging.WARNING)
    session = SessionLocal()
    try:
        if positional:
            ticker = positional[0]
            if session.query(Asset).filter_by(ticker=ticker).first() is None and "." not in ticker:
                ticker += ".NS"
            s = score(ticker, timeframe, as_of, session=session, use_cache=use_cache)
            print(f"\n  {s.ticker} [{s.timeframe}] bar {s.as_of}  status={s.status}"
                  f"{'  (cached)' if s.cached else ''}{'  STALE' if s.is_stale else ''}")
            if s.message:
                print(f"  {s.message}")
            print(f"  TECH_SCORE {s.tech_score:+.3f}   regime {s.regime} "
                  f"(confidence {s.regime_confidence:.2f})")
            print(f"  bars {s.bars_available}  warmup ok {s.warmup_satisfied}  "
                  f"staleness {s.staleness_days} business day(s)")
            print("  families: " + "  ".join(f"{k} {v:+.2f}" for k, v in s.family_scores.items()))
            for r in s.reasons:
                print(f"    - {r}")
            print()
            return

        df = score_universe("stock", timeframe, as_of, session=session, use_cache=use_cache)
        cols = ["rank", "ticker", "as_of", "tech_score", "score_trend", "score_momentum",
                "score_volume", "score_sr", "score_mtf", "regime", "regime_confidence",
                "status", "is_stale", "cached", "top_reason"]
        out = df[cols].copy()
        width = 10 if timeframe == "1d" else 16
        out["as_of"] = out["as_of"].map(lambda t: "" if t is None or pd.isna(t) else str(t)[:width])
        with pd.option_context("display.width", 250, "display.max_rows", None,
                               "display.max_colwidth", 60, "display.float_format", "{:+.3f}".format):
            print(out.to_string(index=False))
        n_ok = int((df["status"] == "ok").sum())
        print(f"\n  {n_ok}/{len(df)} scored, {int(df['cached'].sum())} from cache, "
              f"{int(df['is_stale'].sum())} stale.\n")
    finally:
        session.close()


def cmd_evaluate():
    """Walk-forward forward-return evaluation of every technical signal and the baselines."""
    from investbot.evaluation.loader import has_index_membership
    from investbot.evaluation.report import run_evaluation, universe_note

    index_code = "NIFTY50"
    args = sys.argv[2:]
    opts = {"--horizons": "1,5,20", "--start": None, "--end": None, "--universe": None}
    i = 0
    while i < len(args):
        if args[i] in opts and i + 1 < len(args):
            opts[args[i]] = args[i + 1]
            i += 2
        else:
            print(f"unknown argument {args[i]!r}\n{__doc__}")
            sys.exit(1)
    horizons = tuple(int(h) for h in opts["--horizons"].split(",") if h.strip())
    start = datetime.strptime(opts["--start"], "%Y-%m-%d").date() if opts["--start"] else None
    end = datetime.strptime(opts["--end"], "%Y-%m-%d").date() if opts["--end"] else None
    universe = opts["--universe"]
    if universe not in (None, "index", "active"):
        print(f"--universe must be index or active, not {universe!r}")
        sys.exit(1)

    logging.getLogger("investbot.scoring").setLevel(logging.WARNING)
    session = SessionLocal()
    try:
        has_members = has_index_membership(session, index_code)
        if universe is None:
            universe = "index" if has_members else "active"
            if not has_members:
                print(f"\n  WARNING: index_membership has no {index_code} rows (run "
                      "`python main.py setup`); evaluating today's active stocks instead, "
                      "with survivorship bias.\n")
        elif universe == "index" and not has_members:
            print(f"index_membership has no {index_code} rows; run `python main.py setup` first")
            sys.exit(1)
        logger.info("Evaluating horizons %s, universe %s (one asset at a time)...",
                    horizons, universe)
        report, csv, table, info = run_evaluation(horizons, start, end, session=session,
                                                  universe=universe, index_code=index_code)
    finally:
        session.close()
    n_prom = int(table["promising"].sum())
    print(f"\n  {universe_note(info).replace('**', '')}\n")
    print(f"  Report:  {report}")
    print(f"  Metrics: {csv}")
    print(f"  Promising signal × horizon tests: {n_prom}\n")


# ── CLI dispatch ──────────────────────────────────────────
COMMANDS = {
    "setup": cmd_setup,
    "ingest": cmd_ingest,
    "run": cmd_run,
    "status": cmd_status,
    "gaps": cmd_gaps,
    "dq": cmd_dq,
    "score": cmd_score,
    "evaluate": cmd_evaluate,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)

    COMMANDS[sys.argv[1]]()
