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
"""
import sys
import logging
import signal
from datetime import datetime

from investbot.config.settings import LOG_LEVEL
from investbot.data.db import init_db, SessionLocal
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
        logger.info(f"  Daily prices:    {price_stats['daily_total']} rows")
        logger.info(f"  Intraday prices: {price_stats['intraday_total']} rows")
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
    """Audit stored price rows against the ingestion rules; optionally delete the bad ones."""
    from collections import Counter, defaultdict
    from investbot.data.quality import audit_prices, purge_bad_rows

    purge = "--purge" in sys.argv[2:]
    session = SessionLocal()
    try:
        bad = audit_prices(session)
        if not bad:
            print("\n  Data quality OK: no bad price rows.\n")
            return

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
            print(f"\n  Purged {deleted} row(s). Run `python main.py ingest` to re-fetch them.\n")
            return
        print("\n  Re-run with --purge to delete these rows.\n")
        sys.exit(1)
    finally:
        session.close()


# ── CLI dispatch ──────────────────────────────────────────
COMMANDS = {
    "setup": cmd_setup,
    "ingest": cmd_ingest,
    "run": cmd_run,
    "status": cmd_status,
    "gaps": cmd_gaps,
    "dq": cmd_dq,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)

    COMMANDS[sys.argv[1]]()
