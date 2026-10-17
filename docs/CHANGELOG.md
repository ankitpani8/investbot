# Changelog

- 2026-10-17: Migrated Phase 1 and the Phase 2 indicator layer from Google Drive to WSL and git. Added CLAUDE.md, contract tests and the backlog. Fixed the DB path (`.env` was pointing SQLite inside Google Drive).
- 2026-10-17: Fixed the daily smoke-test failure. 50 NULL-close bars on 2026-10-16 (plus 108 zero-volume placeholder bars) poisoned the indicators. Ingestion now validates every price row (`investbot/data/quality.py`), added `python main.py dq [--purge]`, purged 158 bad rows, and added NaN-propagation and ingestion tests.
- 2026-10-17 005-housekeeping: TATAMOTORS.NS → TMPV.NS (old row deactivated, not deleted), removed duplicate smoke script, pct_change(fill_method=None), FutureWarning is an error in pytest, seed no longer mutates TAX_REGIMES
- 2026-10-17 000-ingest-refresh-window: price ingestion upserts a trailing window (last 5 daily sessions, last 2 intraday trading days), skips today's daily bar before 16:00 IST and the forming 15m candle, never overwrites a good row with a rejected one, and logs inserted/updated/rejected per asset.
- 2026-10-17 010-candle-patterns: added `candle_patterns.py`: all 61 TA-Lib CDL* patterns as raw `CDL_*` and normalised `CDLN_*` columns plus `CDL_BULL_COUNT`/`CDL_BEAR_COUNT`/`CDL_NET_SCORE` (warmup NaN, no-lookahead tests). Not wired into `add_all_indicators`
