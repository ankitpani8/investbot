---
id: 060-scorer
depends: [050-confluence, 000-ingest-refresh-window]
---
# scorer.py: score(ticker) orchestrator with caching

## Requirements
- New module `investbot/scoring/technical/scorer.py`. It is the **only** technical module that does DB I/O.
- `score(ticker, timeframe="1d", as_of=None) -> TechnicalScore` (a dataclass) with fields: ticker, as_of bar timestamp, tech_score, family scores, regime, regime_confidence, reasons, data_quality flags (bars available, warmup satisfied, staleness).
- `as_of` lets backtests score historically: only bars ≤ as_of are loaded. This is the backtest no-lookahead entry point.
- `score_universe(asset_type="stock", ...) -> pd.DataFrame` scores all active assets, keeping memory under control (one asset at a time).
- Persist results to the technical-scores table defined in the master plan schema (check models.py; add the table if it's missing, schema-first, with an Alembic-free migration consistent with the existing `init_db`).
- Cache: skip recomputation when the latest bar hasn't changed since the last stored score.
- CLI: `python main.py score [TICKER]` prints a ranked table.

## Acceptance criteria
- Tests on the scratch DB or an in-memory SQLite with fixture data: `score(as_of=t)` equals scoring a DB truncated at t; the cache hit path; a missing or insufficient-data asset gives a flagged result, not an exception.
- `python main.py score` runs on the scratch DB. The suite is green.
