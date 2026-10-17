---
id: 000-ingest-refresh-window
depends: []
---
# Ingestion: refresh the recent window and never store an unfinished daily bar

Decision (made by Ankit): **upsert a trailing refresh window**, and also never store today's daily bar before the session is final.
Today ingestion skips dates already in the table, so a bad or partial bar never heals, and Yahoo revises recent bars.

## Requirements
- `investbot/config/settings.py`: add `DAILY_REFRESH_SESSIONS = 5`, `INTRADAY_REFRESH_DAYS = 2` and `DAILY_BAR_FINAL_AFTER = "16:00"` (IST).
- Daily ingestion: re-fetch and **upsert** (update the existing row, insert the missing one) the last `DAILY_REFRESH_SESSIONS` trading sessions per asset. Older history is still insert-only.
- Don't store a daily bar dated today (Asia/Kolkata) before `DAILY_BAR_FINAL_AFTER`.
- Intraday: upsert the last `INTRADAY_REFRESH_DAYS` days. Drop the still-forming 15m candle (the one whose end time is in the future).
- Every upserted row still goes through `investbot/data/quality.py`. A bad row is never written over a good one.
- The upsert must be idempotent: running ingest twice gives an identical DB.
- Log per asset: inserted, updated, rejected counts.

## Acceptance criteria
- Tests with stubbed yfinance (no network) cover: a revised bar gets updated; a NULL-close bar arriving over a good row leaves the good row; today's bar before 16:00 IST is skipped and after 16:00 it's stored; the forming 15m candle is dropped; a second run makes no changes.
- `python -m pytest -q` is green and `python main.py dq` is clean on the scratch DB.
