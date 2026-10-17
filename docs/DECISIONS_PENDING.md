# Decisions

Agents add open questions under **Pending**. Ankit moves them to **Decided** with the answer.

## Pending
- 2026-10-17 **Meaning of "days" in `INTRADAY_REFRESH_DAYS`** (000-ingest-refresh-window). Chose: the last N *trading* days present in the downloaded data, so a Monday run still refreshes Friday. Alternative: N calendar days back from now (a Monday run with N=2 would refresh nothing from the previous week). The daily window works the same way (last N sessions in the download). To switch, change the `key`/window computation passed to `_refresh_start` in `investbot/data/ingestion/prices.py`.

## Decided
- 2026-10-17 **Zero-volume flat daily bars** (O=H=L=C, volume 0, e.g. 2025-03-18 and 2026-01-15) are yfinance placeholders and are rejected by `quality.py`. They are not real sessions, and they would inject fake zero-range bars into ATR and the patterns. Backup before purge: `~/investbot_data/investbot.db.pre-dq-purge-20261017`.
- 2026-10-17 **Stale recent bars**: upsert a trailing window (5 daily sessions, 2 intraday days) and never store today's daily bar before 16:00 IST. This is task 000-ingest-refresh-window.
- 2026-10-17 **Agent permissions**: agents run with permission checks skipped, but they have no GitHub credentials and use a scratch DB. Only the runner pushes to main, and only after the full test suite passes.
