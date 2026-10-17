# Decisions

Agents add open questions under **Pending**. Ankit moves them to **Decided** with the answer.

## Pending
_(none yet)_

## Decided
- 2026-10-17 **Zero-volume flat daily bars** (O=H=L=C, volume 0, e.g. 2025-03-18 and 2026-01-15) are yfinance placeholders and are rejected by `quality.py`. They are not real sessions, and they would inject fake zero-range bars into ATR and the patterns. Backup before purge: `~/investbot_data/investbot.db.pre-dq-purge-20261017`.
- 2026-10-17 **Stale recent bars**: upsert a trailing window (5 daily sessions, 2 intraday days) and never store today's daily bar before 16:00 IST. This is task 000-ingest-refresh-window.
- 2026-10-17 **Agent permissions**: agents run with permission checks skipped, but they have no GitHub credentials and use a scratch DB. Only the runner pushes to main, and only after the full test suite passes.
