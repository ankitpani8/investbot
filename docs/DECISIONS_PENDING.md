# Decisions

Agents add open questions under **Pending**. Ankit moves them to **Decided** with the answer.

## Pending
- 2026-10-17 **Meaning of "days" in `INTRADAY_REFRESH_DAYS`** (000-ingest-refresh-window). Chose: the last N *trading* days present in the downloaded data, so a Monday run still refreshes Friday. Alternative: N calendar days back from now (a Monday run with N=2 would refresh nothing from the previous week). The daily window works the same way (last N sessions in the download). To switch, change the `key`/window computation passed to `_refresh_start` in `investbot/data/ingestion/prices.py`.
- 2026-10-17 **Candle `CDL_NET_SCORE` weighting and scale** (`investbot/scoring/technical/candle_patterns.py`).
  Chosen: every pattern counts equally; a confirmed ±200 signal weighs 1.0, plain ±100 and TA-Lib's weaker ±80 (ENGULFING/HARAMI/HARAMICROSS) weigh 0.5; the sum is divided by `NET_SCORE_SCALE = 3.0` and clipped to [-1, 1]. On 20 NIFTY50 daily histories about 2.2% of bars clip.
  Alternatives: (a) a larger scale (e.g. 5.0) so busy bars don't saturate; (b) per-pattern reliability weights from a backtest (e.g. Bulkowski hit rates or our own forward returns); (c) treat ±80 as its own weight (e.g. 0.4) instead of folding it into plain.
  How to switch: change `CONFIRMED_WEIGHT`, `PLAIN_WEIGHT` or `NET_SCORE_SCALE` at the top of the module. (b) needs a weights dict and the backtest layer. Nothing consumes the score yet, so changing it is free until `scorer.py` lands.

## Decided
- 2026-10-17 **Zero-volume flat daily bars** (O=H=L=C, volume 0, e.g. 2025-03-18 and 2026-01-15) are yfinance placeholders and are rejected by `quality.py`. They are not real sessions, and they would inject fake zero-range bars into ATR and the patterns. Backup before purge: `~/investbot_data/investbot.db.pre-dq-purge-20261017`.
- 2026-10-17 **Stale recent bars**: upsert a trailing window (5 daily sessions, 2 intraday days) and never store today's daily bar before 16:00 IST. This is task 000-ingest-refresh-window.
- 2026-10-17 **Agent permissions**: agents run with permission checks skipped, but they have no GitHub credentials and use a scratch DB. Only the runner pushes to main, and only after the full test suite passes.
