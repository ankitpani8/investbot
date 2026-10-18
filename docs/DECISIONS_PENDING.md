# Decisions

Agents add open questions under **Pending**. Ankit moves them to **Decided** with the answer.

## Pending
_(none)_

## Decided
- 2026-10-17 **Zero-volume flat daily bars** (O=H=L=C, volume 0, e.g. 2025-03-18 and 2026-01-15) are yfinance placeholders and are rejected by `quality.py`. They are not real sessions, and they would inject fake zero-range bars into ATR and the patterns. Backup before purge: `~/investbot_data/investbot.db.pre-dq-purge-20261017`.
- 2026-10-17 **Stale recent bars**: upsert a trailing window (5 daily sessions, 2 intraday days) and never store today's daily bar before 16:00 IST. This is task 000-ingest-refresh-window.
- 2026-10-17 **Agent permissions**: agents run with permission checks skipped, but they have no GitHub credentials and use a scratch DB. Only the runner pushes to main, and only after the full test suite passes.

- 2026-10-18 **Evaluate before tuning (priority change).** TECH_SCORE correlates 0.82 with the past 10-day return and about -0.02 with the next 10 days (RELIANCE daily): it describes the past and doesn't yet predict. A walk-forward, forward-return evaluation harness is built next (task 100-eval-harness), before more fundamental, sentiment or allocation work. From now on, weights, thresholds and polarity changes need harness evidence, not intuition.
- 2026-10-18 **INTRADAY_REFRESH_DAYS** counts *trading* days present in the download (as implemented).
- 2026-10-18 **CDL_NET_SCORE** stays as implemented (confirmed 1.0, plain 0.5, scale 3.0). It gets replaced by harness- or ML-learned weights; no more hand-tuning.
- 2026-10-18 **REGIME_CONFIDENCE** stays "top score minus runner-up" (ambiguity means low conviction, so the allocator leans to risk parity), but it's smoothed with a causal 5-bar EMA to cut flip churn, which costs tax and fees. The raw value is kept as `REGIME_CONFIDENCE_RAW`. Window 100 and chaos midpoint 1.25 stay. (task 110)
- 2026-10-18 **Support/resistance**: `SR_STRENGTH_*` becomes a 0–1 share of total recency weight (comparable across assets); cluster tolerance becomes ATR-based (`SR_CLUSTER_ATR_MULT = 0.25` × ATR_14) instead of a fixed 0.5%; add an unclipped `VP_DIST` and keep `VP_POS` clipped; pivots stay 3/3. (task 120)
- 2026-10-18 **Multi-timeframe**: `TIMEFRAME_WEIGHTS = (1, 2, 3)` for 15m/1h/1d, so the higher timeframe dominates (less 15m noise, fewer trades, less STCG). RSI bands stay 55/45. The provisional same-day daily bar is rejected (not NSE's official close; lookahead risk). (task 130)
- 2026-10-18 **Confluence polarity** stays as implemented (follow momentum, fade to 0 at extremes, ×0.5 in chaotic regimes, NaN until every family is out of warmup). Alternatives (a)/(b), full mean reversion vs regime-dependent polarity, are to be tested in the harness, not chosen by hand.
- 2026-10-18 **Scorer storage** stays as implemented: separate `technical_scores` table on [-1, 1], 750/1600-bar lookback, daily bars count from 16:00 IST, stale after 3 business days.
- 2026-10-18 **CDL3LINESTRIKE** keeps TA-Lib's sign; learned weights make the sign irrelevant.
- 2026-10-18 **Cumulative returns across a missing close** (`CUMLOGRET_1`, `CUMPCTRET_1`) go NaN on the gap bar and restart from 0 on the next valid bar, instead of silently skipping the period. (task 130)
