# Backlog

Status: `[ ]` open · `[~]` in progress (put the agent branch name next to it) · `[x]` done

## Housekeeping
- [ ] Replace TATAMOTORS.NS (delisted on Yahoo after the 2025 demerger) in `investbot/config/tickers.py`. Check the current NIFTY50 constituents and update the list.
- [ ] Remove the duplicate `investbot/scoring/technical/smoke_indicators.py` (the canonical copy is in `scripts/`).
- [ ] Update `docs/investbot_master_plan_v3.md` to v4: TA-Lib instead of pandas-ta, institutional flow signals, the cross-asset-class allocation layer, the 21-day LODR lag.

## Data quality
- [x] Reject incomplete OHLC rows at ingestion, add `python main.py dq [--purge]`, clean existing rows (`agent/fix-daily-nan`)
- [ ] Backfill the 158 purged bars with one `python main.py ingest` run (rows 2026-10-16 plus 108 zero-volume placeholders on 7 dates). Needs a human to trigger it, since agents must not hit yfinance. Placeholders yfinance still returns are rejected again, so those days stay missing.
- [ ] Stale partial bars: ingestion skips dates that already exist, so a bar stored with a valid close mid-session is never refreshed. Options: re-fetch and upsert the last N days on every run, or skip today's bar until the session is closed. Not decided.
- [ ] Run `dq` after the daily price job in the scheduler and surface failures (it exits 1 on bad rows).
- [ ] `scripts/smoke_indicators.py` picks the asset with the most rows, so the asset under test changes when rows are purged (RELIANCE → ITC). Take an optional ticker argument and default to a fixed one.
- [ ] `_extensions.py` `close.pct_change()` forward-fills NaN by default (FutureWarning). Pass `fill_method=None` so a NaN close is not masked in the return columns.
- [ ] Add `dq` to the CLI list in `CLAUDE.md`.

## Phase 2: technical scoring
- [x] Indicator layer (TA-Lib core + extensions): 112 columns, smoke-tested
- [ ] `candle_patterns.py`
- [ ] `support_resistance.py` (swing pivots, volume profile, Fibonacci)
- [ ] `regime.py` (Hurst, Lyapunov, regime_confidence)
- [ ] `multi_timeframe.py`
- [ ] `confluence.py`
- [ ] `scorer.py`: `score(ticker)`

## Later phases
See `CLAUDE.md` → Build order and the master plan.
