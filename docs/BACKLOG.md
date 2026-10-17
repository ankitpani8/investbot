# Backlog

Status: `[ ]` open · `[~]` in progress (put the agent branch name next to it) · `[x]` done

## Housekeeping
- [x] Replace TATAMOTORS.NS with TMPV.NS; seed deactivates `RETIRED_TICKERS` without deleting history (`agent/005-housekeeping`)
- [ ] Sync `NIFTY50_TICKERS` with the current index. NSE's 2026-09-15 release excludes WIPRO and includes DRREDDY (already listed). The list also has WIPRO twice. Check any other drift against the niftyindices.com constituent CSV, and move removed names to `RETIRED_TICKERS`.
- [x] Remove the duplicate `investbot/scoring/technical/smoke_indicators.py` (`agent/005-housekeeping`)
- [x] pytest promotes FutureWarning to error (`agent/005-housekeeping`)
- [ ] Update `docs/investbot_master_plan_v3.md` to v4: TA-Lib instead of pandas-ta, institutional flow signals, the cross-asset-class allocation layer, the 21-day LODR lag.

## Data quality
- [x] Reject incomplete OHLC rows at ingestion, add `python main.py dq [--purge]`, clean existing rows (`agent/fix-daily-nan`)
- [ ] Backfill the 158 purged bars with one `python main.py ingest` run (rows 2026-10-16 plus 108 zero-volume placeholders on 7 dates). Needs a human to trigger it, since agents must not hit yfinance. Placeholders yfinance still returns are rejected again, so those days stay missing.
- [x] Stale partial bars: upsert a trailing refresh window (5 daily sessions, 2 intraday trading days), never store today's daily bar before 16:00 IST, drop the forming 15m candle (`agent/000-ingest-refresh-window`)
- [ ] Daily ingestion still downloads `DAILY_PERIOD` (5y) on every run even though only the trailing window can change. Fetch a short period when the asset already has history.
- [ ] `ingest_all_prices` returns rows written (inserted + updated); `cmd_ingest` labels them "rows". Surface inserted/updated/rejected totals in the summary.
- [ ] Run `dq` after the daily price job in the scheduler and surface failures (it exits 1 on bad rows).
- [ ] `scripts/smoke_indicators.py` picks the asset with the most rows, so the asset under test changes when rows are purged (RELIANCE → ITC). Take an optional ticker argument and default to a fixed one.
- [x] `_extensions.py` `pct_change(fill_method=None)`: a NaN close is no longer masked in the return columns (`agent/005-housekeeping`)
- [ ] `CUMPCTRET_1` and `CUMLOGRET_1` use cumprod/cumsum, which skip NaN, so a single missing close silently drops one period from the cumulative return. Decide whether the cumulative columns should go NaN or carry on.
- [ ] Add `dq` to the CLI list in `CLAUDE.md`.

## Phase 2: technical scoring
- [x] Indicator layer (TA-Lib core + extensions): 112 columns, smoke-tested
- [x] `candle_patterns.py`: all 61 TA-Lib CDL* patterns, raw + normalised, bull/bear counts, CDL_NET_SCORE (`agent/010-candle-patterns`)
- [ ] `confluence.py` must de-duplicate overlapping candle patterns. DOJI, LONGLEGGEDDOJI, RICKSHAWMAN, HIGHWAVE and SPINNINGTOP fire together, so `CDL_BULL_COUNT`/`CDL_BEAR_COUNT` overcount one candle shape.
- [ ] TA-Lib signs CDL3LINESTRIKE by the three prior candles, so the classic bullish strike (three red, then a big green) comes out -100. Decide whether `scorer.py`/`confluence.py` flips it or leaves TA-Lib's convention.
- [ ] Candle patterns are price-only. Consider volume confirmation (e.g. relative volume on the pattern bar) in `confluence.py`.
- [ ] `support_resistance.py` (swing pivots, volume profile, Fibonacci)
- [x] `regime.py`: rolling DFA Hurst, Rosenstein Lyapunov, realised vol, REGIME + smooth REGIME_CONFIDENCE (`agent/030-regime`)
- [ ] Regime thresholds were hand-calibrated on in-sample NIFTY50 daily data (label shares ~47% range / 50% trend / 3% chaotic). Re-fit them, or at least sanity-check them, against forward returns once the backtest layer exists.
- [ ] `LYAP_CHAOS` is calibrated for daily bars at `regime_window=100`. Intraday (15m/1h) bars need their own calibration before `multi_timeframe.py` uses regime on them.
- [ ] HURST_100 still reads ~0.53 on a random walk (DFA small-sample bias). Consider a Monte-Carlo bias correction per window if the regime layer needs it.
- [ ] `candle_patterns.py`
- [x] `support_resistance.py` (swing pivots, volume profile, Fibonacci) (`agent/020-support-resistance`)
- [ ] `support_resistance.py` follow-ups: the volume profile is ~60% of its runtime (~0.3 s/asset daily). If `scorer.py` recomputes every asset each run, cache it or update the profile incrementally for the newest bar only.
- [ ] `support_resistance.py` follow-up: S/R levels come only from price pivots. The master plan calls volume-weighted zones the *primary* S/R method. Decide in `confluence.py` how `SR_*` and `VP_*` combine (e.g. boost an S/R level that sits inside the value area or near the POC).
- [ ] `regime.py` (Hurst, Lyapunov, regime_confidence)
- [ ] `multi_timeframe.py`
- [ ] `confluence.py`
- [ ] `scorer.py`: `score(ticker)`

## Later phases
See `CLAUDE.md` → Build order and the master plan.
