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
- [ ] Add `dq` and `score` to the CLI list in `CLAUDE.md`.

## Phase 2: technical scoring
- [x] Indicator layer (TA-Lib core + extensions): 112 columns, smoke-tested
- [x] `candle_patterns.py`: all 61 TA-Lib CDL* patterns, raw + normalised, bull/bear counts, CDL_NET_SCORE (`agent/010-candle-patterns`)
- [x] `confluence.py` must de-duplicate overlapping candle patterns. DOJI, LONGLEGGEDDOJI, RICKSHAWMAN, HIGHWAVE and SPINNINGTOP fire together, so `CDL_BULL_COUNT`/`CDL_BEAR_COUNT` overcount one candle shape. Done in `agent/050-confluence`: the CANDLES signal excludes `INDECISION_PATTERNS` (those five plus DRAGONFLYDOJI and GRAVESTONEDOJI, which TA-Lib only ever emits as +100).
- [ ] TA-Lib signs CDL3LINESTRIKE by the three prior candles, so the classic bullish strike (three red, then a big green) comes out -100. Decide whether `scorer.py`/`confluence.py` flips it or leaves TA-Lib's convention. (`confluence.py` currently keeps TA-Lib's sign.)
- [ ] Candle patterns are price-only. Consider volume confirmation (e.g. relative volume on the pattern bar) in `confluence.py`.
- [ ] `support_resistance.py` (swing pivots, volume profile, Fibonacci)
- [x] `regime.py`: rolling DFA Hurst, Rosenstein Lyapunov, realised vol, REGIME + smooth REGIME_CONFIDENCE (`agent/030-regime`)
- [ ] Regime thresholds were hand-calibrated on in-sample NIFTY50 daily data (label shares ~47% range / 50% trend / 3% chaotic). Re-fit them, or at least sanity-check them, against forward returns once the backtest layer exists.
- [ ] `LYAP_CHAOS` is calibrated for daily bars at `regime_window=100`. Intraday (15m/1h) bars need their own calibration before `multi_timeframe.py` uses regime on them.
- [ ] HURST_100 still reads ~0.53 on a random walk (DFA small-sample bias). Consider a Monte-Carlo bias correction per window if the regime layer needs it.
- [ ] `candle_patterns.py`
- [x] `support_resistance.py` (swing pivots, volume profile, Fibonacci) (`agent/020-support-resistance`)
- [ ] `support_resistance.py` follow-ups: the volume profile is ~60% of its runtime (~0.3 s/asset daily). `scorer.py` now skips unchanged assets via its cache (a full daily universe run is ~30 s cold, ~2 s warm), so this only matters for backtests that score many dates; there, update the profile incrementally for the newest bar.
- [ ] `support_resistance.py` follow-up: S/R levels come only from price pivots. The master plan calls volume-weighted zones the *primary* S/R method. Decide in `confluence.py` how `SR_*` and `VP_*` combine (e.g. boost an S/R level that sits inside the value area or near the POC).
- [ ] `regime.py` (Hurst, Lyapunov, regime_confidence)
- [x] `multi_timeframe.py`: session-aware 15m→1h resampling (close-time labels), HTF alignment with previous-session daily values, MTF_TREND_ALIGN / MTF_MOMENTUM_ALIGN / MTF_REGIME_1D (`agent/040-multi-timeframe`)
- [ ] `multi_timeframe.py` follow-up: the DB has only ~60 sessions of 15m data (yfinance caps 15m history at 60 days), so 1h EMA_50 warmup eats ~7 sessions and there is too little intraday history to backtest MTF signals. Consider storing 15m bars continuously from now on and/or ingesting yfinance 1h bars (730-day cap) as a separate interval.
- [x] `multi_timeframe.py` follow-up: `scorer.py` should call `mtf_alignment` only on recent 15m history plus enough daily bars for regime warmup (~251), one asset at a time (8 GB RAM). Done in `agent/060-scorer`: ≤1600 15m bars + ≤750 daily bars per asset.
- [ ] `multi_timeframe.py` follow-up: holidays and special sessions (e.g. Muhurat trading, an evening session outside 09:15–15:30) make `resample_ohlcv` raise on out-of-session bars. Decide whether to drop them or support per-date session bounds.
- [x] `confluence.py`: 30 rule-based signals in 7 families, within-family mean, regime-blended family weights from `config/scoring_weights.py`, TECH_SCORE / TECH_SCORE_<FAMILY> / TECH_REASONS, explicit column→family contract (`agent/050-confluence`)
- [ ] `confluence.py` follow-up: signal rules and family weights are hand-set. Calibrate against forward returns once the backtest layer exists (the ML layer is meant to replace `FAMILY_WEIGHTS`). One unscientific data point: on RELIANCE.NS daily, TECH_SCORE correlates 0.82 with the past 10-day return and about -0.02 with the next 10 days.
- [ ] `confluence.py` follow-up: a signal's contribution is family weight / signals in the family, so each of the 2 S/R signals outweighs each of the 12 trend signals and S/R tends to top `TECH_REASONS`. Consider explaining at family level first (top family, then its top signal) once the dashboard shows reasons.
- [x] `confluence.py` follow-up (`agent/060-scorer`, `compute_daily_frame` / `compute_intraday_frame`): `scorer.py` must join the upstream outputs into one frame on the same index before calling `compute_confluence` (daily: indicators + candles + S/R + regime; intraday: plus MTF). The MTF family is NaN, and dropped from the weights, on daily-only frames.
- [ ] `confluence.py` follow-up: `SR_DIST_*` and `VP_POS` are two equal signals in the S/R family. The volume-weighted combination the master plan wants (boost a level inside the value area or near the POC) is still open; see the S/R follow-up above.
- [x] `scorer.py`: `score(ticker, timeframe, as_of)` / `score_universe`, `technical_scores` table, input-hash cache, `python main.py score [TICKER]` (`agent/060-scorer`)
- [ ] `scorer.py` follow-up: backtests calling `score(as_of=t)` for every date recompute a 750-bar window each time (~0.5 s/asset/date). Add a batch path that computes the frame once on the history and reads every bar (identical up to the fixed lookback window), and test it against `score(as_of=t)`.
- [ ] `scorer.py` follow-up: 24x7 assets (crypto). The as_of rule (a daily bar counts from 16:00 IST) and the NSE-session 15m resampling in `mtf_alignment` are equity-specific; `score(<crypto>, "15m")` comes back `status="error"`. Take the cutoff and session from `asset_classes` (`market_open/close`, `is_24x7`).
- [ ] `scorer.py` follow-up: the combined layer should write `scores.technical_score` (0-100 in the plan) from `technical_scores.tech_score` ([-1, 1]); see the pending decision on the scale.
- [ ] `scorer.py` follow-up: add a scheduler job that runs `score_universe` after the daily price job (and `"15m"` after the intraday job).
- [ ] `scorer.py` follow-up: staleness counts business days without an NSE holiday calendar, so a long holiday weekend can read 1 day staler than it is.
- [ ] The scratch DB still has TATAMOTORS.NS active with no price rows (the seed's `RETIRED_TICKERS` deactivation has not run there, and TMPV.NS is absent). `main.py score` lists it as `no_data`. Check the real DB after the next `setup`.

## Later phases
See `CLAUDE.md` → Build order and the master plan.
