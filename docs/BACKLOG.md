# Backlog

Status: `[ ]` open · `[~]` in progress (put the agent branch name next to it) · `[x]` done

## Housekeeping
- [x] Replace TATAMOTORS.NS with TMPV.NS; seed deactivates `RETIRED_TICKERS` without deleting history (`agent/005-housekeeping`)
- [x] Sync `NIFTY50_TICKERS` with the current index: duplicate WIPRO removed; WIPRO, BPCL, BRITANNIA, DIVISLAB, HEROMOTOCO, INDUSINDBK moved to `RETIRED_TICKERS`; BSE, ETERNAL, INDIGO, JIOFIN, MAXHEALTH added; list as of 2026-10-18 (`agent/140-nifty50-constituents-sync`)
- [ ] Historical NIFTY50 constituents for survivorship-free evaluation: store dated membership (symbol, added, removed) so the evaluation harness only scores a stock while it was in the index. `NIFTY50_TICKERS` is today's list only, so back-tests on it have survivorship bias. Source: NSE/niftyindices reconstitution circulars (index change history); needs a manual or one-off scrape. Needed by the evaluation harness (100-eval-harness).
- [ ] Run `python main.py ingest` for the 5 new constituents (BSE, ETERNAL, INDIGO, JIOFIN, MAXHEALTH): they are seeded but have no price rows yet. Needs a human to trigger it. Yahoo resolves all five symbols (checked 2026-10-18). ETERNAL (ex-Zomato) and JIOFIN (listed Aug 2023) have short or renamed history.
- [ ] Constituent sync is manual; add a periodic check of `ind_nifty50list.csv` against `NIFTY50_TICKERS` (NSE reconstitutes in March and September). Source used: https://archives.nseindia.com/content/indices/ind_nifty50list.csv (the niftyindices.com URL timed out).
- [x] Remove the duplicate `investbot/scoring/technical/smoke_indicators.py` (`agent/005-housekeeping`)
- [x] pytest promotes FutureWarning to error (`agent/005-housekeeping`)
- [ ] Update `docs/investbot_master_plan_v3.md` to v4: TA-Lib instead of pandas-ta, institutional flow signals, the cross-asset-class allocation layer, the 21-day LODR lag.

## Data quality
- [x] Reject incomplete OHLC rows at ingestion, add `python main.py dq [--purge]`, clean existing rows (`agent/fix-daily-nan`)
- [ ] Backfill the 158 purged bars with one `python main.py ingest` run (rows 2026-10-16 plus 108 zero-volume placeholders on 7 dates). Needs a human to trigger it, since agents must not hit yfinance. Placeholders yfinance still returns are rejected again, so those days stay missing.
- [x] Stale partial bars: upsert a trailing refresh window (5 daily sessions, 2 intraday trading days), never store today's daily bar before 16:00 IST, drop the forming 15m candle (`agent/000-ingest-refresh-window`)
- [ ] Daily ingestion still downloads `DAILY_PERIOD` (5y) on every run even though only the trailing window can change. Fetch a short period when the asset already has history.
- [ ] `ingest_all_prices` returns rows written (inserted + updated); `cmd_ingest` labels them "rows". Surface inserted/updated/rejected totals in the summary.
- [x] Run `dq` after the daily price job in the scheduler and surface failures: `run_post_close` audits after ingest, logs one ERROR per (table, ticker), never purges, and bad rows block scoring (`agent/210-scheduler-post-close-pipeline`)
- [x] `scripts/smoke_indicators.py [TICKER]`, default `DEFAULT_SMOKE_TICKER = "RELIANCE.NS"`; falls back to the largest asset with a printed warning, unknown explicit ticker exits 1 (`agent/210-scheduler-post-close-pipeline`)
- [x] `_extensions.py` `pct_change(fill_method=None)`: a NaN close is no longer masked in the return columns (`agent/005-housekeeping`)
- [x] `CUMPCTRET_1` / `CUMLOGRET_1` are NaN on any NaN-return bar and restart from the next valid bar (`agent/130-mtf-weights-and-cumret`)
- [x] Add `dq` and `score` to the CLI list in `CLAUDE.md` (already in the Layout line)

## Phase 2: technical scoring
- [x] Indicator layer (TA-Lib core + extensions): 112 columns, smoke-tested
- [x] `candle_patterns.py`: all 61 TA-Lib CDL* patterns, raw + normalised, bull/bear counts, CDL_NET_SCORE (`agent/010-candle-patterns`)
- [x] `confluence.py` must de-duplicate overlapping candle patterns. DOJI, LONGLEGGEDDOJI, RICKSHAWMAN, HIGHWAVE and SPINNINGTOP fire together, so `CDL_BULL_COUNT`/`CDL_BEAR_COUNT` overcount one candle shape. Done in `agent/050-confluence`: the CANDLES signal excludes `INDECISION_PATTERNS` (those five plus DRAGONFLYDOJI and GRAVESTONEDOJI, which TA-Lib only ever emits as +100).
- [ ] TA-Lib signs CDL3LINESTRIKE by the three prior candles, so the classic bullish strike (three red, then a big green) comes out -100. Decide whether `scorer.py`/`confluence.py` flips it or leaves TA-Lib's convention. (`confluence.py` currently keeps TA-Lib's sign.)
- [ ] Candle patterns are price-only. Consider volume confirmation (e.g. relative volume on the pattern bar) in `confluence.py`.
- [ ] `support_resistance.py` (swing pivots, volume profile, Fibonacci)
- [x] `regime.py`: rolling DFA Hurst, Rosenstein Lyapunov, realised vol, REGIME + smooth REGIME_CONFIDENCE (`agent/030-regime`)
- [x] Smooth REGIME_CONFIDENCE with a causal 5-bar EMA (`CONFIDENCE_EMA_SPAN`), restart after NaN gaps; raw gap kept as `REGIME_CONFIDENCE_RAW`; confluence/MTF use the smoothed value; `SCORER_VERSION` 2 (`agent/110-regime-confidence-smoothing`). On scratch-DB daily data (RELIANCE/INFY/HDFCBANK/ITC) crossings of confidence = 0.1 drop from ~170-186 to 16-28 per ~985 bars.
- [ ] Regime-confidence follow-up: test `CONFIDENCE_EMA_SPAN` (5) against forward returns and turnover in the evaluation harness, alongside the threshold re-fit below.
- [ ] Regime-confidence follow-up: `technical_scores` stores only the smoothed `regime_confidence`. Add a `regime_confidence_raw` column if the dashboard or the harness needs the unsmoothed gap per stored row.
- [ ] Regime thresholds were hand-calibrated on in-sample NIFTY50 daily data (label shares ~47% range / 50% trend / 3% chaotic). Re-fit them, or at least sanity-check them, against forward returns once the backtest layer exists.
- [ ] `LYAP_CHAOS` is calibrated for daily bars at `regime_window=100`. Intraday (15m/1h) bars need their own calibration before `multi_timeframe.py` uses regime on them.
- [ ] HURST_100 still reads ~0.53 on a random walk (DFA small-sample bias). Consider a Monte-Carlo bias correction per window if the regime layer needs it.
- [ ] `candle_patterns.py`
- [x] `support_resistance.py` (swing pivots, volume profile, Fibonacci) (`agent/020-support-resistance`)
- [ ] `support_resistance.py` follow-ups: the volume profile is ~60% of its runtime (~0.3 s/asset daily). `scorer.py` now skips unchanged assets via its cache (a full daily universe run is ~30 s cold, ~2 s warm), so this only matters for backtests that score many dates; there, update the profile incrementally for the newest bar.
- [ ] `support_resistance.py` follow-up: S/R levels come only from price pivots. The master plan calls volume-weighted zones the *primary* S/R method. Decide in `confluence.py` how `SR_*` and `VP_*` combine (e.g. boost an S/R level that sits inside the value area or near the POC).
- [x] `support_resistance.py`: `SR_STRENGTH_*` as a 0-1 share of total recency weight, ATR-based cluster tolerance (`SR_CLUSTER_ATR_MULT = 0.25`, `SR_CLUSTER_PCT` only as warmup fallback), unclipped `VP_DIST` in ATR units, `SCORER_VERSION` "2" (`agent/120-sr-normalisation`)
- [ ] `confluence.py` follow-up: `VP_DIST` and `SR_STRENGTH_*` are mapped to `None` (not scored): `VP_DIST` duplicates `VP_POS`, and neither strength nor distance-in-ATR has a rule yet. Once the harness exists, test whether a strength-weighted `SR_POSITION` or a breakout rule on `VP_DIST` adds forward-return signal.
- [ ] `support_resistance.py` follow-up: `SR_DIST_*_PCT` and the fixed `SR_RECENCY_HALFLIFE`/`SR_MAX_PIVOTS` are still price-% and bar counts; consider ATR-unit distances (`SR_DIST_*_ATR`) next to the pct ones for cross-asset comparability. TA-Lib ATR turns NaN for the rest of the series after one NaN OHLC bar, which silently switches clustering to the `SR_CLUSTER_PCT` fallback (same behaviour as `ATR_14` in the indicator layer).
- [ ] `regime.py` (Hurst, Lyapunov, regime_confidence)
- [x] `multi_timeframe.py`: session-aware 15m→1h resampling (close-time labels), HTF alignment with previous-session daily values, MTF_TREND_ALIGN / MTF_MOMENTUM_ALIGN / MTF_REGIME_1D (`agent/040-multi-timeframe`)
- [x] `multi_timeframe.py` `TIMEFRAME_WEIGHTS` = (1, 2, 3) for 15m/1h/1d; `SCORER_VERSION` 3 (`agent/130-mtf-weights-and-cumret`)
- [ ] `multi_timeframe.py` follow-up: with 1:2:3 weights a full-strength daily trend against full-strength 15m and 1h trends nets to exactly 0 in `MTF_TREND_ALIGN`. Evaluate in the harness whether the weights (or a daily tie-break) matter.
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
- [x] `scorer.py` follow-up (daily part): `score_universe("stock", "1d")` runs after a clean dq audit in the post-close pipeline (`agent/210-scheduler-post-close-pipeline`)
- [ ] `scorer.py` follow-up (15m part): score `"15m"` after the intraday job. Left out of 210 because the 15m history is too thin for MTF (see the MTF item).
- [ ] Scheduler: `audit_prices` scans every stored row (all history, both tables), so one old bad intraday row blocks daily scoring until someone runs `main.py dq --purge`. Consider auditing only the rows touched by the run, or a daily-only gate.
- [ ] Scheduler: `run_post_close` has no alert channel beyond ERROR logs; add a notification when scoring is skipped.
- [ ] `scorer.py` follow-up: staleness counts business days without an NSE holiday calendar, so a long holiday weekend can read 1 day staler than it is.
- [ ] The scratch DB still has TATAMOTORS.NS active with no price rows (the seed's `RETIRED_TICKERS` deactivation has not run there, and TMPV.NS is absent). `main.py score` lists it as `no_data`. Check the real DB after the next `setup`.

## Evaluation
- [x] Walk-forward forward-return harness: `investbot/evaluation/` (panel, next-open labels, baselines, Newey-West IC, quintiles, turnover, cost check, regime/year splits, BH q-values), `python main.py evaluate`, first report `docs/reports/eval_20261018.md` (`agent/100-eval-harness`)
- [ ] **Data: historical NIFTY50 constituents** (survivorship bias). The harness uses today's constituents over the whole history, which biases results upward. Store index membership with dates (NSE/niftyindices.com change notices), let `build_dataset` take the universe as of each date, and keep delisted/removed names' price history.
- [x] Harness follow-up: the cost check holds open[t+1] → close[t+h] and skips the overnight gap before the next entry, so short horizons read as large absolute losses (open-to-close drift). Add a continuously-held variant (open-to-open between rebalances) for the portfolio numbers. Done: `FWD_OO_RET_h` label, `metrics.held_portfolio` (compounded CAGR gross/net/universe, vol, max DD, IR), report table "Held portfolio (open-to-open, compounded)" (`agent/150-eval-held-portfolio-pooled-ic`)
- [x] Harness follow-up: the chaotic regime has only 41 OOS dates with ≥ 10 chaotic names, so its cross-sectional split is thin. Add a pooled (panel) IC or lower `MIN_NAMES` for that split. Done: `metrics.pooled_ic` (dates with ≥ `POOLED_MIN_NAMES` = 3 names, Newey-West t) on every `regime:<r>` row and in the report's "By regime" table (`agent/150-eval-held-portfolio-pooled-ic`)
- [ ] Harness follow-up: **dividends are missing from every label.** `close`/`open` are split-adjusted but not dividend-adjusted (ingestion uses `auto_adjust=False`), so forward returns, the held portfolio and the universe CAGR omit dividend yield (~1-1.5%/yr for NIFTY50) and high-yield names look worse than they are. The stored `adj_close` cannot be used as-is: its adjustment factors go stale outside the 5-day refresh window (only recent rows are re-downloaded when a new dividend shifts the whole history). Needs a corporate-actions table (ex-date, dividend per share) or a periodic full `adj_close` refresh, then a total-return label.
- [ ] Harness follow-up: `held_portfolio` leaves out rebalance dates with no top-bucket members (e.g. thin regime subsets), so those CAGRs cover invested periods only. If regime-conditional portfolios are ever acted on, model those periods as cash instead.
- [ ] Harness follow-up: the IC numbers in `eval_20261018_b.md` differ slightly from `eval_20261018.md` (e.g. TECH_SCORE h = 1 mean IC −0.004 → −0.005, SIG_SR_POSITION and TECH_SCORE_SR move most). The pre-task code (`f38ad97`) reproduces `_b` exactly on the scratch DB, so the difference is the data/code snapshot the first report ran on (likely before the S/R normalisation of task 120 landed), not task 150. Record `SCORER_VERSION` and a DB fingerprint (max bar date, row count) in the report Setup so runs are comparable.
- [ ] Harness follow-up: the MTF family (and intraday signals) cannot be evaluated: only ~60 sessions of 15m data exist. Revisit once 1h/15m history accumulates (see the multi-timeframe follow-up).
- [ ] Harness follow-up: the panel computes features on the full history, while `score()` uses a 750-bar window; EMA-seeded columns differ by ~1e-5. Once the scorer batch path exists, evaluate through it so the harness measures exactly what production scores.
- [ ] Harness follow-up: tax drag (STCG 20% on gains of short holds) is not in the cost check yet; add it via `trading/tax_ledger.py` when the backtest layer exists.
- [ ] Harness follow-up: `statsmodels` is not installed, so the BH test against it is skipped (the hand example runs). Fine as is; no dependency added.
- [ ] Decide on the first report's findings (see DECISIONS_PENDING): RVOL_CONFIRM is significant only inverted at h = 1; trend signals lean negative at h = 20 without surviving BH; nothing beats the baselines as designed.

## Phase 5d: tax and trading
- [x] `trading/tax_rules.py`, `transaction_costs.py`, `lot_tracker.py`: regime params, FY label, calendar-month holding period, cost calculator, FIFO lots with realisations and unrealised-by-lot (`agent/110-transaction-costs-lots`)
- [ ] DP charges: add the depository (CDSL/NSDL) charge plus the broker's DP fee per scrip per sell day to `estimate_costs` (not per order). Rate and who bills it needs checking against the broker's contract note.
- [ ] Align the master plan with the tax decisions: §2.8 4e says specific identification but FIFO is mandated (Circular 768); the `>= 365 days` SQL differs from `is_long_term`; the cost table has no DP charge.
- [x] Fiscal-year tax ledger (§2.8 4f): `trading/tax_ledger.py` (`compute_fiscal_year_tax`, `incremental_tax`, `CarryForward`, `FiscalYearTax.to_ledger_row`), dated regimes (`regime_for`, `IN_EQUITY_LISTED_PRE_2024_07_23`), seed now updates existing regime rows (`agent/120-tax-ledger`)
- [ ] Tax ledger follow-up: persist the ledger. Write `FiscalYearTax.to_ledger_row()` into `user_tax_ledger` (one row per user and FY, carry-forward chained from the previous FY's `carry_forward_out`) when the paper engine exists.
- [ ] Tax ledger follow-up (out of scope of 120): 31-Jan-2018 grandfathering (s.112A), basic-exemption-limit adjustment for resident individuals, s.87A rebate, slab-rate assets, and equity LTCG before 2018-04-01 (exempt under s.10(38); `regime_for` raises there). Surcharge is a caller-supplied flat rate (no income-based slabs, no 15% cap on capital-gains surcharge).
- [ ] Tax ledger follow-up: a loss with no eligible gain is carried as the type it was incurred in. Within a set-off pass STCL is applied before LTCL, which can leave less flexible LTCL carried when STCL could have been kept instead; a smarter order (LTCL against LTCG first) would keep the more flexible STCL. Same total tax this year, better carry-forward.
- [ ] Tax ledger follow-up: `compute_fiscal_year_tax` treats every `STCG`/`LTCG` realisation as equity (`regime_for("equity", ...)`) and every `FLAT` one as crypto. Realisations will need an asset-class tag when other asset classes (gold, REITs) get regimes.
- [ ] Transaction-cost regime dating: `estimate_costs` still takes the regime the caller passes (STT, stamp duty and exchange charges also changed over 2018-2026). `IN_EQUITY_LISTED_PRE_2024_07_23` copies the current charge rates; verify them before using it in backtest costs.
- [ ] Tax filter (§2.8 4a/4b, patience window using `days_to_long_term`, effective threshold using `round_trip_cost_pct` and tax drag). Under FIFO it reasons about how much and when to sell, not which lots.
- [ ] Persist lots (a `tax_lots` table or similar) and wire `sell_fifo` into the paper engine; check the existing schema first.
- [ ] Equity grandfathering of pre-1 Feb 2018 cost (Section 112A) is not modelled; irrelevant while paper trading starts now.
- [ ] Brokerage for crypto is the equity flat ₹20 placeholder; model exchange percentage fees when the crypto engine exists.

## Phase 5b: allocation
- [x] `trading/risk.py`: `log_returns`, `estimate_covariance` (point-in-time `as_of`, Ledoit-Wolf constant-correlation shrinkage, NaN-robust exclusions), `risk_parity_weights` (ERC, Spinu Newton), `inverse_volatility_weights`, `risk_contributions`, `portfolio_volatility`, `cap_weights` (position + sector caps) (`agent/130-risk-covariance-parity`)
- [ ] Risk follow-up: `assets.sector` is NULL for non-equity; the allocator must map those to a label (e.g. the asset class) before calling `cap_weights`, which raises on a missing sector.
- [ ] Risk follow-up: `estimate_covariance` uses the close-to-close returns of whatever frame it is given. When the eval/backtest layer calls it, build `close` from `adj_close` (dividends and splits) so a split day is not a fake -50% return; `log_returns` does not detect corporate actions.
- [ ] Risk follow-up: assets excluded for short history (new listings like TMPV.NS) get no risk-parity weight; decide in the allocator (task 140) how they enter the blend.

## Later phases
See `CLAUDE.md` → Build order and the master plan.

## Platform
- [ ] Move the venv to Python 3.12 and pandas 3. Run the full suite under pandas 3 first: the `.asi8` mixed-unit bug in multi_timeframe (fixed 2026-10-17) showed pandas 3 can silently break timestamp alignment. Then remove the `pandas<3` pin. (pandas-3)
