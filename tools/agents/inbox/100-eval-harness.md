---
id: 100-eval-harness
depends: []
model: opus
---
# Walk-forward forward-return evaluation harness

Ankit's top priority. TECH_SCORE correlates 0.82 with the *past* 10-day return and about -0.02 with the *next* 10 days, so today it describes the past instead of predicting the future. Build the measuring instrument that every later weight, threshold and polarity decision will be judged by. Read the "Evaluate before tuning" entry in docs/DECISIONS_PENDING.md.

## Requirements
- New package `investbot/evaluation/` (pure compute where possible; the DB is read only through a small loader).
- **Panel builder** `build_panel(assets=None, start=None, end=None) -> pd.DataFrame`: for each active stock, load its full daily history and compute the feature frame once with `scorer.compute_daily_frame` (it's truncation-invariant, so full-history features at bar t equal what was known at t). Process one asset at a time (8 GB RAM) and keep only the needed columns: TECH_SCORE, every TECH_SCORE_<FAMILY>, every confluence signal column, REGIME, REGIME_CONFIDENCE, plus baseline factors. Output a long panel indexed by (date, ticker).
- **Forward returns, no leakage.** The signal at bar t is known after t's close (16:00 IST), so the label is entry at the **next bar's open** and exit at close(t+h): `FWD_RET_h = close[t+h] / open[t+1] - 1` for h ∈ {1, 5, 20}. Also report the close-to-close variant for comparison. Labels live in a separate frame and are never joined into feature computation. Add a test that features at t are unchanged when all data after t is deleted, and that labels at t use only bars t+1..t+h.
- **Baselines**, so a signal has to beat something trivial: 20-day momentum, 12-1 month momentum (skip the last 21 days), 5-day reversal (negative 5-day return), and 20-day realised volatility (low-vol anomaly; use the negative sign).
- **Metrics per signal × horizon:**
  - daily cross-sectional Spearman IC; mean IC; IC standard deviation; ICIR; t-stat with **Newey-West** standard errors (lag = h−1, because overlapping windows autocorrelate); hit rate (share of days with IC > 0)
  - quintile portfolios (50 names, so quintiles, not deciles): mean forward return per quintile, Q5−Q1 spread with Newey-West t-stat, monotonicity (Spearman of quintile rank vs mean return)
  - top-quintile turnover (share of names changing per rebalance), which is the cost proxy
  - the same metrics split by REGIME (range/trend/chaotic) and by year
- **Walk-forward discipline.** Report metrics on rolling out-of-sample blocks: expanding train start, a 6-month test block, and a purge/embargo gap of max(h) bars between train and test. Nothing is fitted yet, but the split utilities are what the ML phase reuses. Unit-test that no test-block date falls inside train + embargo.
- **Multiple-testing honesty.** Many signals × 3 horizons are tested. Report the raw p-value plus Benjamini–Hochberg adjusted q-values across all tests, and label a signal "promising" only if q < 0.10 and the IC sign holds in at least 70% of test blocks.
- **Cost reality check.** For the Q5 long-only portfolio, show gross and net returns after Indian costs: round-trip ≈ 0.25% (brokerage, STT, exchange fees, GST, stamp, slippage). Take the exact values from `investbot/config/` or the master plan if they're there, otherwise define a documented constant.
- **Survivorship caveat.** The universe is today's NIFTY50 constituents, which biases results upward. Print this in every report and record it in the BACKLOG as a data task (historical constituents).
- **CLI:** `python main.py evaluate [--horizons 1,5,20] [--start YYYY-MM-DD]` writes a markdown report to `docs/reports/eval_<YYYYMMDD>.md` (tables: top signals by |ICIR|, baselines, family scores, TECH_SCORE, regime splits, BH-adjusted list) and the full metrics CSV to `tmp_output/` (git-ignored). It must run on the real DB in a few minutes on the laptop.
- Generate the first report from the scratch DB and commit it. In CHANGELOG, state plainly whether TECH_SCORE or any family/signal beats the baselines out of sample. A null result is a valid, valuable result; don't tune anything to make it look better.

## Acceptance criteria
- Tests: label alignment (synthetic series where FWD_RET is known exactly); feature/label separation; Newey-West against a hand-computed small example; BH against `statsmodels` if installed or a hand example; purge/embargo; quintile assignment with ties and NaNs.
- A synthetic "oracle" signal (equal to the true forward return) gives IC ≈ 1, and a random signal gives IC ≈ 0 with |t| < 2. Both tests must be in the suite.
- `python main.py evaluate` runs end to end on the scratch DB and the report is committed.
- No dropna() in compute functions, no lookahead, and the suite is green.
