# Decisions

Agents add open questions under **Pending**. Ankit moves them to **Decided** with the answer.

## Pending
- 2026-10-17 **Meaning of "days" in `INTRADAY_REFRESH_DAYS`** (000-ingest-refresh-window). Chose: the last N *trading* days present in the downloaded data, so a Monday run still refreshes Friday. Alternative: N calendar days back from now (a Monday run with N=2 would refresh nothing from the previous week). The daily window works the same way (last N sessions in the download). To switch, change the `key`/window computation passed to `_refresh_start` in `investbot/data/ingestion/prices.py`.
- 2026-10-17 **Candle `CDL_NET_SCORE` weighting and scale** (`investbot/scoring/technical/candle_patterns.py`).
  Chosen: every pattern counts equally; a confirmed ±200 signal weighs 1.0, plain ±100 and TA-Lib's weaker ±80 (ENGULFING/HARAMI/HARAMICROSS) weigh 0.5; the sum is divided by `NET_SCORE_SCALE = 3.0` and clipped to [-1, 1]. On 20 NIFTY50 daily histories about 2.2% of bars clip.
  Alternatives: (a) a larger scale (e.g. 5.0) so busy bars don't saturate; (b) per-pattern reliability weights from a backtest (e.g. Bulkowski hit rates or our own forward returns); (c) treat ±80 as its own weight (e.g. 0.4) instead of folding it into plain.
  How to switch: change `CONFIRMED_WEIGHT`, `PLAIN_WEIGHT` or `NET_SCORE_SCALE` at the top of the module. (b) needs a weights dict and the backtest layer. Nothing consumes the score yet, so changing it is free until `scorer.py` lands.
- 2026-10-17 **Regime label inputs and thresholds** (`investbot/scoring/technical/regime.py`, 030-regime).
  Chosen: the label uses the 100-bar Hurst and Lyapunov (`REGIME_WINDOW = 100`, which responds within months, not a year). Trend vs range is `sigmoid(0.5·(H−0.5)/0.05 + 0.5·(ADX_14−25)/5)`. Chaos is `sigmoid(0.75·(VOL_RATIO−1.25)/0.2 + 0.25·(LYAP_100−0.30)/0.02)`, where VOL_RATIO = 20-bar vol / 250-bar vol. `REGIME` is the argmax of the three scores and `REGIME_CONFIDENCE` = top score − runner-up, which is continuous and 0 where the label flips. On the 50 NIFTY50 daily histories this gives about 47% range, 50% trend and 3% chaotic. The chaos midpoint (1.25, about the 85th percentile of VOL_RATIO) is set on the cautious side, so stress gets flagged more often, not less.
  Alternatives: (a) `regime_window = 250` for a slower, steadier label; (b) a higher chaos midpoint (1.4 gives about 1.7% chaotic); (c) confidence = the top score itself, rescaled from [1/3, 1], instead of the top-two gap, which never drops to 0; (d) fit the thresholds on forward returns once backtesting exists.
  How to switch: pass `regime_window=` to `compute_regime`, or edit the named constants at the top of the module (`HURST_*`, `ADX_*`, `VOL_RATIO_*`, `LYAP_*`, `*_WEIGHT`). (c) is a one-line change in `regime_scores`. Nothing consumes the regime yet.

## Decided
- 2026-10-17 **Zero-volume flat daily bars** (O=H=L=C, volume 0, e.g. 2025-03-18 and 2026-01-15) are yfinance placeholders and are rejected by `quality.py`. They are not real sessions, and they would inject fake zero-range bars into ATR and the patterns. Backup before purge: `~/investbot_data/investbot.db.pre-dq-purge-20261017`.
- 2026-10-17 **Stale recent bars**: upsert a trailing window (5 daily sessions, 2 intraday days) and never store today's daily bar before 16:00 IST. This is task 000-ingest-refresh-window.
- 2026-10-17 **Agent permissions**: agents run with permission checks skipped, but they have no GitHub credentials and use a scratch DB. Only the runner pushes to main, and only after the full test suite passes.
