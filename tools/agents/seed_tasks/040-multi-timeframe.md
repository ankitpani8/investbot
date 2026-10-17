---
id: 040-multi-timeframe
depends: [030-regime]
---
# multi_timeframe.py: 15m / 1h / daily alignment without lookahead

## Requirements
- New module `investbot/scoring/technical/multi_timeframe.py`, pure compute.
- `resample_ohlcv(df_15m, rule)` builds 1h bars from 15m, NSE-session aware (09:15–15:30 IST, with the last bar possibly partial). Bars are labelled by **close time**.
- `align_higher_timeframe(base_df, htf_df, prefix)` attaches higher-timeframe features to each base bar, using only HTF bars that had **closed** at the base bar's timestamp. For daily features on intraday bars, that means the **previous** day's values. This is the most common MTF lookahead bug; test it explicitly.
- `mtf_alignment(...)` returns `MTF_TREND_ALIGN` (-1..+1: agreement of trend direction across 15m/1h/1d using EMA_20 vs EMA_50 and SUPERTd), `MTF_MOMENTUM_ALIGN` (RSI_14 regime agreement) and `MTF_REGIME_1D` (from regime.py).
- Prefix convention: `H1_` and `D1_`.

## Acceptance criteria
- A test where daily values are known proves no intraday bar on day D sees day D's daily close.
- Truncation invariance on the base timeframe, the resample test (counts and OHLC aggregation correct), and a green suite.
