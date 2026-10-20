# InvestBot — Master Plan v4.0

## Changes in v4 (2026-10-19)

The file keeps its v3 name because CLAUDE.md links to it. Where CLAUDE.md and this plan still differ, CLAUDE.md wins. Only decided items were folded in (CLAUDE.md, and the **Decided** section of `docs/DECISIONS_PENDING.md`); nothing under **Pending** was resolved.

1. **TA-Lib instead of pandas-ta.** Touched: §2.6, §3 (architecture diagram), §5.1 (library line, why pandas-ta was dropped, indicator list heading and family lists, custom-indicator table heading, feature total), §7, §8, Phase 2 task 1. Indicators that the indicator layer does not compute are listed as "Not implemented"; Ichimoku's Chikou Span is excluded (lookahead). Source: CLAUDE.md "Source of truth", first bullet.
2. **Institutional flow rules.** New §5.5. FII and DII net flow combined, 3-day z-score > 1.5 activation, bulk/block deals confirmation-only, shareholding change as a slow independent signal, 21-day LODR shareholding lag; everything else marked "to be designed". Source: CLAUDE.md "Source of truth", second bullet.
3. **Top-level cross-asset-class allocation layer.** Touched: §3 (architecture diagram) and §6 (next to "Cross-Asset Rebalancing"). Source: CLAUDE.md "Source of truth", third bullet.
4. **Evaluation before tuning.** New §9.4 on the walk-forward forward-return harness and the regime study, with links to `docs/reports/`. Weight, threshold and polarity changes need harness evidence. Source: CLAUDE.md "Current priorities" 1 and 4; Decided 2026-10-18 "Evaluate before tuning".
5. **Point-in-time fundamentals.** Touched: §5.2 and Phase 3 (§10). A filing's numbers count from its publication date, or conservatively from period end + 45 days (quarterly) / + 60 days (annual), per SEBI LODR Reg. 33. Kept separate from the 21-day shareholding lag in §5.5. Source: CLAUDE.md "Current priorities" 3.
6. **§7 project structure** replaced with the layout that exists (CLAUDE.md "Layout" plus `investbot/trading/`, `investbot/evaluation/`, `scripts/`, `tools/`); modules that do not exist yet are marked "(planned)". Source: CLAUDE.md "Layout".
7. **Decided entries that change plan text** (one sentence each):
   - Confluence polarity (Decided 2026-10-18 "Confluence polarity"): §5.1 confluence scoring sentence.
   - Scorer storage on [-1, 1] in `technical_scores` (Decided 2026-10-18 "Scorer storage"): same §5.1 sentence.
   - MTF weights 1/2/3 for 15m/1h/1d (Decided 2026-10-18 "Multi-timeframe"): §5.1 multi-timeframe sentence and Phase 2 task 6.
   - REGIME_CONFIDENCE as a smoothed top-two gap (Decided 2026-10-18 "REGIME_CONFIDENCE"): §2.3 output bullet.
   - S/R strength normalisation (Decided 2026-10-18 "Support/resistance"): §5.1 custom-indicator table, S/R row.
   - Retired-from-index means deactivated (Decided 2026-10-18 "Retired-from-index means deactivated"): Phase 1 task 3.
8. **Facts fixed to match the code (task 360, 2026-10-20).** §5.1 Hurst row (rolling DFA-1, windows 100 and 250, was R/S), Phase 2 task 7 (Hurst windows 100/250, label window 100, Rosenstein Lyapunov, was 50/100/200), Phase 7 task 9 (285 feature columns, was "145+"), and the regime-study open question now points to the 2026-10-20 H6 and task-350 results. The 0-100 text is unchanged.

**Open questions affecting this plan.** These are still under **Pending** in `docs/DECISIONS_PENDING.md`; the plan text they touch is unchanged and not resolved here.

- 2026-10-18 *Holding-period rule*: the `>= 365` days SQL in §4.10 and the 365-day wording in §2.8 and §6.
- 2026-10-18 *FIFO only vs. specific identification*: §2.8 4e, §6 step 5f, Phase 5d task 13.
- 2026-10-18 *Transfer expenses on crypto (FLAT term)*: the §2.8 cost table.
- 2026-10-18 *Tax ledger: rate-regime split inside one FY*, *Tax ledger: exemption amount*, *Tax ledger: crypto before 2022-04-01*: §2.8 4f and §4.10 (BACKLOG: "Align the master plan with the tax decisions").
- 2026-10-18 *Risk parity = equal risk contribution, not inverse volatility*: §2.7, §6 step 3, Phase 5b task 4.
- 2026-10-18 *Covariance window and history requirement*: §2.7 Layer 1, §6 step 3.
- 2026-10-18 *Position and sector caps*: §4.7 `users` defaults, §6 step 3.
- 2026-10-18 *Evaluation cost per round trip*: the §2.8 "~0.25-0.35%" round-trip cost and §9.4.
- 2026-10-18 *Quintiles of discrete signals*, *"Promising" rule details*, *Harness finding for you: SIG_RVOL_CONFIRM polarity*, *Point-in-time universe: judgment calls*: §9.4 and the polarity wording in §5.1.
- 2026-10-18 *Corporate actions: an ex-dated table instead of a periodic `adj_close` re-download*: §9.4 (total-return labels).
- 2026-10-18 *Index membership: how far back, and temporary members*: §9.4 (point-in-time universe).
- 2026-10-18 *Regime study: results* (with its 2026-10-19 re-run): §2.3 and §9.4. Options include re-fitting regime thresholds, changing the smoothing span, or a fixed Black-Litterman / risk-parity blend, which would affect §2.7 Layer 2. Latest evidence: the 2026-10-20 H6 section (real minus shuffled null, task 330, `docs/reports/regime_study_20261020.md`: 0 of 6 supported) and the task-350 robust-inference section (`docs/reports/regime_study_20261020_b.md`: no verdict change in the index universe).
- 2026-10-19 *Polarity variants: harness results*: §5.1 polarity wording and §9.4.
- 2026-10-19 *Total-return labels: judgment calls*: §9.4.
- Not decided anywhere: how the combined layer maps the [-1, 1] technical score onto the 0-100 scale used in §4.5, §5.4 and §6 (see the §5.1 note), and the design of the cross-asset-class layer (§6) and of institutional flow (§5.5).

---

## 1. Vision

A personal investment assistant that answers one continuous question: **"Right now, given my total capital (held positions + wallet cash), what is the optimal allocation?"**

This is a pigeonhole problem running in near-real-time. N rupees, M possible slots (assets + cash). The system continuously compares actual allocation vs ideal allocation and closes the gap. A "buy" is rebalancing toward an asset. A "sell" is rebalancing away. New cash deposited simply increases total capital, triggering the same rebalancing logic. There are no separate "entry" and "exit" decisions — just one unified control loop.

**The system removes emotion from trading.** It scores, decides, trades, and notifies. The user's only decisions are: how much capital to deploy, what risk profile to use, and whether to keep cash idle or reinvest after a sell.

**Tax-awareness is load-bearing, not optional.** For taxable retail investors, the tax cost of a trade can exceed the alpha the trade captures. The system treats tax as a first-class constraint on every rebalancing decision — adjusting thresholds, deferring marginal trades near holding period cliffs, proactively harvesting losses, and optimizing lot selection. The algorithm doesn't just decide *what* to hold and *when* to rebalance — it decides *whether the after-tax economics justify acting at all*.

---

## 2. Theoretical Foundations

### 2.1 Factor-Based Investing (Fama-French + Carhart)

The scoring system provides systematic exposure to empirically proven return factors:

- **Quality + Value** → Fundamental scoring (Buffett checklist): gross margins, SGA efficiency, low debt, earnings consistency, balance sheet strength, free cash flow quality
- **Momentum** → Technical scoring: trend-following via indicator confluence
- **Sentiment/Behavioral** → Sentiment scoring: captures crowd positioning and herding

### 2.2 Control Theory — PID Portfolio Controller

Portfolio management is framed as a control problem, directly analogous to line-following robots or inverted pendulum balancing.

**The "no reference signal" problem is solved by the scoring system.** At any moment, the combined score defines what the ideal portfolio should look like. That's the reference trajectory. The error signal is the gap between actual and ideal allocation. The controller's job is to close that gap, subject to constraints (transaction costs, position limits, risk, tax).

**PID mapping:**

- **P (Proportional):** How far is my current portfolio from target right now? Large error → large trade. The gain Kp controls how aggressively the system responds to current misalignment.
- **I (Integral):** Has the portfolio been persistently drifting from target? Accumulated error over time triggers rebalancing even when the instantaneous error is small. Prevents over-trading on noise while catching real drift.
- **D (Derivative):** Is the deviation accelerating? This is the early warning system. If a stock's score is dropping fast (not just low, but falling rapidly), the D term triggers action before the P term registers a large error. This is the trailing stop-loss derived from control theory rather than a fixed percentage.

**Risk profile as controller gains:**

| Risk Profile | Kp (Proportional) | Ki (Integral) | Kd (Derivative) | Behavior |
|---|---|---|---|---|
| Aggressive | High | Low | Moderate | Big trades, quick reaction, moderate caution |
| Moderate | Medium | Medium | Medium | Balanced response across all terms |
| Conservative | Low | High | High | Small trades, waits for persistent drift, exits early at first trouble |

An aggressive investor is like having a higher proportional constant — the swings on either side are bigger, but they follow the path. A conservative investor has high D gain — they're extremely sensitive to acceleration, exiting at the first sign of momentum loss.

**Adaptive Gain Scheduling:** The PID gains are not fixed. They are multiplied by a regime confidence factor derived from market dynamics (see §2.3). When the market is orderly, gains operate normally. When the market is chaotic, gains are dampened — the system becomes cautious automatically.

### 2.3 Regime Detection (Chaos Theory / Econophysics)

Markets alternate between ordered (trending) and chaotic (random) regimes. This insight comes from the logistic map: for certain parameter values, deterministic systems produce output indistinguishable from randomness. Market prices may be generated by relatively simple nonlinear feedback mechanisms (fear/greed, supply/demand, leverage/deleveraging), but the resulting series appears random because the system is in a chaotic regime.

**Practical tools:**

- **Hurst Exponent (H):** Computed on a rolling window. H > 0.5 = trending regime (trust momentum signals, increase position sizes). H ≈ 0.5 = random walk (reduce exposure, tighten stops). H < 0.5 = mean-reverting (consider contrarian signals).
- **Rolling Lyapunov Exponent:** Measures sensitivity to initial conditions. Spikes indicate high chaos — reduce position sizes and tighten stops.
- **LSTM/GRU Regime Transition Predictor:** Trained on sequences of market microstructure features to predict regime *transitions* before they happen. Regime changes have precursors: volatility clustering changes shape, sector correlations break down, volume patterns shift. Input features per timestep: rolling volatility (multiple windows), inter-stock correlation matrix changes, volume profile shifts, put-call ratio changes, VIX/India VIX levels, sector rotation patterns, breadth indicators (advance/decline ratio). Target: regime label at t+N (trending / random / mean-reverting), derived from the Hurst exponent computed ex-post. Architecture: LSTM/GRU with sequence length of 20-60 bars → dense layers → softmax over 3 regime classes. Requires 6+ months of accumulated intraday archive for training — system accumulates this archive from day one.
- **Output:** A regime_confidence multiplier (0 to 1) applied to the technical component of the combined score, and used as adaptive gain scheduling for the PID controller. regime_confidence is the top regime score minus the runner-up (ambiguity means low conviction, so the allocator leans to risk parity), smoothed with a causal 5-bar EMA to cut flip churn (which costs tax and fees); the raw gap is kept as `REGIME_CONFIDENCE_RAW`.

**Key insight:** These tools don't tell you what to buy. They tell you how much to trust your other signals right now. This is exactly adaptive gain scheduling in control theory — adjusting controller gains based on plant dynamics.

### 2.4 Momentum — Three Levels, One Force

| Level | What It Measures | Timeframe | System Component |
|---|---|---|---|
| Fama-French MOM factor | Which stocks have been winning (3-12 months) | Medium-term | Informs universe quality |
| Technical indicators (MACD, RSI, etc.) | Current trend strength and acceleration | Short-term (intraday to weekly) | Entry/exit timing via confluence |
| PID Derivative term | Rate of change of portfolio drift | Continuous | Rebalancing urgency — momentum of the momentum signal |

All three measure serial correlation in returns at different resolutions. The Fama-French factor tells you which assets to be in. Technical indicators tell you whether now is a good time. The PID derivative tells you how urgently to act. They reinforce each other.

### 2.5 Fibonacci & Support/Resistance

- **Fibonacci retracement levels** (23.6%, 38.2%, 50%, 61.8%) are included as one input to support/resistance detection — not because of mathematical truth about markets, but because enough traders use them that order flow clusters at these levels. Self-fulfilling prophecy with measurable predictive power about where orders sit.
- **Volume-weighted support/resistance zones** are the primary S/R method. These identify price levels where historically significant trading volume occurred, indicating genuine supply/demand imbalance.

### 2.6 Machine Learning Pipeline

**Technical Scoring — PCA + XGBoost/LightGBM:**

The confluence scoring problem is fundamentally a feature engineering and selection problem. The TA-Lib indicator layer plus our `_extensions.py` provides ~112 indicator columns, but many are correlated (RSI and Stochastic measure similar things, MACD and EMA crossovers overlap). The ML pipeline uses these TA-Lib features (see §5.1). Rather than hand-picking indicators and hand-tuning weights:

1. Compute all indicator columns as raw features for each asset at each timestamp
2. Apply PCA (Principal Component Analysis) to reduce dimensionality — this finds the orthogonal combinations of indicators that explain the most variance in the data, eliminating redundancy
3. Train XGBoost/LightGBM on the PCA components to predict: "did the stock go up X% in the next N days?"
4. The model learns which factor combinations are genuinely predictive vs noise

This replaces the hand-tuned confluence weights with empirically optimized, nonlinear scoring. PCA handles multicollinearity, XGBoost captures interaction effects (e.g., "RSI < 30 is only bullish when ADX > 25 and volume is rising").

**Training data:** Generated from Phase 7 backtesting — every timestamp's indicator values paired with the actual forward return. As the system runs live, it accumulates its own intraday archive, which grows the training set over time.

**Sentiment Analysis — FinBERT (Transformer-Based from Day 1):**

FinBERT (ProsusAI/finbert, free via HuggingFace) is the sentiment engine. Transformer-based NLP is used for all language tasks in the system — no bag-of-words or rule-based approaches. FinBERT understands financial context: it correctly identifies "going to the moon 🚀" as bullish and "management is cooking the books" as bearish. Later, the model can be fine-tuned on Indian market social media text (Indian English/Hindi-English slang) if the pre-trained model underperforms on local content.

### 2.7 Portfolio Rebalancing — Four-Layer Approach

Portfolio rebalancing is separated into four distinct sub-problems, each solved by the right tool:

**Layer 1 — Target Weight Computation: Black-Litterman Model**

The Black-Litterman model (1992) starts with a market equilibrium portfolio (e.g., market-cap-weighted NIFTY50) and adjusts it based on "views" — quantified beliefs about which assets will outperform. Our combined scoring system IS the views input. A stock with a combined score of 85 translates to a view that it will outperform by some margin. Black-Litterman then computes optimal portfolio weights that balance these views against the market equilibrium, properly accounting for the covariance structure between stocks. This prevents a naive problem: if RELIANCE and JIOFINANCE both score 85 but are 90% correlated, simple score-proportional weighting would overweight both, concentrating risk. Black-Litterman recognizes the correlation and distributes weight more efficiently.

**Layer 2 — Regime-Dependent Blending: Risk Parity Fallback**

When regime_confidence is high (trending, orderly market), the system uses Black-Litterman weights — trusting the scoring system's views. When regime_confidence is low (chaotic, random market), the system blends toward risk parity weights — equalizing risk contribution rather than trusting potentially unreliable signals. The blending is continuous, not binary:

```
effective_target = regime_confidence × BL_weights + (1 - regime_confidence) × risk_parity_weights
```

This means: in a strongly trending market, the portfolio is almost entirely score-driven. In chaos, it falls back to a defensive, diversified posture. In between, it's a blend.

**Layer 3 — Execution Speed: PID Controller**

Given the target weights (from Layers 1+2), the PID controller determines how aggressively to close the gap between actual and target allocation. It does NOT decide what to hold — only how fast to get there. The PID gains are parameterized by the user's per-asset-class risk appetite, and further modulated by regime_confidence (adaptive gain scheduling).

**Layer 4 — Tax Filter & Optimizer (see §2.8)**

Before any trade is executed, the tax filter evaluates whether the after-tax economics justify acting. This layer can defer, modify, or enhance PID-generated signals based on holding period, unrealized gain/loss, available exemptions, and fiscal year position.

### 2.8 Tax-Aware Portfolio Optimization

**Why tax is load-bearing for this system:**

This system serves taxable retail investors. Unlike hedge funds (which use structural engineering to manage tax) or HFT desks (where per-trade alpha dwarfs tax), a medium-frequency factor-based strategy targeting 15-20% gross annual returns can lose 4%+ of portfolio value annually to tax drag if rebalancing is frequent and all gains are short-term. Intelligent tax management delivers 1-2% annual "tax alpha" — often larger than stock selection alpha itself.

The single highest-value tax optimization is: **don't trade unnecessarily.** A buy-and-hold NIFTY50 index has massive tax efficiency simply because it barely trades. The PID controller's tendency to rebalance is the primary thing the tax layer tempers. Everything else is second-order polish.

**Tax does not override investment decisions.** If a position genuinely should be sold for risk reasons, sell it. But if it's a marginal rebalance that's only modestly favorable pre-tax, and selling triggers a large tax bill, skip it. The bar for action is higher, not the direction.

**Layer 4 Components:**

**4a. Tax-Adjusted Rebalance Threshold:**

The minimum control signal magnitude needed to trigger a trade is dynamically raised based on the tax cost of that specific trade:

```
effective_threshold = base_threshold + tax_drag_estimate(position) + transaction_cost_estimate(trade)
```

Where `tax_drag_estimate` considers: the applicable tax rate (STCG vs LTCG vs flat crypto rate based on holding period), the unrealized gain/loss amount, remaining annual LTCG exemption, and current fiscal year tax position. A sell of a position with large short-term gains needs a much stronger signal than a sell of a position at a loss.

**4b. Tax Patience Window:**

For positions within N days of the LTCG threshold (configurable, default 45 days) with unrealized gains:

```
tax_savings = unrealized_gain × (stcg_rate - ltcg_rate)
expected_drawdown_risk = f(regime_confidence, asset_volatility, days_to_ltcg_threshold)
hold_if: tax_savings > expected_drawdown_risk
```

If the signal is catastrophic (hard exit triggers — trailing SL broken, support collapsed with volume, combined score below 20), tax patience is overridden. Don't hold through a crash to save 7.5%.

The PID's integral term accumulates deferred error, so if the signal persists past the patience window, it eventually overcomes the tax threshold.

**4c. Tax-Loss Harvesting Engine:**

Runs as a proactive overlay, not just reactive:
- Continuously scan portfolio for positions with unrealized losses
- When a position has a meaningful unrealized loss AND the system doesn't have a strong bullish signal for it, harvest the loss (sell, book the loss, optionally re-enter immediately — India has no wash sale rule, unlike the US 30-day rule)
- Prioritize harvesting STCL (offsets both STCG and LTCG) over LTCL (only offsets LTCG)
- Near fiscal year-end (February-March), run an aggressive harvest sweep
- For crypto: loss harvesting has zero value (India's Section 115BBH forbids any loss set-off, even between different crypto assets) — skip entirely for crypto asset class

**4d. LTCG Exemption Optimization:**

Track how much of the ₹1.25L annual LTCG exemption (Section 112A, listed equity only) has been consumed. If there's remaining headroom near year-end, proactively realize long-term gains up to the exemption limit — this is "free" rebalancing that incurs zero tax.

**4e. Lot-Level Accounting (Specific Identification):**

When selling a partial position, choose which specific lots to sell. India's default is FIFO (first-in-first-out), but specific identification is allowed if documented. The system should choose lots that minimize tax impact:
1. Sell loss lots first (to harvest losses)
2. Sell lots past 365 days before lots under 365 days (LTCG rate < STCG rate)
3. Sell lots with smallest gains first when all are LTCG
4. Within STCG lots, sell those closest to the 365-day threshold last (they'll cross into LTCG territory soon)

**4f. Fiscal Year Tax Budget:**

Maintain a running ledger throughout April 1 - March 31 (Indian fiscal year):
- Cumulative realized STCG and LTCG
- Cumulative realized STCL and LTCL
- Net tax liability estimate (with surcharge and cess)
- Remaining LTCG exemption (₹1.25L for listed equity)
- Carried-forward losses from prior years (and their expiry year — 8 year limit)
- Projected year-end tax under current trajectory

**Asset-Class-Specific Tax Behavior:**

| Asset Class | Tax Regime | Implications for System Behavior |
|---|---|---|
| Indian Listed Equity | STCG 20% (Section 111A), LTCG 12.5% (Section 112A), ₹1.25L exemption, loss offset allowed, no wash sale | Standard tax filter — patience window active, loss harvesting valuable, lot optimization matters |
| Crypto (VDA) | 30% flat (Section 115BBH), 1% TDS (Section 194S), NO loss offset at all, no STCG/LTCG distinction | Much wider rebalance thresholds — only sell on very strong signals. Loss harvesting is worthless. TDS tracking required. |
| US Equities (future) | US withholding 25% on dividends (DTAA), LTCG in India after 24 months for unlisted / 12 months for listed on recognized exchange, foreign tax credit available | DTAA credit tracking needed. Different holding period rules. |
| F&O (if ever added) | Business income at slab rate, STT significantly increased in Budget 2024/2025/2026 | Entirely different tax treatment — speculative business income rules apply |
| Commodities (future) | STCG at slab rate, LTCG 12.5% after 24 months, no Section 112A exemption | Longer holding period for LTCG. No ₹1.25L exemption. |

**Transaction Cost Summary (Indian Equity Delivery, per trade):**

| Charge | Rate | Applied To | Side |
|---|---|---|---|
| STT | 0.1% | Transaction value | Both buy and sell |
| Stamp Duty | 0.015% | Transaction value | Buy side only |
| Exchange Transaction Charges (NSE) | ~0.00297-0.00325% | Turnover | Both |
| SEBI Turnover Fee | 0.0001% (₹10 per crore) | Turnover | Both |
| Brokerage (discount broker) | ₹20 flat per order | Per order | Both |
| GST | 18% | On (brokerage + exchange charges + SEBI fees) | Both |
| **Approximate round-trip cost** | **~0.25-0.35%** | | |

These costs are added to the tax_drag_estimate when computing effective_threshold. A round trip that captures only 0.5% pre-tax alpha is actually negative after transaction costs + STCG tax.

### 2.9 Crypto On-Chain Fundamentals

For crypto assets, traditional financial statement analysis doesn't apply. Instead, on-chain data provides the fundamental layer. These metrics are the crypto equivalent of Buffett's financial statement checklist — they measure the health, adoption, and economic activity of the underlying network.

**Network Activity & Adoption:**

| Metric | What It Measures | Bullish Signal |
|---|---|---|
| Daily Active Addresses (DAA) | Unique addresses transacting per day | Sustained increase in DAA |
| New Address Growth Rate | Rate of new wallets being created | Accelerating adoption |
| Transaction Count | Number of on-chain transactions per day | Growing network usage |
| Transaction Volume (USD) | Total value transferred on-chain | Increasing economic activity |
| NVT Ratio (Network Value to Transactions) | Market cap / daily transaction volume — crypto's P/E ratio | Low NVT = undervalued relative to usage |
| NVT Signal (smoothed) | 90-day MA of NVT — reduces noise | Declining NVT Signal |

**Supply & Holder Behavior:**

| Metric | What It Measures | Bullish Signal |
|---|---|---|
| Exchange Net Flow | Net movement of coins to/from exchanges | Negative (outflow) = accumulation |
| Exchange Reserve | Total coins held on exchanges | Declining reserves |
| HODL Waves (1yr+ holder %) | % of supply unmoved for 1+ year | Increasing long-term holding |
| Realized Cap vs Market Cap (MVRV Ratio) | Market cap / realized cap — measures aggregate profit/loss | MVRV < 1 = historically undervalued |
| SOPR (Spent Output Profit Ratio) | Avg profit ratio of coins being spent | SOPR < 1 = selling at loss (capitulation) |
| Supply in Profit % | % of circulating supply currently in profit | Extreme lows = bottom signals |
| Whale Concentration (top 100 addresses) | % of supply held by largest holders | Context-dependent — increasing can be accumulation or centralization risk |
| Stablecoin Supply Ratio (SSR) | BTC market cap / total stablecoin supply — measures buying power | Low SSR = lots of dry powder |

**Mining & Security (PoW chains):**

| Metric | What It Measures | Bullish Signal |
|---|---|---|
| Hash Rate | Total network computing power | Growing = increasing security commitment |
| Hash Rate vs Price divergence | Hash rate up while price down | Miners are committed despite price — bullish divergence |
| Miner Revenue / Transaction Fees | Fee revenue as % of miner income | Sustainable fee market = healthy long-term economics |
| Difficulty Ribbon | Compressed difficulty bands | Ribbon compression = miner capitulation ending |
| Puell Multiple | Daily miner revenue / 365-day MA of revenue | Below 0.5 = historically undervalued |

**DeFi & Ecosystem (for L1/L2 chains):**

| Metric | What It Measures | Bullish Signal |
|---|---|---|
| Total Value Locked (TVL) | Value of assets deposited in DeFi protocols | Growing TVL |
| TVL / Market Cap ratio | Capital efficiency of the chain | High ratio = strong usage relative to valuation |
| DEX Volume / CEX Volume ratio | Shift toward on-chain trading | Increasing ratio |
| Developer Activity (GitHub commits) | Active development on the protocol | Sustained commit activity |
| Protocol Revenue | Fees generated by the protocol | Growing revenue |

**BTC Dominance Cycle:**

| Metric | What It Measures | Signal |
|---|---|---|
| BTC Dominance % | BTC market cap / total crypto market cap | Rising dominance = risk-off / alt-season ending |
| BTC Dominance trend vs altcoin scores | Whether to tilt toward BTC or alts | High BTC dominance + declining trend = potential alt rotation |

**Data Sources:** Glassnode (freemium API), CoinGecko (free API for basic metrics), CryptoQuant, IntoTheBlock, Messari. Free tiers provide sufficient data for scoring — premium APIs are an upgrade path.

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────┐
│           STREAMLIT DASHBOARD (Development UI)           │
│   Scores · Portfolio · Trades · Backtest · Tax Ledger    │
│   Runs locally on laptop during development              │
│   ── migrates to Telegram + PWA once algorithm proven ── │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────┴────────────────────────────────┐
│               BACKEND (FastAPI + APScheduler)            │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │     CROSS-ASSET-CLASS ALLOCATION LAYER (top)       │  │
│  │  Sits above the per-asset-class control loops      │  │
│  │  Runs daily or when the regime changes             │  │
│  └────────────┬───────────────────────────────────────┘  │
│               │                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │          UNIFIED CONTROL LOOP                      │  │
│  │  Runs every N minutes (N = per asset class)        │  │
│  │  During market hours (or 24/7 for crypto)          │  │
│  │                                                    │  │
│  │  1. Pull latest prices                             │  │
│  │  2. Update technical indicators                    │  │
│  │  3. Rescore all assets (tech + sent; fund static)  │  │
│  │  4. Compute ideal allocation (pigeonhole)          │  │
│  │  5. Compute error = ideal − actual                 │  │
│  │  6. Apply PID (gains × regime_confidence)          │  │
│  │  7. TAX FILTER — adjust/defer/enhance signals      │  │
│  │  8. Execute paper trades where |signal| > thresh   │  │
│  │  9. Update trailing stops, tax ledger, notify user │  │
│  └────────────┬───────────────────────────────────────┘  │
│               │                                          │
│  ┌────────────┴───────────────────────────────────────┐  │
│  │            COMBINED SCORING ENGINE                 │  │
│  │  Technical (50%) × regime_confidence               │  │
│  │  + Fundamental (30%) [100% pass-through initially] │  │
│  │  + Sentiment (20%)                                 │  │
│  │  Weights configurable per user                     │  │
│  └───┬──────────────┬──────────────────┬──────────────┘  │
│      │              │                  │                  │
│  ┌───┴────┐   ┌─────┴──────┐   ┌──────┴───────┐        │
│  │TECHNCL │   │FUNDAMENTAL │   │  SENTIMENT   │        │
│  │ TA-Lib │   │  Buffett   │   │ Reddit + X   │        │
│  │        │   │ checklist  │   │  FinBERT     │        │
│  │  +     │   │ (income    │   │ (ProsusAI/   │        │
│  │conflu- │   │  stmt +    │   │  finbert)    │        │
│  │ ence   │   │  balance   │   │              │        │
│  │  +     │   │  sheet +   │   │              │        │
│  │regime  │   │  cash flow │   │              │        │
│  │detect  │   │  + crypto  │   │              │        │
│  │(Hurst/ │   │  on-chain) │   │              │        │
│  │Lyapun.)│   │            │   │              │        │
│  └────────┘   └────────────┘   └──────────────┘        │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │           TAX OPTIMIZATION ENGINE                  │  │
│  │  Tax-adjusted thresholds · Patience window         │  │
│  │  Loss harvesting · LTCG exemption optimization     │  │
│  │  Lot-level accounting · Fiscal year ledger          │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │           PAPER TRADING ENGINE                     │  │
│  │  Execute trades · Track P&L · Wallet management    │  │
│  │  Trailing SL · Position limits · Risk parity       │  │
│  │  Performance metrics per user (pre- and post-tax)  │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
├──────────────────────────────────────────────────────────┤
│              DATA LAYER (SQLite → PostgreSQL)            │
│  Intraday prices · Daily prices · Financials             │
│  On-chain metrics · Sentiment · Scores · Users           │
│  Portfolios · Trades · Tax ledgers · Lot tracking        │
└──────────────────────────────────────────────────────────┘
```

---

## 4. Data Model

All tables described here in SQL for precision. Implemented via Python (SQLAlchemy ORM).

### 4.1 Core Tables (Asset-Class Agnostic)

```sql
CREATE TABLE asset_classes (
    id                          INTEGER PRIMARY KEY,
    name                        TEXT NOT NULL,              -- 'equity', 'crypto', 'commodity', 'forex'
    description                 TEXT,
    scoring_interval_minutes    INTEGER DEFAULT 30,         -- how often to rescore
    market_open                 TIME,                       -- '09:15' for NSE, NULL for 24/7
    market_close                TIME,                       -- '15:30' for NSE, NULL for 24/7
    is_24x7                     BOOLEAN DEFAULT FALSE       -- TRUE for crypto
);

CREATE TABLE exchanges (
    id              INTEGER PRIMARY KEY,
    asset_class_id  INTEGER NOT NULL REFERENCES asset_classes(id),
    code            TEXT NOT NULL UNIQUE,    -- 'NSE', 'BSE', 'BINANCE', 'MCX'
    name            TEXT NOT NULL,
    country         TEXT,
    currency        TEXT NOT NULL            -- 'INR', 'USD'
);

CREATE TABLE assets (
    id              INTEGER PRIMARY KEY,
    exchange_id     INTEGER NOT NULL REFERENCES exchanges(id),
    ticker          TEXT NOT NULL,           -- 'RELIANCE.NS', 'BTC-USD'
    name            TEXT NOT NULL,
    asset_type      TEXT,                    -- 'stock', 'etf', 'crypto', 'commodity_future'
    sector          TEXT,                    -- NULL for non-equity
    industry        TEXT,                    -- NULL for non-equity
    is_active       BOOLEAN DEFAULT TRUE,
    UNIQUE(exchange_id, ticker)
);
```

### 4.2 Price Data

```sql
CREATE TABLE price_daily (
    asset_id    INTEGER NOT NULL REFERENCES assets(id),
    date        DATE NOT NULL,
    open        REAL, high REAL, low REAL, close REAL,
    adj_close   REAL,
    volume      INTEGER,
    PRIMARY KEY (asset_id, date)
);

CREATE TABLE price_intraday (
    asset_id    INTEGER NOT NULL REFERENCES assets(id),
    timestamp   DATETIME NOT NULL,
    interval    TEXT NOT NULL,           -- '15m', '30m', '1h'
    open        REAL, high REAL, low REAL, close REAL,
    volume      INTEGER,
    PRIMARY KEY (asset_id, timestamp, interval)
);
```

### 4.3 Fundamental Data — Equity Financials (Expandable)

```sql
CREATE TABLE financials (
    id                  INTEGER PRIMARY KEY,
    asset_id            INTEGER NOT NULL REFERENCES assets(id),
    fiscal_year         INTEGER NOT NULL,
    period_type         TEXT DEFAULT 'annual',      -- 'annual' or 'quarterly'
    filing_date         DATE,

    -- === INCOME STATEMENT ===
    revenue                     REAL,
    cogs                        REAL,
    gross_profit                REAL,
    sga_expense                 REAL,
    rnd_expense                 REAL,
    depreciation                REAL,
    amortization                REAL,
    interest_expense            REAL,
    restructuring               REAL,
    other_operating             REAL,
    ebit                        REAL,
    pretax_income               REAL,
    tax_paid                    REAL,
    net_income                  REAL,
    eps                         REAL,
    shares_outstanding          REAL,

    -- === BALANCE SHEET ===
    -- Current Assets
    cash_and_equivalents        REAL,
    short_term_investments      REAL,
    accounts_receivable         REAL,
    net_receivables             REAL,
    inventory                   REAL,
    prepaid_expenses            REAL,
    other_current_assets        REAL,
    total_current_assets        REAL,
    -- Non-Current Assets
    property_plant_equipment    REAL,
    accumulated_depreciation    REAL,
    net_ppe                     REAL,
    goodwill                    REAL,
    intangible_assets           REAL,
    long_term_investments       REAL,
    other_noncurrent_assets     REAL,
    total_assets                REAL,
    -- Current Liabilities
    accounts_payable            REAL,
    short_term_debt             REAL,
    current_portion_lt_debt     REAL,
    accrued_liabilities         REAL,
    deferred_revenue_current    REAL,
    other_current_liabilities   REAL,
    total_current_liabilities   REAL,
    -- Non-Current Liabilities
    long_term_debt              REAL,
    deferred_tax_liabilities    REAL,
    deferred_revenue_noncurrent REAL,
    pension_obligations         REAL,
    other_noncurrent_liabilities REAL,
    total_liabilities           REAL,
    -- Shareholders Equity
    common_stock                REAL,
    preferred_stock             REAL,
    additional_paid_in_capital  REAL,
    retained_earnings           REAL,
    treasury_stock              REAL,
    accumulated_other_comprehensive_income REAL,
    minority_interest           REAL,
    total_shareholders_equity   REAL,

    -- === CASH FLOW STATEMENT ===
    -- Operating Activities
    operating_cash_flow         REAL,
    depreciation_amortization_cf REAL,
    stock_based_compensation    REAL,
    change_in_working_capital   REAL,
    change_in_receivables       REAL,
    change_in_inventory         REAL,
    change_in_payables          REAL,
    other_operating_activities  REAL,
    -- Investing Activities
    capital_expenditure         REAL,
    acquisitions                REAL,
    purchase_of_investments     REAL,
    sale_of_investments         REAL,
    other_investing_activities  REAL,
    investing_cash_flow         REAL,
    -- Financing Activities
    debt_issued                 REAL,
    debt_repaid                 REAL,
    dividends_paid              REAL,
    share_buybacks              REAL,
    shares_issued               REAL,
    other_financing_activities  REAL,
    financing_cash_flow         REAL,
    -- Derived
    free_cash_flow              REAL,
    net_change_in_cash          REAL,

    UNIQUE(asset_id, fiscal_year, period_type)
);

CREATE TABLE financial_data_availability (
    asset_id                    INTEGER NOT NULL REFERENCES assets(id),
    fiscal_year                 INTEGER NOT NULL,
    period_type                 TEXT NOT NULL,
    income_stmt_available       BOOLEAN DEFAULT FALSE,
    balance_sheet_available     BOOLEAN DEFAULT FALSE,
    cash_flow_available         BOOLEAN DEFAULT FALSE,
    last_updated                DATETIME,
    PRIMARY KEY (asset_id, fiscal_year, period_type)
);
```

### 4.4 Fundamental Data — Crypto On-Chain Metrics

```sql
CREATE TABLE onchain_metrics (
    id                          INTEGER PRIMARY KEY,
    asset_id                    INTEGER NOT NULL REFERENCES assets(id),
    date                        DATE NOT NULL,
    -- Network Activity
    daily_active_addresses      INTEGER,
    new_addresses               INTEGER,
    transaction_count           INTEGER,
    transaction_volume_usd      REAL,
    nvt_ratio                   REAL,
    nvt_signal_90d              REAL,
    -- Supply & Holders
    exchange_net_flow           REAL,
    exchange_reserve            REAL,
    hodl_wave_1yr_plus_pct      REAL,
    mvrv_ratio                  REAL,
    sopr                        REAL,
    supply_in_profit_pct        REAL,
    whale_concentration_top100  REAL,
    stablecoin_supply_ratio     REAL,
    -- Mining (PoW only, NULL for PoS)
    hash_rate                   REAL,
    difficulty                  REAL,
    miner_revenue_usd           REAL,
    fee_revenue_pct             REAL,
    puell_multiple              REAL,
    -- DeFi / Ecosystem
    tvl_usd                     REAL,
    tvl_market_cap_ratio        REAL,
    dex_volume_usd              REAL,
    protocol_revenue_usd        REAL,
    developer_commits_30d       INTEGER,
    -- Market Structure
    btc_dominance_pct           REAL,
    funding_rate                REAL,
    open_interest_usd           REAL,
    UNIQUE(asset_id, date)
);
```

### 4.5 Scoring & Indicators

```sql
CREATE TABLE indicator_cache (
    asset_id    INTEGER NOT NULL REFERENCES assets(id),
    timestamp   DATETIME NOT NULL,
    indicator   TEXT NOT NULL,              -- 'RSI_14', 'MACD', 'HURST_100', etc.
    value       REAL,
    PRIMARY KEY (asset_id, timestamp, indicator)
);

CREATE TABLE scores (
    asset_id            INTEGER NOT NULL REFERENCES assets(id),
    timestamp           DATETIME NOT NULL,
    technical_score     REAL,               -- 0-100
    fundamental_score   REAL,               -- 0-100
    sentiment_score     REAL,               -- 0-100
    regime_confidence   REAL,               -- 0-1
    combined_score      REAL,               -- weighted composite, regime-adjusted
    signal_reasons      TEXT,               -- JSON: which indicators triggered and why
    PRIMARY KEY (asset_id, timestamp)
);
```

### 4.6 Sentiment

```sql
CREATE TABLE sentiment_raw (
    id          INTEGER PRIMARY KEY,
    asset_id    INTEGER REFERENCES assets(id),
    source      TEXT NOT NULL,                   -- 'reddit', 'twitter'
    subreddit   TEXT,
    post_id     TEXT,
    timestamp   DATETIME NOT NULL,
    text        TEXT,
    sentiment   REAL,                            -- -1 to +1 (FinBERT)
    magnitude   REAL,
    model_version TEXT                           -- 'finbert_v1', 'finbert_finetuned_v1'
);

CREATE TABLE sentiment_aggregated (
    asset_id        INTEGER NOT NULL REFERENCES assets(id),
    date            DATE NOT NULL,
    source          TEXT NOT NULL,
    avg_sentiment   REAL,
    mention_count   INTEGER,
    bullish_pct     REAL,
    bearish_pct     REAL,
    neutral_pct     REAL,
    contrarian_flag BOOLEAN DEFAULT FALSE,       -- extreme bullish + high volume
    PRIMARY KEY (asset_id, date, source)
);
```

### 4.7 Users — Exhaustive Profile

```sql
CREATE TABLE users (
    id                          INTEGER PRIMARY KEY,
    created_at                  DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_active_at              DATETIME,

    -- === IDENTITY & ACCOUNT ===
    name                        TEXT,
    email                       TEXT,
    phone                       TEXT,
    telegram_id                 TEXT UNIQUE,
    pan_number                  TEXT,                       -- for tax reporting integration (future)
    country_of_residence        TEXT DEFAULT 'IN',          -- drives tax regime: 'IN', 'US', etc.
    tax_residency_status        TEXT DEFAULT 'resident',    -- 'resident', 'nri', 'rnor'
    age                         INTEGER,
    kyc_status                  TEXT DEFAULT 'pending',     -- 'pending', 'verified' (future, real money)

    -- === CAPITAL & CASH FLOW ===
    initial_capital             REAL DEFAULT 0,
    monthly_contribution_amount REAL DEFAULT 0,             -- SIP-like inflows
    monthly_contribution_day    INTEGER DEFAULT 1,          -- day of month
    monthly_withdrawal_amount   REAL DEFAULT 0,             -- if drawing income from portfolio
    emergency_reserve_amount    REAL DEFAULT 0,             -- system won't deploy this
    cash_buffer_pct             REAL DEFAULT 5.0,           -- min % always in wallet

    -- === TIME HORIZON & GOALS ===
    investment_horizon_years    INTEGER DEFAULT 5,
    primary_goal                TEXT DEFAULT 'wealth_growth', -- 'wealth_growth', 'retirement', 'house_purchase', 'child_education', 'income_generation'
    target_corpus               REAL,                       -- optional goal amount
    target_date                 DATE,                       -- optional goal date
    drawdown_tolerance_pct      REAL DEFAULT 20.0,          -- max peak-to-trough before panic

    -- === RISK & STRATEGY PREFERENCES (Global Defaults) ===
    overall_risk_appetite       INTEGER DEFAULT 50,         -- 0 (ultra-conservative) to 100 (ultra-aggressive)
    max_single_position_pct     REAL DEFAULT 20.0,          -- max % in one asset
    max_sector_concentration_pct REAL DEFAULT 40.0,         -- max % in one sector
    min_score_threshold         REAL DEFAULT 50.0,          -- min combined score to consider
    rebalance_sensitivity       INTEGER DEFAULT 50,         -- 0-100, maps to PID threshold
    max_daily_loss_pct          REAL DEFAULT 5.0,           -- kill switch (future, real money)

    -- Factor Tilts (1.0 = neutral, >1 = overweight, <1 = underweight)
    factor_tilt_momentum        REAL DEFAULT 1.0,
    factor_tilt_value           REAL DEFAULT 1.0,
    factor_tilt_quality         REAL DEFAULT 1.0,
    factor_tilt_low_volatility  REAL DEFAULT 1.0,

    -- === SCORING WEIGHT PREFERENCES ===
    weight_technical            REAL DEFAULT 0.50,
    weight_fundamental          REAL DEFAULT 0.30,
    weight_sentiment            REAL DEFAULT 0.20,
    allow_contrarian_signals    BOOLEAN DEFAULT TRUE,

    -- === TAX PROFILE ===
    annual_income_slab          TEXT DEFAULT '30pct',        -- 'nil', '5pct', '20pct', '30pct' — for dividend/F&O taxation
    filing_status               TEXT DEFAULT 'individual',   -- 'individual', 'huf'
    tax_optimization_aggressiveness INTEGER DEFAULT 50,      -- 0-100, how much to defer for tax
    auto_harvest_losses         BOOLEAN DEFAULT TRUE,
    prefer_ltcg_exemption_usage BOOLEAN DEFAULT TRUE,        -- realize LTCG up to exemption near year-end
    tax_patience_window_days    INTEGER DEFAULT 45,          -- days before LTCG threshold to activate patience

    -- === BEHAVIORAL & NOTIFICATION PREFERENCES ===
    notification_channel        TEXT DEFAULT 'telegram',     -- 'telegram', 'email', 'push', 'sms' (comma-separated for multi)
    notification_frequency      TEXT DEFAULT 'real_time',    -- 'real_time', 'digest_daily', 'digest_weekly'
    notification_types_enabled  TEXT DEFAULT 'trades,exits,rebalance_warnings,tax_events',
    quiet_hours_start           TIME DEFAULT '22:00',
    quiet_hours_end             TIME DEFAULT '07:00',
    timezone                    TEXT DEFAULT 'Asia/Kolkata',
    language_preference         TEXT DEFAULT 'en',           -- 'en', 'hi'
    require_confirmation_above  REAL,                        -- trades above ₹X need confirmation (future, real money)

    -- === ETHICAL / ESG FILTERS ===
    exclude_sectors             TEXT,                        -- JSON: ['tobacco', 'gambling', 'defense', 'fossil_fuels']
    exclude_tickers             TEXT,                        -- JSON: ['TICKER1', 'TICKER2']
    require_esg_rating_min      REAL,                       -- minimum ESG rating (future)
    shariah_compliant_only      BOOLEAN DEFAULT FALSE,

    -- === PERFORMANCE & REPORTING ===
    benchmark_preference        TEXT DEFAULT 'NIFTY50',      -- 'NIFTY50', 'NIFTY500', 'custom'
    report_delivery_frequency   TEXT DEFAULT 'weekly',       -- 'daily', 'weekly', 'monthly'
    report_format               TEXT DEFAULT 'dashboard_only', -- 'email_summary', 'pdf', 'dashboard_only'
    show_after_tax_returns      BOOLEAN DEFAULT TRUE,

    -- === SYSTEM CONFIGURATION ===
    is_paper_trading            BOOLEAN DEFAULT TRUE,
    paper_trading_start_date    DATE,
    real_trading_allowed        BOOLEAN DEFAULT FALSE,       -- future gating
    kill_switch_active          BOOLEAN DEFAULT FALSE,       -- emergency: force all to cash

    -- === EXPERIMENTAL / FUTURE ===
    allows_leverage             BOOLEAN DEFAULT FALSE,
    allows_shorting             BOOLEAN DEFAULT FALSE,
    allows_derivatives          BOOLEAN DEFAULT FALSE,
    ml_model_version_preference TEXT DEFAULT 'latest',       -- A/B testing ready
    backtest_profile_id         INTEGER                      -- which calibration set
);
```

### 4.8 Per-Asset-Class Preferences

```sql
CREATE TABLE user_asset_class_preferences (
    id                          INTEGER PRIMARY KEY,
    user_id                     INTEGER NOT NULL REFERENCES users(id),
    asset_class_id              INTEGER NOT NULL REFERENCES asset_classes(id),
    is_enabled                  BOOLEAN DEFAULT TRUE,       -- FALSE = force allocation to 0, liquidate
    risk_appetite               INTEGER DEFAULT 50,         -- 0-100, per-class override

    -- PID gains derived from risk_appetite (computed, not user-set)
    kp                          REAL DEFAULT 1.0,
    ki                          REAL DEFAULT 0.5,
    kd                          REAL DEFAULT 0.5,

    -- Constraints
    trailing_sl_pct             REAL DEFAULT 10.0,
    max_class_allocation_pct    REAL DEFAULT 100.0,          -- max % of total portfolio in this class
    min_class_allocation_pct    REAL DEFAULT 0.0,            -- minimum % — forces diversification
    rebalance_frequency_override INTEGER,                    -- optional, slower than scoring interval

    -- Tax behavior per class
    tax_patience_enabled        BOOLEAN DEFAULT TRUE,        -- 365-day cliff awareness
    loss_harvesting_enabled     BOOLEAN DEFAULT TRUE,        -- proactive loss harvesting

    UNIQUE(user_id, asset_class_id)
);
```

### 4.9 Wallets, Portfolios, Trading

```sql
CREATE TABLE wallets (
    id          INTEGER PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id),
    currency    TEXT DEFAULT 'INR',
    balance     REAL DEFAULT 0.0,
    updated_at  DATETIME
);

CREATE TABLE portfolios (
    id              INTEGER PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(id),
    asset_id        INTEGER NOT NULL REFERENCES assets(id),
    quantity        REAL NOT NULL,
    avg_price       REAL NOT NULL,
    date_entered    DATE NOT NULL,
    trailing_sl     REAL,
    is_paper        BOOLEAN DEFAULT TRUE,
    UNIQUE(user_id, asset_id, is_paper)
);

CREATE TABLE trades (
    id              INTEGER PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(id),
    asset_id        INTEGER NOT NULL REFERENCES assets(id),
    action          TEXT NOT NULL,           -- 'BUY', 'SELL'
    quantity        REAL NOT NULL,
    price           REAL NOT NULL,
    timestamp       DATETIME NOT NULL,
    combined_score  REAL,
    signal_reasons  TEXT,                    -- JSON
    is_paper        BOOLEAN DEFAULT TRUE,
    pnl             REAL,                   -- realized P&L for sells
    -- Tax fields
    tax_type        TEXT,                   -- 'STCG', 'LTCG', 'FLAT_30', 'SLAB'
    tax_amount      REAL,                   -- estimated tax on this trade
    lot_ids_sold    TEXT,                   -- JSON: which lots were sold
    stt_paid        REAL,
    stamp_duty_paid REAL,
    transaction_costs REAL                  -- total non-tax friction
);

CREATE TABLE notifications (
    id              INTEGER PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(id),
    trade_id        INTEGER REFERENCES trades(id),
    message         TEXT NOT NULL,
    sent_at         DATETIME,
    read_at         DATETIME,
    action_taken    TEXT                    -- 'reinvest', 'hold', NULL
);
```

### 4.10 Tax Tables

```sql
CREATE TABLE tax_regimes (
    id                      INTEGER PRIMARY KEY,
    country                 TEXT NOT NULL,              -- 'IN', 'US'
    asset_class_id          INTEGER REFERENCES asset_classes(id),
    regime_name             TEXT NOT NULL,              -- 'IN_EQUITY_LISTED', 'IN_CRYPTO_VDA', etc.
    -- Capital Gains
    stcg_rate               REAL,                       -- 0.20 for IN equity
    ltcg_rate               REAL,                       -- 0.125 for IN equity
    ltcg_holding_days       INTEGER,                    -- 365 for IN listed equity, 730 for others
    ltcg_annual_exemption   REAL,                       -- 125000 for IN equity (Section 112A only)
    flat_tax_rate           REAL,                       -- 0.30 for IN crypto (overrides STCG/LTCG)
    -- Loss Rules
    loss_offset_allowed     BOOLEAN,                    -- TRUE for equity, FALSE for crypto
    stcl_offsets            TEXT,                       -- 'stcg_and_ltcg' or 'none'
    ltcl_offsets            TEXT,                       -- 'ltcg_only' or 'none'
    loss_carryforward_years INTEGER,                    -- 8 for India
    wash_sale_days          INTEGER DEFAULT 0,          -- 0 for India, 30 for US
    -- Transaction Costs
    tds_rate                REAL,                       -- 0.01 for crypto
    stt_buy_rate            REAL,                       -- 0.001 for equity delivery
    stt_sell_rate           REAL,                       -- 0.001 for equity delivery
    stamp_duty_rate         REAL,                       -- 0.00015 for equity delivery (buy side)
    exchange_txn_charge_rate REAL,                      -- ~0.0000325 (NSE)
    sebi_turnover_fee_rate  REAL,                       -- 0.000001 (₹10/crore)
    gst_rate                REAL DEFAULT 0.18,          -- 18% on brokerage + exchange charges
    -- Validity
    effective_from          DATE,
    effective_to            DATE                        -- NULL if current
);

CREATE TABLE user_tax_ledger (
    id                          INTEGER PRIMARY KEY,
    user_id                     INTEGER NOT NULL REFERENCES users(id),
    fiscal_year                 TEXT NOT NULL,               -- '2025-26' (April-March)
    -- Realized gains/losses
    realized_stcg              REAL DEFAULT 0,
    realized_ltcg              REAL DEFAULT 0,
    realized_stcl              REAL DEFAULT 0,
    realized_ltcl              REAL DEFAULT 0,
    -- Exemptions
    ltcg_exemption_used        REAL DEFAULT 0,              -- of ₹1.25L
    ltcg_exemption_remaining   REAL DEFAULT 125000,
    -- Tax estimate
    estimated_tax_liability    REAL DEFAULT 0,
    stt_paid_ytd               REAL DEFAULT 0,
    stamp_duty_paid_ytd        REAL DEFAULT 0,
    total_transaction_costs_ytd REAL DEFAULT 0,
    -- Carried forward
    carried_forward_stcl       REAL DEFAULT 0,
    carried_forward_ltcl       REAL DEFAULT 0,
    cf_stcl_by_year            TEXT,                        -- JSON: {"2024-25": 50000, "2023-24": 30000}
    cf_ltcl_by_year            TEXT,                        -- JSON: same, with expiry tracking
    -- Crypto-specific
    crypto_realized_gains      REAL DEFAULT 0,              -- tracked separately (30% flat, no offset)
    crypto_tds_paid            REAL DEFAULT 0,
    --
    last_updated               DATETIME,
    UNIQUE(user_id, fiscal_year)
);

CREATE TABLE lot_tracking (
    id              INTEGER PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(id),
    asset_id        INTEGER NOT NULL REFERENCES assets(id),
    buy_trade_id    INTEGER REFERENCES trades(id),
    quantity        REAL NOT NULL,
    buy_price       REAL NOT NULL,
    buy_date        DATE NOT NULL,
    remaining_qty   REAL NOT NULL,                          -- decreases as lots are sold
    holding_days    INTEGER GENERATED ALWAYS AS
                     (julianday('now') - julianday(buy_date)) STORED,
    unrealized_gain REAL,                                   -- updated periodically
    tax_status      TEXT GENERATED ALWAYS AS
                     (CASE WHEN julianday('now') - julianday(buy_date) >= 365
                      THEN 'LTCG' ELSE 'STCG' END) STORED,
    days_to_ltcg    INTEGER GENERATED ALWAYS AS
                     (CASE WHEN julianday('now') - julianday(buy_date) < 365
                      THEN 365 - (julianday('now') - julianday(buy_date))
                      ELSE 0 END) STORED
);
```

---

## 5. Scoring Engines — Detailed Design

### 5.1 Technical Scoring Engine

**Library:** TA-Lib (C library plus `ta-lib-python`) for standard indicators, with our own `_extensions.py` for indicators TA-Lib lacks, plus custom confluence and regime detection logic. `indicators.py` is the public API (`add_all_indicators`) over `_talib_core.py` and `_extensions.py`.

**Why pandas-ta was dropped:** the library is abandoned and has lookahead bias in DPO and Ichimoku. DPO is therefore not implemented, and Ichimoku's Chikou Span is excluded (it is the close shifted back in time, i.e. lookahead); Tenkan, Kijun and Senkou A/B are computed.

**Indicator List (TA-Lib + `_extensions.py`):** each family below keeps the original plan list; names under "Not implemented" are not computed by the indicator layer. Column names are a contract (`RSI_14`, `MACD_12_26_9`, ...).

**Momentum (38 in the plan list, 15 implemented):**
Implemented: APO (Absolute Price Oscillator), BOP (Balance of Power), CCI (Commodity Channel Index, period=20), CMO (Chande Momentum Oscillator), MACD (12,26,9), MOM (Momentum), PPO (Percentage Price Oscillator), ROC (Rate of Change), RSI (14), SLOPE (Linear Regression Slope, as `LINEARREG_SLOPE_14`), STOCH (Stochastic Oscillator 14,3), STOCHRSI (Stochastic RSI), TRIX (Triple Exponential Average), UO (Ultimate Oscillator), WILLR (Williams %R)
Not implemented: AO, BIAS, BRAR, CFO, CG, COPPOCK, CTI, ER, ERI, FISHER, INERTIA, KDJ, KST, PGO, PSL, PVO, QQE, RSX, RVGI, SMI, STC, TD_SEQ, TSI

**Overlap / Moving Averages (33 in the plan list, 17 implemented):**
Implemented: DEMA (Double EMA), EMA (Exponential MA — periods: 5, 8, 13, 20, 50, 100, 200), HL2 (as `MEDPRICE`), HLC3 (as `TYPPRICE`), ICHIMOKU (Tenkan, Kijun, Senkou A/B; Chikou excluded, see above), KAMA (Kaufman Adaptive MA), LINREG (Linear Regression), MIDPOINT, MIDPRICE, OHLC4 (as `AVGPRICE`), SMA (Simple MA — periods: 5, 10, 20, 50, 100, 200), SUPERTREND (ATR-based), T3 (Tillson T3), TEMA (Triple EMA), TRIMA (Triangular MA), WCP (Weighted Close Price, as `WCLPRICE`), WMA (Weighted MA)
Not implemented: ALMA, FWMA, HILO, HMA, HWMA, JMA, MCGD, PWMA, RMA, SINWMA, SSF, SWMA, VIDYA, VWAP, VWMA, ZLMA

**Trend (16 in the plan list, 4 implemented):**
Implemented: ADX (Average Directional Index, 14), AROON (Aroon Up/Down/Oscillator), PSAR (Parabolic SAR), VORTEX (Vortex Indicator — VI+ and VI-)
Not implemented: AMAT, CHOP, CKSP, DECAY, DPO (lookahead), INCREASING, DECREASING, LONG_RUN, SHORT_RUN, QSTICK, TTM_TREND, VHF

**Volatility (14 in the plan list, 5 implemented):**
Implemented: ATR (Average True Range, 14), BBANDS (Bollinger Bands 20,2), KC (Keltner Channel), NATR (Normalized ATR), TRUE_RANGE
Not implemented: ABERRATION, ACCBANDS, DONCHIAN, HWCHANNEL, MASSI, PDIST, RVI, THERMO, UI

**Volume (16 in the plan list, 5 implemented):**
Implemented: AD (Accumulation/Distribution), ADOSC (AD Oscillator), MFI (Money Flow Index, 14), OBV (On-Balance Volume), VP (Volume Profile, computed in `support_resistance.py`)
Not implemented: AOBV, CMF, EFI, EOM, KVO, NVI, PVI, PVOL, PVR, PVT, VWAP (also in Overlap)

**Statistics (10 in the plan list, 2 implemented):**
Implemented: STDEV (as `STDDEV_20`), VARIANCE (as `VAR_20`)
Not implemented: ENTROPY, KURTOSIS, MAD, MEDIAN, QUANTILE, SKEW, TOS_STDEVALL, ZSCORE

**Candle Patterns (`candle_patterns.py`: all 61 TA-Lib CDL* functions, normalised to -1/0/+1 plus a strength score; computed from OHLC):**
CDL_DOJI, CDL_HAMMER, CDL_HANGINGMAN, CDL_SHOOTINGSTAR, CDL_MARUBOZU_BULL, CDL_MARUBOZU_BEAR, CDL_ENGULFING_BULL, CDL_ENGULFING_BEAR, CDL_HARAMI_BULL, CDL_HARAMI_BEAR, CDL_MORNINGSTAR, CDL_EVENINGSTAR, CDL_PIERCING, CDL_DARKCLOUDCOVER, CDL_SPINNINGTOP, CDL_THREEWHITESOLDIERS, CDL_THREEBLACKCROWS
Not implemented: CDL_INSIDE (Inside Bar; TA-Lib has no such function)

**Performance (4 metrics, implemented as `LOGRET_1`, `CUMLOGRET_1`, `PCTRET_1`, `CUMPCTRET_1`):**
LOG_RETURN, CUMLOG_RETURN, PCT_RETURN, CUMPCT_RETURN

**Cycles (1 in the plan list, not implemented):**
EBSW (Even Better Sine Wave)

**Custom Indicators (not in TA-Lib):**

| Indicator | Implementation | Purpose |
|---|---|---|
| Hurst Exponent (rolling, 100 and 250 bars) | Rolling DFA-1 (detrended fluctuation analysis) on log returns, `HURST_WINDOWS = (100, 250)` in `regime.py`; R/S analysis was not used because it is biased upwards on short samples | Regime detection |
| Lyapunov Exponent (simplified) | Divergence rate of nearby trajectories | Chaos detection |
| Volume-Weighted S/R Zones | Swing-pivot clusters (3/3 pivots, ATR-based cluster tolerance 0.25 × ATR_14, `SR_STRENGTH_*` as a 0–1 share of total recency weight, comparable across assets) plus volume profile (`VP_POS` clipped, `VP_DIST` unclipped in ATR units) | Key levels |
| Fibonacci Retracement Levels | 23.6%, 38.2%, 50%, 61.8%, 78.6% | Order flow clustering |
| Multi-Timeframe Alignment Score | Cross-timeframe signal agreement | Confirmation bonus |
| Squeeze Detector | BB inside KC | Volatility compression → expansion |
| Divergence Detector (RSI, MACD, OBV) | Price vs indicator divergence (not implemented) | Reversal early warning |

**Total: 112 indicator columns per asset per timestamp from `add_all_indicators`**, plus the candle-pattern, support/resistance and volume-profile, regime and multi-timeframe outputs of their own modules.

**Confluence Scoring (0-100):**

Each indicator generates a sub-signal: bullish (+1), neutral (0), or bearish (-1), following momentum and fading to 0 at oscillator extremes (not mean reversion), weighted by historical reliability. The score is damped ×0.5 in chaotic regimes and is NaN until every family is out of warmup. Confluence score = weighted sum, normalized to [-1, 1] and stored in the separate `technical_scores` table (750 daily / 1600 15-minute bar lookback, a daily bar counts from 16:00 IST, stale after 3 business days). The 0-100 `technical_score` of the `scores` table (§4.5) is produced by the combined layer from this value; that mapping is to be designed. Mean-reversion and regime-dependent polarity variants are tested in the harness (§9.4), not chosen by hand.

Initially, confluence weights are hand-configured based on indicator category and known reliability, loaded from a config file. After backtesting generates a labeled dataset, PCA + XGBoost/LightGBM replaces hand-tuned weights with empirically optimized, nonlinear scoring. Same scorer interface, different weight source.

**Multi-Timeframe:** Indicators computed on 15-minute candles (intraday signals), 1-hour candles, and daily candles (trend context), weighted 1/2/3 so the higher timeframe dominates (less 15m noise, fewer trades, less STCG). Score bonus when multiple timeframes agree.

### 5.2 Fundamental Scoring Engine

**Point-in-time rule (applies to every sub-score below):** a filing's numbers count only from its publication date, or, conservatively when the publication date is not known, from period end + 45 days (quarterly) / + 60 days (annual), per SEBI LODR Reg. 33. yfinance statements carry period-end dates, not filing dates, so using them as of period end is lookahead. This is a different lag from the 21-day shareholding-filing lag in §5.5. Fundamental scoring is paused until fundamentals are point-in-time.

**Equity — Income Statement Sub-Scores (from Buffett notes):**

| Metric | Calculation | Scoring Rule |
|---|---|---|
| Gross Margin | gross_profit / revenue (avg available years) | >40%: 100, 20-40%: 60, <20%: 20 |
| Gross Margin Consistency | std deviation of gross margin over years | Low variance: 100, High: 40 |
| SGA Ratio | sga / gross_profit (avg) | <30%: 100, 30-50%: 60, >50%: 20 |
| SGA Consistency | std deviation over years | Low: 100, High: 40 |
| R&D Burden | rnd / gross_profit | <10%: 100, 10-30%: 60, >30%: 20 |
| Depreciation Burden | depreciation / ebit | <10%: 100, 10-15%: 70, >15%: 40 |
| Interest Burden | interest_expense / ebit | <15%: 100, 15-25%: 60, >25%: 20 |
| Interest Burden (Industry Relative) | rank within industry peers | Lowest in industry: bonus |
| Net Margin | net_income / revenue | >20%: 100, 10-20%: 60, <10%: 20 |
| Net Margin Trend | slope of net margin over time | Positive slope: bonus, Negative: penalty |
| EPS Consistency | is EPS consistently increasing? | Consistent increase: 100, Volatile: 40 |
| Tax Honesty | tax_paid vs expected rate × pretax_income | Match: no penalty, Mismatch: flag |

**Equity — Balance Sheet Sub-Scores (to be completed from book notes):**

| Metric | Calculation | Scoring Rule |
|---|---|---|
| Current Ratio | total_current_assets / total_current_liabilities | 1.5-2.5: 100, 1.0-1.5 or 2.5-4.0: 60, <1.0 or >4.0: 20 |
| Quick Ratio | (current_assets - inventory) / current_liabilities | >1.0: 100, 0.5-1.0: 60, <0.5: 20 |
| Cash to Total Assets | cash_and_equivalents / total_assets | Context-dependent — high cash can mean strength or poor capital allocation |
| Receivables Growth vs Revenue Growth | YoY receivables growth vs YoY revenue growth | Receivables growing slower: 100, Faster: flag |
| Inventory Turnover | cogs / inventory | Higher is better (industry-relative) |
| Debt to Equity | total_liabilities / total_shareholders_equity | <0.5: 100, 0.5-1.0: 70, 1.0-2.0: 40, >2.0: 20 |
| Long-term Debt to Net Income | long_term_debt / net_income | <3 years earnings to repay: 100, 3-5: 60, >5: 20 |
| Retained Earnings Trend | YoY growth in retained_earnings | Consistent positive growth: 100, Declining: 20 |
| ROE (Return on Equity) | net_income / total_shareholders_equity | >20%: 100, 10-20%: 60, <10%: 20 |
| ROA (Return on Assets) | net_income / total_assets | >10%: 100, 5-10%: 60, <5%: 20 |
| Goodwill Trajectory | YoY change in goodwill | Stable/declining: 100, Rapidly increasing (via acquisitions): 40 |
| Treasury Stock | presence and growth of treasury_stock | Share buybacks = positive signal |
| Preferred Stock | presence of preferred_stock | Preferred stock = negative for common shareholders |

**Equity — Cash Flow Sub-Scores (to be completed from book notes):**

| Metric | Calculation | Scoring Rule |
|---|---|---|
| Operating CF Quality | operating_cash_flow / net_income | >1.0: 100 (earnings are backed by real cash), <0.8: flag |
| CapEx Intensity | capital_expenditure / net_income | <25%: 100, 25-50%: 60, >50%: 20 |
| Free Cash Flow Yield | free_cash_flow / market_cap | >5%: strong, 2-5%: moderate, <2%: weak |
| FCF Consistency | std deviation of FCF over years | Low variance: 100, High: 40 |
| Buyback vs Dilution | share_buybacks vs shares_issued | Net buyback: positive, Net dilution: negative |
| Dividend Sustainability | dividends_paid / free_cash_flow | <60%: sustainable, 60-80%: warning, >80%: dangerous |
| Debt Repayment Pattern | debt_repaid / debt_issued over time | Net repayment trend: positive |
| Cash Generation Efficiency | operating_cash_flow / revenue | Higher is better, consistency matters |

**Crypto — On-Chain Fundamental Score (see §2.9 for full metric list):**

Composite on-chain score from network activity, supply dynamics, mining health, and ecosystem metrics. Each metric generates a sub-signal (bullish/neutral/bearish) with configurable weights. Traditional financial metrics (income statement, balance sheet, cash flow) return N/A for crypto — the on-chain score IS the fundamental score for this asset class.

**Non-equity, non-crypto assets:** Fundamental score returns 100% (not applicable). Technical and sentiment scores drive allocation.

**Update frequency:** Equity fundamentals: quarterly, synced to financial_data_availability. Crypto on-chain: daily. Fundamental score only recalculates when expected data for a period is ingested.

### 5.3 Sentiment Scoring Engine

**Sources:**
- Reddit: r/IndianStreetBets, r/IndiaInvestments, r/IndianStockMarket (via praw)
- Twitter/X: cashtags and ticker mentions (via snscrape or API)
- Crypto-specific: r/CryptoCurrency, r/Bitcoin, r/ethereum, Crypto Twitter

**Processing:**
- FinBERT (ProsusAI/finbert, free via HuggingFace) from day one — transformer-based NLP for all sentiment analysis
- Fine-tuning on Indian market social media text if pre-trained model underperforms on local content (Indian English, Hindi-English code-mixing, slang)

**Scoring (0-100):** Strong bullish with volume: 80-100. Mild bullish: 60-80. Neutral/low mentions: 40-60. Bearish: 20-40. Strong bearish: 0-20.

**Contrarian flag:** Extremely high bullish sentiment + very high mention volume = potential crowded trade. System flags this as a caution signal rather than amplifying the bullish score.

### 5.4 Combined Scoring

```
adjusted_technical = technical_score × regime_confidence
combined = (adjusted_technical × w_tech) + (fundamental_score × w_fund) + (sentiment_score × w_sent)
```

Default weights: Technical 50%, Fundamental 30%, Sentiment 20%. Configurable per user via weight_technical, weight_fundamental, weight_sentiment.

### 5.5 Institutional Flow Signals

Status: rules decided, design open, not implemented.

- FII and DII net flow are **always combined, never used alone**.
- The combined signal **activates only after 3 consecutive days with z-score > 1.5**.
- Bulk and block deals are **confirmation-only**: they never activate a signal on their own.
- Shareholding change is a **slow, independent signal**.
- SEBI LODR shareholding filings lag **21 days** (not 45). This is a separate lag from the financial-results lags in §5.2.

To be designed: data sources, the z-score definition and window, signal weights, and how the signal enters the combined score (§5.4). None of these is decided.

---

## 6. The Unified Control Loop

This is the heart of the system. It runs every `scoring_interval_minutes` (per asset class) during market hours.

```
EVERY scoring_interval_minutes:

  1. INGEST
     Pull latest intraday prices for all assets in this asset class

  2. SCORE
     For each asset:
       - Compute technical indicators on latest intraday candles
       - Compute technical confluence score (0-100)
       - Compute regime confidence (Hurst, Lyapunov, LSTM when available)
       - Retrieve latest fundamental score (static until quarterly update / daily for crypto on-chain)
       - Retrieve latest sentiment score (updated separately on its own schedule)
       - Compute combined_score = weighted composite × regime adjustment

  3. ALLOCATE (Three-Layer Portfolio Construction — see §2.7)
     Across ALL asset classes (not just the one being scored):
       - Skip any asset class where user.is_enabled = FALSE (target = 0%)
       - Total capital = sum(position_values) + wallet_cash - emergency_reserve

       Layer 1 — Black-Litterman target weights:
       - Combined scores become "views" (beliefs about outperformance)
       - Covariance matrix computed from price history
       - BL model outputs optimal weights balancing views vs market equilibrium
       - Apply position limits + sector limits + max_class_allocation_pct constraints
       - Apply ESG/ethical filters (excluded sectors, tickers, shariah compliance)

       Layer 2 — Regime-dependent blending:
       - Compute risk_parity_weights (inverse-volatility, no views needed)
       - Blend: effective_target = regime_confidence × BL_weights
                                 + (1 - regime_confidence) × risk_parity_weights

       Output: ideal_allocation = {asset_id: target_pct, cash: remaining_pct}

  4. CONTROL (PID — Layer 3, execution speed)
     For each asset in (current_portfolio ∪ ideal_allocation):
       — Look up Kp, Ki, Kd from user_asset_class_preferences
       error(t) = target_weight − current_weight
       P = Kp × error(t)
       I = Ki × Σ error(t-n : t)
       D = Kd × (error(t) − error(t-1))
       control_signal = (P + I + D) × regime_confidence    ← adaptive gain

  5. TAX FILTER (Layer 4 — before execution)
     For each asset with |control_signal| > base_threshold:
       a. Compute tax_drag_estimate for this specific trade:
          - Identify which lots would be sold (lot selection algorithm)
          - Compute applicable tax rate per lot (STCG/LTCG/flat)
          - Estimate tax cost of trade
          - Add transaction costs (STT, stamp duty, brokerage, GST)
       b. Compute effective_threshold = base_threshold + tax_drag + txn_costs
       c. Check tax patience window:
          - If selling position with lots approaching 365-day cliff (within tax_patience_window_days)
          - AND signal is not catastrophic (not a hard exit trigger)
          - AND expected tax savings > expected drawdown risk
          - → DEFER the trade, accumulate PID integral error
       d. Check loss harvesting opportunity:
          - If position has unrealized loss
          - AND no strong bullish signal for this asset
          - AND loss_harvesting_enabled for this asset class
          - AND asset class allows loss offset (NOT crypto)
          - → HARVEST the loss (sell, optionally immediate re-entry)
       e. Check LTCG exemption opportunity (near fiscal year end):
          - If user has remaining LTCG exemption headroom
          - AND position has LTCG-eligible lots with unrealized gains
          - → Realize gains up to exemption limit (free rebalancing)
       f. Apply lot selection for approved sells:
          - Loss lots first → LTCG lots → STCG lots (largest gain last)

       if |control_signal| > effective_threshold AND not deferred:
         → Compute trade size (proportional to |control_signal|)
         → Execute paper trade with optimal lot selection
         → Update portfolio, wallet, trailing stops
         → Update tax ledger (realized gains/losses, exemption usage)
         → Log trade with signal_reasons + tax_reasons
         → Notify user

  6. MONITOR (Hard Exits — separate from PID, override tax patience)
     For each held position:
       - Check trailing stop-loss against current price
       - Check if price broke key support with volume
       - Check if combined_score dropped catastrophically (below 20)
       If any trigger → immediate full sell → cash to wallet → notify user
       → "Sold X because Y. ₹Z returned to wallet. Tax impact: ₹T. Reinvest or hold?"
```

### Timing per Asset Class

| Asset Class | Scoring Interval | Market Hours | Implementation Priority |
|---|---|---|---|
| Indian Equities (NIFTY50) | 15-30 minutes | 9:15 AM - 3:30 PM IST | Priority 1 |
| US Equities | 15-30 minutes | 9:30 AM - 4:00 PM ET | Priority 3 |
| Crypto | 1-5 minutes | 24/7 | Priority 2 |
| Commodities (MCX) | 30-60 minutes | 9:00 AM - 11:30 PM IST | Priority 4 |

### Cross-Asset Rebalancing

Step 3 (ALLOCATE) runs across all asset classes simultaneously. Even though crypto scores every 5 minutes and equities every 30 minutes, the ideal allocation considers the entire portfolio. If crypto becomes very attractive, the system may suggest reducing equity exposure to free up capital — that's the pigeonhole principle in action.

**Top-level cross-asset-class allocation layer.** A layer above the per-asset-class control loops decides the allocation across asset classes. It is not one of the per-class loops, and it runs daily or when the regime changes. How it splits capital between classes and how the per-class loops consume its output is to be designed; the Step 3 description above predates it.

---

## 7. Project Structure

Layout as of 2026-10-19. Modules marked "(planned)" do not exist yet.

```
investbot/                           # repo root
├── CLAUDE.md                        # project instructions; wins over this plan where they conflict
├── README.md
├── main.py                          # CLI: setup | ingest | run | status | gaps | dq | score | evaluate
├── requirements.txt
├── pytest.ini
├── .env.example                     # settings come from .env (never committed)
│
├── investbot/
│   ├── config/
│   │   ├── settings.py              # .env settings, DB path, defaults
│   │   ├── tickers.py               # NIFTY50 ticker list + metadata, RETIRED_TICKERS
│   │   ├── tax_regimes.py           # Tax rate tables per country/asset class
│   │   ├── scoring_weights.py       # Confluence family weights
│   │   ├── index_history/
│   │   │   └── nifty50_changes.csv  # Sourced NIFTY50 change list (point-in-time membership)
│   │   └── risk_profiles.py         # (planned) PID gain presets per risk level
│   │
│   ├── data/
│   │   ├── db.py                    # SQLite connection, session management
│   │   ├── models.py                # SQLAlchemy ORM (tables from §4)
│   │   ├── seed.py                  # Asset classes, exchanges, assets, tax regimes, index membership
│   │   ├── quality.py               # Data-quality audit (python main.py dq)
│   │   ├── corporate_actions.py     # Dividends and splits by ex-date
│   │   ├── index_membership.py      # As-of index constituent queries
│   │   ├── ingestion/
│   │   │   ├── prices.py            # yfinance daily + intraday pulls
│   │   │   ├── financials.py        # yfinance financial statements
│   │   │   ├── onchain.py           # (planned) Crypto on-chain metrics (Glassnode/CoinGecko)
│   │   │   └── sentiment_ingest.py  # (planned) Reddit (praw) + Twitter pulls
│   │   └── migrations/              # Schema version tracking
│   │
│   ├── scoring/
│   │   ├── technical/
│   │   │   ├── indicators.py        # Public API (add_all_indicators) over _talib_core.py + _extensions.py
│   │   │   ├── _talib_core.py       # TA-Lib wrapper
│   │   │   ├── _extensions.py       # Indicators TA-Lib lacks (Ichimoku without Chikou, Supertrend, Keltner/squeeze, ...)
│   │   │   ├── candle_patterns.py   # TA-Lib CDL* patterns, normalised
│   │   │   ├── support_resistance.py # Swing pivots, volume profile, Fibonacci
│   │   │   ├── regime.py            # Hurst, Lyapunov, trend/range/chaotic labels, regime_confidence
│   │   │   ├── multi_timeframe.py   # 15m / 1h / daily alignment
│   │   │   ├── confluence.py        # Family-normalised confluence scoring
│   │   │   └── scorer.py            # score(ticker): caching and DB I/O (technical_scores)
│   │   ├── fundamental/
│   │   │   ├── income_statement.py  # Buffett income statement checklist (composite is a 100% pass-through)
│   │   │   ├── balance_sheet.py     # (planned) Balance sheet scoring
│   │   │   ├── cash_flow.py         # (planned) Cash flow scoring
│   │   │   ├── onchain_scorer.py    # (planned) Crypto on-chain fundamental scoring
│   │   │   └── scorer.py            # (planned) Composite fundamental score (dispatches by asset class)
│   │   ├── sentiment/               # Package stub only
│   │   │   ├── finbert_analyzer.py  # (planned) FinBERT (ProsusAI/finbert) scoring + aggregation
│   │   │   └── scorer.py            # (planned) Daily sentiment score per asset
│   │   └── combined.py              # (planned) Weighted composite + regime adjustment
│   │
│   ├── evaluation/                  # Walk-forward forward-return harness (§9.4)
│   │   ├── loader.py                # The only DB access: universe, bars, corporate actions
│   │   ├── panel.py                 # (date, ticker) features + labels, point-in-time universe
│   │   ├── labels.py                # Next-open price and total-return labels
│   │   ├── baselines.py             # Momentum, 12-1 momentum, reversal, low-vol
│   │   ├── walkforward.py           # Expanding-train / half-year test blocks with embargo
│   │   ├── stats.py                 # Newey-West, BH q-values, Spearman
│   │   ├── metrics.py               # IC, quintiles, turnover, costs, held portfolio
│   │   ├── variants.py              # Pre-registered polarity variants
│   │   ├── evaluate.py              # Signal x horizon x subset metrics table
│   │   ├── regime_study.py          # Regime label vs ex-post regime
│   │   └── report.py                # Markdown report + CSV
│   │
│   ├── ml/                          # Package stub only
│   │   ├── feature_pipeline.py      # (planned) TA-Lib features -> PCA -> feature matrix
│   │   ├── confluence_model.py      # (planned) XGBoost/LightGBM: PCA features -> forward returns
│   │   ├── regime_lstm.py           # (planned) LSTM/GRU regime transition prediction
│   │   ├── finbert_finetune.py      # (planned) Fine-tune FinBERT on Indian market text
│   │   └── training.py              # (planned) Training scripts, uses backtest data as training set
│   │
│   ├── trading/
│   │   ├── lot_tracker.py           # Lot-level accounting
│   │   ├── tax_ledger.py            # Fiscal year tax budget tracking
│   │   ├── tax_rules.py             # Regime parameters, fiscal year, holding-period rule
│   │   ├── transaction_costs.py     # STT, stamp duty, exchange charges, SEBI fees, GST
│   │   ├── risk.py                  # Covariance, risk-parity weights, position/sector caps
│   │   ├── allocator.py             # (planned) Black-Litterman + risk parity blending
│   │   ├── pid_controller.py        # (planned) PID rebalancing loop
│   │   ├── tax_optimizer.py         # (planned) Tax filter layer (threshold adjustment, patience, harvesting)
│   │   ├── exit_monitor.py          # (planned) Trailing SL, support break, catastrophic drop
│   │   ├── paper_engine.py          # (planned) Execute paper trades, wallet management
│   │   └── performance.py           # (planned) Win rate, Sharpe, drawdown per user (pre- and post-tax)
│   │
│   ├── dashboard/                   # Package stub only
│   │   └── app.py                   # (planned) Streamlit dashboard
│   │
│   ├── backtest/                    # Package stub only
│   │   ├── runner.py                # (planned) Historical replay of full pipeline
│   │   ├── metrics.py               # (planned) Full KPI framework (see §9)
│   │   ├── parameter_sweep.py       # (planned) Multi-profile backtesting
│   │   └── report.py                # (planned) Backtest results + ML training data
│   │
│   └── scheduler/
│       └── jobs.py                  # APScheduler: per-asset-class intervals, post-close pipeline
│
├── scripts/
│   ├── smoke_indicators.py          # End-to-end indicator smoke test against the real DB
│   └── regime_study.py              # python -m scripts.regime_study
│
├── tests/                           # pytest, no network or DB needed
├── tools/agents/                    # Autonomous-agent task inbox and runner
└── docs/
    ├── investbot_master_plan_v3.md  # This plan (v4.0 content)
    ├── BACKLOG.md
    ├── CHANGELOG.md
    ├── DECISIONS_PENDING.md
    └── reports/                     # Evaluation and regime-study reports
```

The database lives outside the repo at `~/investbot_data/investbot.db` and is never committed.

---

## 8. Tech Stack

| Component | Technology | Why |
|---|---|---|
| Language | Python 3.11+ | Natural fit for data/ML work, single language |
| Backend | FastAPI | Async, fast, serves bot + future API/PWA |
| Database | SQLite (initially) → PostgreSQL | Zero infra, fits 8GB RAM comfortably |
| ORM | SQLAlchemy | Schema management, easy migration path |
| Scheduler | APScheduler | In-process, no extra infra needed |
| Technical Analysis | TA-Lib + our `_extensions.py` | Industry-standard C implementation, actively maintained; extensions cover what TA-Lib lacks |
| Regime Detection | Custom (numpy/scipy) | Hurst exponent, Lyapunov — no library needed |
| ML — Feature Engineering | scikit-learn (PCA), XGBoost/LightGBM | Dimensionality reduction + nonlinear scoring |
| ML — Sentiment | FinBERT (ProsusAI/finbert, HuggingFace) | Financial domain transformer, free |
| ML — Regime | PyTorch (LSTM/GRU) | Regime transition prediction |
| Data Source (prices) | yfinance | Free, covers NSE, 15m intraday for last 60 days |
| Data Source (on-chain) | Glassnode (freemium), CoinGecko (free) | Crypto fundamental data |
| Reddit API | praw | Official Reddit API wrapper |
| Dashboard | Streamlit | Instant local UI, zero frontend work, live refresh |
| Backtesting | Custom | Need to test full pipeline including PID + tax, not just signals |

---

## 9. Backtesting KPI Framework

### 9.1 Metrics

**Return Metrics:**
- Total return (gross and net of tax + transaction costs)
- Annualized return (CAGR)
- Monthly return distribution (mean, median, skew, kurtosis)
- Alpha vs benchmark (NIFTY50 equal-weight buy-and-hold)
- Tax alpha (return improvement from tax optimization vs naive tax-unaware version)
- Transaction cost drag (total costs as % of average portfolio value)

**Risk Metrics:**
- Max drawdown (peak-to-trough)
- Max drawdown duration (days to recover)
- Sharpe ratio (risk-free rate = India 91-day T-bill)
- Sortino ratio (downside deviation only)
- Calmar ratio (annualized return / max drawdown)
- Value at Risk (95th percentile daily loss)
- Conditional VaR / Expected Shortfall

**Trade Quality Metrics:**
- Win rate (% of trades that were profitable)
- Average win / average loss ratio
- Profit factor (gross profits / gross losses)
- Average holding period
- Trades per month
- Turnover rate (annual portfolio turnover)

**Regime Adaptation Metrics:**
- Return in trending regimes vs random regimes
- Drawdown in chaotic regimes (should be lower than buy-and-hold)
- Regime detection accuracy (ex-post Hurst vs system's real-time estimate)
- PID gain behavior during regime transitions

**Portfolio Construction Metrics:**
- Average position count
- Concentration (Herfindahl index)
- Sector exposure over time
- Factor exposure (momentum, value, quality)
- Black-Litterman vs risk parity weight comparison

**Tax Optimization Metrics:**
- % of gains realized as LTCG vs STCG
- Tax-loss harvest value captured
- LTCG exemption utilization rate
- Average effective tax rate on realized gains
- Tax patience window hit rate (how often deferral was correct)
- After-tax return improvement vs tax-naive baseline

### 9.2 Backtest Parameter Sweep

**Primary sweep dimensions (test all combinations):**

| Dimension | Values | Count |
|---|---|---|
| overall_risk_appetite | 20, 40, 60, 80 | 4 |
| tax_optimization_aggressiveness | 0 (none), 50 (moderate), 100 (maximum) | 3 |
| initial_capital | ₹5L, ₹25L, ₹1Cr | 3 |

**Total primary runs: 36**

**Secondary sweep (test selectively on best primary configs):**

| Dimension | Values |
|---|---|
| Fiscal year start offset | April, July, October, January |
| Scoring weight tilt | Tech-heavy (60/20/20), Balanced (50/30/20), Fund-heavy (30/50/20) |
| Drawdown tolerance | 10%, 20%, 30% |

**Go/No-Go Thresholds:**

| Metric | Minimum Acceptable | Target |
|---|---|---|
| After-tax CAGR vs buy-and-hold NIFTY50 | +2% | +5% |
| Max drawdown | <25% (moderate profile) | <20% |
| Sharpe ratio | >0.8 | >1.2 |
| Win rate | >45% | >55% |
| Tax alpha (vs tax-naive) | >0.5% annual | >1.5% |

**Diagnostic Decision Framework:**

If a metric fails, trace back to the responsible phase:
- Low alpha → Phase 2 (technical scoring) or Phase 3 (fundamental scoring) — scores aren't predictive
- High drawdown → Phase 5c (PID gains too aggressive) or §2.3 (regime detection failing)
- Poor win rate → Phase 5b (allocation engine) — position sizing or entry timing
- High tax drag → Phase 5 tax filter — thresholds too permissive, patience window too short
- High turnover → PID integral gain too low, threshold too sensitive

### 9.3 Streamlit Dashboard Pages

- **Scores:** Live NIFTY50 rankings with combined scores, sub-scores, signal reasoning. Sortable, filterable.
- **Portfolio:** Current holdings per user, unrealized P&L, trailing SL levels, allocation pie chart, lot-level detail with tax status.
- **Trades:** Full trade history with timestamps, scores at time of trade, realized P&L, tax impact per trade.
- **Tax:** Fiscal year ledger, LTCG exemption tracker, carried-forward losses, projected year-end tax, loss harvesting opportunities.
- **Backtest:** Equity curve, drawdown chart, monthly returns heatmap, Sharpe/Sortino, parameter sweep comparison, pre- and post-tax returns.
- **Settings:** User profile — risk appetite sliders (0-100 per asset class), enable/disable asset classes, scoring weights, tax preferences, ethical filters.
- **Invest:** Input an amount, see the suggested allocation with tax implications, execute paper trades.

### 9.4 Evaluation Before Tuning: Walk-Forward Forward-Return Harness

TECH_SCORE correlated 0.82 with the past 10-day return and about -0.02 with the next 10 days (RELIANCE, daily): it described the past and did not yet predict. So the forward-return harness comes before more scoring layers, and **weight, threshold and polarity changes need harness evidence, not intuition** (Decided 2026-10-18 "Evaluate before tuning"). It is a signal-level check; the full-pipeline backtest of Phase 7 comes after it. After the harness, the order is: whatever its first report says is most promising, then the batch-scoring path that backtests need, then sentiment and allocation.

What exists (`investbot/evaluation/`, run with `python main.py evaluate [--universe index|active]`):

- **Walk-forward, nothing fitted:** expanding train window, half-year test blocks and an embargo before each block; every signal is evaluated with its designed polarity.
- **Next-open labels:** the signal is known after the close of day t and entry is at the next open, at several horizons. **Total-return labels** (cash dividends on their ex-dates from `corporate_actions`) are the primary label, with price-only rows alongside.
- **Point-in-time NIFTY50 universe:** a stock is scored only on dates when it was an index member (`index_membership`); `--universe active` gives today's constituents for comparison ("Universe effect" table).
- **Metrics:** daily cross-sectional rank IC with Newey-West t-statistics, quintile / top-bucket portfolios, turnover, a cost check and a held (open-to-open) portfolio, regime and year splits, against baselines (momentum, 12-1 momentum, reversal, low-vol).
- **Multiple testing:** Benjamini-Hochberg q-values over every signal x horizon test.
- **Pre-registered variants:** alternatives (currently the confluence polarity variants OSC_REVERT, OSC_MOMENTUM and REGIME_POLARITY) are specified before they are looked at and enter the same BH family.
- **Regime study:** `python -m scripts.regime_study` tests the real-time regime label against the ex-post regime with pre-registered hypotheses, on the same universe and splits.

Reports in `docs/reports/`:

- [`eval_20261018.md`](reports/eval_20261018.md): first harness report (today's constituents), [`eval_20261018_b.md`](reports/eval_20261018_b.md): with the held portfolio, [`eval_20261018_c.md`](reports/eval_20261018_c.md): point-in-time universe
- [`eval_20261019.md`](reports/eval_20261019.md): pre-registered polarity variants
- [`regime_study_20261018.md`](reports/regime_study_20261018.md) and [`regime_study_20261019.md`](reports/regime_study_20261019.md) (point-in-time universe): regime label vs the ex-post regime

What these reports mean for weights, polarity and the regime layer is decided by Ankit and tracked in `docs/DECISIONS_PENDING.md`, not in this plan.

---

## 10. Phase Plan

### Phase 1 — Data Pipeline & Storage (~1 week)

**Objective:** Populated database with NIFTY50 daily + intraday prices, available financials, and schema for all future data. Scheduled refresh running.

**Tasks:**
1. Project structure, venv, requirements.txt
2. SQLAlchemy models for ALL tables (including tax, lot tracking, on-chain — schema-complete from day one)
3. Seed database: asset_classes (equity, crypto), exchanges (NSE, BINANCE), assets (NIFTY50 tickers; names that left the index are kept with `is_active = FALSE`, so price history is kept but the allocator and scorer stop considering them)
4. Price ingestion: 2+ years daily OHLCV + 60 days of 15-minute intraday via yfinance
5. Financial ingestion: available annual/quarterly financials via yfinance
6. Populate financial_data_availability
7. Seed tax_regimes table with current Indian tax rates (equity + crypto)
8. APScheduler: intraday price refresh every 15-30 min during market hours, daily candle update after close, quarterly financial check
9. Error handling: retry logic, data validation, gap detection

**Deliverable:** `python main.py` → database populated with all tables created, scheduler running. Can query any NIFTY50 stock's prices or financials.

---

### Phase 2 — Technical Scoring Engine (~2-3 weeks)

**Objective:** Given any asset with price data, produce a technical score (0-100) with signal reasoning, on both daily and intraday timeframes.

**Tasks:**
1. TA-Lib integration: compute the implemented indicators from the §5.1 list (`indicators.py` over `_talib_core.py` + `_extensions.py`)
2. Confluence scoring: each indicator → sub-signal → weighted sum → normalized score
3. Volume-weighted support/resistance zone detection
4. Fibonacci retracement level computation
5. Candlestick pattern detection (all patterns listed in §5.1)
6. Multi-timeframe module: 15m + 1h + daily alignment, weighted 1/2/3
7. Regime detection: rolling Hurst exponent by DFA-1 (windows 100 and 250 bars, `HURST_WINDOWS`; the regime label reads the 100-bar window, `REGIME_WINDOW = 100`) and a rolling largest Lyapunov exponent (Rosenstein estimate, same windows)
8. Technical scorer: price_data → (technical_score, regime_confidence, signal_reasons)
9. Cache indicators in indicator_cache table
10. Unit tests against known indicator values
11. Historical calibration: run on 2 years of data, verify score distributions

**Deliverable:** `technical_scorer.score("RELIANCE.NS")` → score, regime confidence, signal breakdown.

---

### Phase 3 — Fundamental Scoring Engine (~1-2 weeks)

**Status: paused** until fundamentals are point-in-time (§5.2); the evaluation harness precondition is met (§9.4). Every fundamental input counts only from the filing's publication date, or conservatively from period end + 45 days (quarterly) / + 60 days (annual), per SEBI LODR Reg. 33; yfinance gives period ends, not filing dates. Balance-sheet and cash-flow rules also wait for Ankit's notes.

**Objective:** Income statement scoring implemented per Buffett checklist. Balance sheet and cash flow sub-scores stubbed (return 100%) until book notes completed. Composite fundamental score returns 100% pass-through initially; income statement sub-scores computed and stored for analysis.

**Tasks:**
1. Implement all income statement sub-scores from §5.2 table
2. Stub balance_sheet.py and cash_flow.py → return 100% (columns populated, scoring TBD)
3. Composite scorer with configurable sub-score weights
4. Synced update logic: only recalculate when financial_data_availability shows new data
5. Store in scores table — sub-scores visible for inspection even while composite is pass-through
6. Crypto on-chain scorer: stub → returns 100% (data ingestion pipeline to be built when crypto asset class is enabled)

**Deliverable:** `fundamental_scorer.score("RELIANCE.NS")` → 100% composite (initially), but income statement sub-scores visible for inspection.

---

### Phase 4 — Sentiment Scoring Engine (~1-2 weeks)

**Objective:** Daily sentiment pulse per NIFTY50 stock from Reddit and Twitter using FinBERT.

**Tasks:**
1. Reddit API setup (praw), subreddit configuration
2. Reddit scraper: pull posts/comments, match NIFTY50 tickers
3. Twitter/X collection (snscrape or API)
4. FinBERT integration: ProsusAI/finbert model loaded, batch inference pipeline
5. Aggregate to daily per-asset per-source scores
6. Contrarian signal flagging (extreme bullish + high volume)
7. Schedule: pulls during market hours, daily aggregation

**Deliverable:** `sentiment_scorer.score("RELIANCE.NS", date)` → sentiment score + mention volume + contrarian flag.

---

### Phase 5 — Unified Control Loop + Paper Trading + Tax Engine (~3-4 weeks)

**Sub-phase 5a — Combined Scoring (3-4 days):**
1. Weighted composite with regime adjustment
2. Daily + intraday combined scores for all NIFTY50

**Sub-phase 5b — Allocation Engine (3-4 days):**
3. Black-Litterman model: combined scores as views + covariance matrix → target weights
4. Risk parity weights: inverse-volatility baseline
5. Regime-dependent blending
6. Position limits + sector limits + max class allocation constraints
7. ESG/ethical filters (excluded sectors, tickers)

**Sub-phase 5c — PID Controller (3-4 days):**
8. PID loop: error computation, P/I/D terms, control signal
9. Risk appetite (0-100 per asset class) → PID gain mapping
10. Adaptive gain scheduling via regime_confidence
11. Rebalance threshold logic
12. Asset class disable logic: is_enabled=FALSE → force target to 0% → liquidate

**Sub-phase 5d — Tax Optimization Engine (4-5 days):**
13. Lot tracker: create lots on buy, select optimal lots on sell (§2.8 4e)
14. Tax-adjusted threshold computation (§2.8 4a)
15. Tax patience window logic (§2.8 4b)
16. Tax-loss harvesting engine (§2.8 4c)
17. LTCG exemption optimizer (§2.8 4d)
18. Fiscal year tax ledger maintenance (§2.8 4f)
19. Transaction cost calculator (STT, stamp duty, exchange charges, SEBI fees, GST)

**Sub-phase 5e — Paper Trading Engine (3-4 days):**
20. Trade execution: update portfolios, create trade records with tax fields
21. Wallet management: cash flows from sells, deposits from user
22. Trailing stop-loss: update on new highs, check against intraday prices
23. Exit monitor: trailing SL, support break, catastrophic score drop (overrides tax patience)
24. Performance tracker: win rate, avg return, max drawdown, Sharpe per user, pre- and post-tax

**Deliverable:** Full automated loop running during market hours. User deposits money → system allocates → trades (tax-aware) → monitors → exits when needed → notifies.

---

### Phase 6 — Streamlit Dashboard (~1 week)

**Objective:** Local dashboard for monitoring and interacting with the system during development and testing. All pages from §9.3 implemented.

**Deliverable:** `streamlit run dashboard/app.py` → full visibility into scoring, portfolio, trades, tax, and backtest results.

---

### Phase 7 — Backtesting (~2-3 weeks)

**Objective:** Validate the full pipeline (including tax optimization) against historical data. Find actual baseline confidence. Generate ML training dataset.

**Tasks:**
1. Replay engine: feed historical data tick-by-tick through the unified control loop (including tax filter)
2. Run full scoring + PID + tax filter + trading simulation
3. Parameter sweep: 36 primary configurations (§9.2)
4. Compute all KPIs from §9.1
5. Compare across risk profiles × tax aggressiveness × capital levels
6. Measure tax alpha: compare tax-aware vs tax-naive versions
7. Benchmark against buy-and-hold NIFTY50 (equal-weight)
8. Generate backtest report with go/no-go assessment
9. Export ML training dataset: every timestamp's full feature values (285 columns: 112 indicators, 125 candle-pattern, 26 support/resistance, 13 regime and 9 confluence columns, of which `TECH_REASONS` is text; counted from `compute_daily_frame` output minus the 6 OHLCV pass-through columns) paired with actual forward returns
10. Diagnostic pass: for any failing metric, trace to responsible phase using decision framework (§9.2)

**Deliverable:** Report showing strategy performance vs benchmark, per risk profile, with confidence metrics. Plus: labeled dataset ready for ML model training. This determines whether we proceed to real money.

**Target:** 3 months of paper trading track record before considering real money.

---

### Phase 8 — ML Scoring Upgrades

Train PCA + XGBoost/LightGBM on backtest dataset to replace hand-tuned confluence weights. Compare ML-scored backtest vs rule-based backtest — only deploy if measurably better. Begin accumulating intraday archive for LSTM regime detection training. Fine-tune FinBERT on accumulated Indian market social media text if pre-trained model shows underperformance.

### Phase 9 — Telegram Bot + Push Notifications

Once algorithm is validated, add Telegram as push notification channel for trade alerts, exit signals, tax events, and reinvestment prompts.

### Phase 10 — PWA Migration

React + Tailwind frontend, Groww-style UI, TradingView charts, web push notifications. Telegram remains as notification channel option.

### Phase 11 — Multi-Exchange Expansion

BSE, NYSE/NASDAQ, Binance, MCX. Schema already supports it. Technical scoring works unchanged. Sentiment sources expand per market. User preferences per asset class already built. Tax regimes per country/asset class already in schema.

### Phase 12 — Real Money Trading

Zerodha Kite Connect or Angel One SmartAPI. Graduated: 10% real, 90% paper. Hard safeguards: max daily loss, kill switch. Only after backtest + 3-month paper track record.

### Phase 13 — LSTM Regime Detection

Once sufficient intraday archive accumulated (6+ months), train LSTM/GRU on market microstructure sequences to predict regime transitions ahead of formula-based detection.

### Phase 14 — Full Fundamental Scoring

Complete Buffett analysis with balance sheet and cash flow statement notes. Screener.in scraping for full 10-year history. Implement all sub-scores from §5.2 balance sheet and cash flow tables.

### Phase 15 — Crypto On-Chain Integration

Connect Glassnode/CoinGecko APIs, populate onchain_metrics table, implement on-chain fundamental scorer. Different tax behavior (30% flat, no loss offset) already encoded in tax_regimes.

---

## 11. Timeline

| Phase | Duration | Cumulative |
|---|---|---|
| Phase 1: Data Pipeline | ~1 week | Week 1 |
| Phase 2: Technical Scoring | ~2-3 weeks | Week 4 |
| Phase 3: Fundamental Scoring | ~1-2 weeks | Week 6 |
| Phase 4: Sentiment Scoring | ~1-2 weeks | Week 8 |
| Phase 5: Control Loop + Paper Trading + Tax | ~3-4 weeks | Week 12 |
| Phase 6: Streamlit Dashboard | ~1 week | Week 13 |
| Phase 7: Backtesting | ~2-3 weeks | Week 16 |

**Total: ~14-16 weeks to functional system with tax optimization**

Then: 3 months of paper trading validation before real money.

---

## 12. Hardware

- **RAM:** 8GB sufficient. SQLite + 50 stocks of intraday data + pandas + FinBERT inference fits.
- **Storage:** < 2GB for all data including 60 days of 15-min candles.
- **Compute:** Scoring 50 stocks every 15 minutes takes seconds. FinBERT inference is the heaviest operation — batch processing keeps it manageable.
- **Network:** Required for yfinance, Reddit API, HuggingFace model download.
- **PS5 (1TB):** Available as supplementary compute/storage for backtest parameter sweeps if needed. Not required for core operations.
- **Backtesting compute:** 36 primary parameter sweep runs × ~5-10 min each = 3-6 hours total on laptop. Parallelizable across CPU cores. PS5 as overflow if expanding to secondary sweep dimensions.

---

## 13. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| yfinance unreliability | Stale/missing prices | Retry logic, aggressive caching, paid fallback (FMP) |
| yfinance intraday limits | Only 60 days of 15m data | Sufficient initially; accumulate own archive over time |
| Sentiment noise | Bad sentiment scores | FinBERT is more robust than rule-based; measure correlation, iterate |
| PID gain tuning | Poor rebalancing behavior | Empirical calibration via backtesting. Start conservative |
| Twitter/X API instability | Lost sentiment source | Reddit is primary. Twitter supplementary |
| Hurst/Lyapunov estimation sensitivity | Unreliable regime detection | Conservative defaults, test multiple window sizes |
| Fundamental data gaps (4-5 yrs not 10) | Incomplete Buffett analysis | Start with available data, supplement with screener.in later |
| Tax law changes | Invalidated tax optimization logic | Tax rates in tax_regimes table with effective_from/effective_to dates — update table, not code |
| FinBERT performance on Indian text | Poor sentiment accuracy | Fine-tuning pipeline built in; fallback to raw FinBERT scores |
| On-chain data API costs | Limited crypto fundamentals | Start with free tiers (CoinGecko, Glassnode basic), upgrade as needed |
| Backtest overfitting | Strategy works historically but fails live | Out-of-sample validation, walk-forward analysis, paper trading validation period |

---

## 14. Success Criteria

- [ ] Near-real-time scoring (15-30 min) of all NIFTY50 stocks during market hours
- [ ] Unified control loop running: score → allocate → PID → tax filter → paper trade → notify
- [ ] Tax-aware rebalancing: patience window, loss harvesting, lot optimization, LTCG exemption usage
- [ ] Streamlit dashboard functional: scores, portfolio, trades, tax ledger, backtest results
- [ ] Per-user performance tracking with real metrics (Sharpe, drawdown, win rate) — both pre- and post-tax
- [ ] Backtest showing measurable edge over equal-weight buy-and-hold NIFTY50 after tax and transaction costs
- [ ] Tax alpha of at least 0.5% annually vs tax-naive baseline
- [ ] 3 months paper trading before considering real money
- [ ] System adapts behavior to market regime (cautious in chaos, bold in trends)
- [ ] Different user risk profiles (per asset class) produce meaningfully different outcomes
- [ ] Full parameter sweep backtest across risk × tax × capital dimensions with diagnostic framework

---

## 15. Key Learnings & Design Principles

1. The investment decision problem is best framed as a single continuous optimization ("where should money be now?") rather than separate periodic and monitoring workflows
2. Control systems theory (PID loops) maps cleanly onto portfolio rebalancing — execution speed, not just target weights, needs to be governed
3. Transformer-based models (FinBERT) are preferred over statistical/bag-of-words approaches for all NLP tasks in the system
4. Data schema should be built completely upfront even if scoring logic is a stub — avoids costly data model rework later
5. Backtesting KPIs need to be defined before coding begins, with explicit thresholds and a diagnostic map back to phases — prevents ambiguous "did it work?" conclusions
6. Regulatory framing matters early: business model choices are constrained by SEBI category, so positioning and monetization strategy should inform architecture decisions
7. Tax-awareness isn't a separate system — it's a modification to the rebalance decision function that raises the bar for action based on after-tax economics
8. The single highest-value tax optimization is "don't trade unnecessarily" — everything else is second-order
9. India's lack of a wash sale rule makes tax-loss harvesting significantly more powerful than in US markets
10. Different asset classes (equity, crypto, commodities) have fundamentally different tax regimes that change optimal system behavior — this must be parameterized, not hardcoded
11. For taxable retail investors, a system that optimally rebalances on a pre-tax basis but ignores tax drag will underperform a simpler buy-and-hold strategy after tax

---

## 16. Business & Regulatory Path

**SaaS tool → SEBI-registered Investment Adviser → platform model**

Relevant SEBI categories explored:
- **Investment Adviser (IA):** AUM-based fee (up to 2.5% of AUM or ₹75,000 fixed per annum per client). Requires net worth ₹50L for individuals. Must pass NISM Investment Adviser certification.
- **Research Analyst (RA):** Can provide buy/sell recommendations. Subscription model allowed. Requires NISM RA certification.
- **Portfolio Manager (PMS):** Minimum ₹50L per client. Discretionary management. High barrier.

Revenue model options: AUM % fee, performance fee (IA structure), SaaS subscription (RA structure), broker revenue share partnership.

Regulatory framing informs architecture: if positioned as IA, system must track individual advice per client and maintain audit trail. If RA, can offer general recommendations. These requirements shape logging, notification, and trade history design — which are already built into the schema.
