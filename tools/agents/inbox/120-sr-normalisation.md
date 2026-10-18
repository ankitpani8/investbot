---
id: 120-sr-normalisation
depends: []
---
# Support/resistance: normalised strength, ATR-based clustering, unclipped VP_DIST

Decided 2026-10-18 (docs/DECISIONS_PENDING.md).

## Requirements
In `investbot/scoring/technical/support_resistance.py`:
- `SR_STRENGTH_SUPPORT` / `SR_STRENGTH_RESIST` become a share in [0, 1]: the level's recency-weighted touches ÷ total recency weight of all pivots in the lookback at bar t.
- Cluster tolerance becomes ATR-based: a new constant `SR_CLUSTER_ATR_MULT = 0.25`, tolerance at bar t = 0.25 × ATR_14[t] (computed causally inside the module with TA-Lib if the column isn't supplied). Keep `SR_CLUSTER_PCT` as a fallback only while ATR is still in warmup, and document this.
- Add `VP_DIST`: unclipped signed distance of close from the value area, in ATR units (0 inside the area, positive above VAH, negative below VAL). Keep `VP_POS` unchanged (clipped).
- Pivots stay 3/3.
- Update the confluence column→family contract (VP_DIST is mapped to the S/R family or explicitly ignored; justify the choice in a comment). Bump `SCORER_VERSION`.

## Acceptance criteria
- Truncation invariance over all S/R columns (existing test extended to the new columns).
- A test where doubling price volatility widens the clusters (fewer, broader levels), and strength shares sum to ≤ 1.
- The zig-zag fixture tests still pass, adjusted for the new tolerance if needed; explain any expected-value changes in the commit message.
- The suite is green.
