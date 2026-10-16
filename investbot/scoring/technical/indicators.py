"""
Technical indicator computation — public API.

Pure compute module: takes an OHLCV DataFrame, returns the same DataFrame
with ~115 technical indicator columns appended. No DB side effects —
caching and DB I/O are orchestrated by scorer.py. No NaN filtering —
warmup rows are preserved so callers can choose their own cutoff.

Architecture:
    indicators.py           ← public API (this file)
        ├─ _talib_core.py   ← TA-Lib wrapper (~90 columns)
        └─ _extensions.py   ← custom indicators TA-Lib lacks (~25 columns)

Why TA-Lib as the engine:
    - 25+ years of production use; industry standard
    - C-based, 2-4x faster than pure-Python alternatives
    - Actively maintained (C core + Python wrapper both updated in 2026)
    - Binary wheels now available, installation friction largely eliminated
    - The problem domain is mature: core technical indicators are a solved
      problem, so low rate of change is a feature, not a bug

Column naming follows pandas-ta conventions where possible, so downstream
code that referenced those names continues to work unchanged. Names are
locked in here — downstream modules (confluence.py, ML pipeline, scorer.py)
depend on them.

Performance note: both core and extensions collect their output columns
into plain dicts, then this orchestrator assembles everything via a single
pd.concat call. This avoids pandas' DataFrame fragmentation penalty from
repeated single-column assignment (O(N²) in total columns, costly at ~115).

Requires:
    TA-Lib C library installed (brew/apt/conda) + ta-lib-python >= 0.5.0
"""
from __future__ import annotations

import logging

import pandas as pd

from investbot.scoring.technical._talib_core import compute_talib_columns
from investbot.scoring.technical._extensions import compute_extension_columns

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
OPTIONAL_COLUMNS: tuple[str, ...] = ("adj_close",)


# ──────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators on an OHLCV DataFrame.

    Parameters
    ----------
    df : DataFrame
        Must have columns ['open', 'high', 'low', 'close', 'volume']
        (lowercase) and a monotonic DatetimeIndex. Additional columns
        (e.g. 'adj_close') are preserved but ignored. Input is NOT
        mutated; a new DataFrame is returned.

    Returns
    -------
    DataFrame
        New DataFrame containing the original OHLCV plus ~115 indicator
        columns. Warmup rows (NaN values in rolling-window indicators)
        are preserved — callers handle NaN filtering per their needs.

    Notes
    -----
    Individual indicator failures are logged at DEBUG level and skipped
    silently. A missing column is the expected outcome — downstream
    scoring handles absence gracefully.

    Execution order matters: TA-Lib core runs first (produces BBANDS),
    then extensions (produces KC, then Squeeze which consumes both).
    Extensions receive the partial frame so they can read BBANDS.
    """
    _validate(df)

    before = df.shape[1]

    # Phase 1 — TA-Lib core produces its columns as a dict.
    core_cols = compute_talib_columns(df)

    # Materialize core columns so extensions can read BBANDS directly.
    # This single concat is cheap (one allocation) compared to O(N) inserts.
    partial = pd.concat([df, pd.DataFrame(core_cols, index=df.index)], axis=1, copy=False)

    # Phase 2 — Extensions read from the partial frame, produce their dict.
    ext_cols = compute_extension_columns(partial)

    # Final assembly — one last concat to add extension columns.
    result = pd.concat(
        [partial, pd.DataFrame(ext_cols, index=df.index)],
        axis=1,
        copy=False,
    )

    added = result.shape[1] - before
    logger.info(
        "Computed technical indicators: %d rows × %d total columns (%d added)",
        len(result), result.shape[1], added,
    )
    return result


def list_indicator_columns(df: pd.DataFrame) -> list[str]:
    """Return indicator columns added by add_all_indicators (non-OHLCV)."""
    skip = set(REQUIRED_COLUMNS) | set(OPTIONAL_COLUMNS)
    return [c for c in df.columns if c not in skip]


# ──────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────

def _validate(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"indicators.py expects a DatetimeIndex; got {type(df.index).__name__}"
        )
    if not df.index.is_monotonic_increasing:
        raise ValueError("Index must be monotonically increasing (sort_index first)")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")
    if len(df) < 30:
        logger.warning(
            "Input has only %d rows — most indicators will be NaN. "
            "Minimum ~200 rows recommended for full indicator stability.",
            len(df),
        )
