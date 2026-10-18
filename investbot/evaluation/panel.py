"""
Panel builder: (date, ticker) feature panel and a separate forward-return label panel.

For each asset, one at a time, the full daily history (up to ``end``) is
loaded and the feature frame computed **once** with
``scorer.compute_daily_frame``. That frame is truncation-invariant (a value at
bar t uses bars <= t only, tested per module and in
``tests/test_evaluation.py``), so the full-history value at t equals what was
knowable at t's close. Only the evaluated columns are kept:

    TECH_SCORE, TECH_SCORE_<FAMILY>     confluence outputs
    SIG_<NAME>                          every confluence signal (``compute_signals``),
                                        prefixed because some names (VP_POS) are
                                        also upstream column names
    REGIME, REGIME_CONFIDENCE           conditioning variables, not signals
    BASE_*                              baseline factors (``baselines.py``)

Labels (``labels.py``) are computed from the same bars into a different frame.
``start`` only trims the output; features always warm up on the full history.
Bars after ``end`` are never loaded, so labels near ``end`` are NaN rather
than reaching past it.
"""
from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from investbot.config.scoring_weights import FAMILIES
from investbot.evaluation.baselines import compute_baselines
from investbot.evaluation.labels import HORIZONS, forward_returns
from investbot.scoring.technical.confluence import compute_signals
from investbot.scoring.technical.scorer import MIN_BARS, compute_daily_frame

logger = logging.getLogger(__name__)

SIGNAL_PREFIX = "SIG_"
SCORE_COLUMNS: tuple[str, ...] = ("TECH_SCORE", *(f"TECH_SCORE_{f.upper()}" for f in FAMILIES))
CONDITIONING_COLUMNS: tuple[str, ...] = ("REGIME", "REGIME_CONFIDENCE")


def asset_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Evaluated feature columns for one asset's daily OHLCV (pure; no dropna)."""
    frame = compute_daily_frame(bars)
    cols: dict[str, np.ndarray] = {c: frame[c].to_numpy(dtype=np.float64) for c in SCORE_COLUMNS}
    for name, values in compute_signals(frame).items():
        cols[SIGNAL_PREFIX + name] = values
    for c in CONDITIONING_COLUMNS:
        cols[c] = frame[c].to_numpy(dtype=np.float64)
    cols.update(compute_baselines(bars))
    return pd.DataFrame(cols, index=bars.index)


def asset_labels(bars: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    return pd.DataFrame(forward_returns(bars, horizons), index=bars.index)


def _stack(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"]))
    out = pd.concat(frames, names=["ticker", "date"]).swaplevel().sort_index()
    out.index = out.index.set_names(["date", "ticker"])
    return out


def build_dataset(
    assets: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    *,
    horizons=HORIZONS,
    session=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    (features, labels), both indexed by (date, ticker) and aligned row for row.

    ``assets`` defaults to every active stock. Assets with fewer than
    ``MIN_BARS`` bars are skipped (logged).
    """
    from investbot.evaluation.loader import active_tickers, load_daily_history

    own = session is None
    if own:
        from investbot.data.db import SessionLocal
        session = SessionLocal()
    feats: dict[str, pd.DataFrame] = {}
    labels: dict[str, pd.DataFrame] = {}
    try:
        tickers = list(assets) if assets is not None else active_tickers(session)
        for ticker in tickers:
            bars = load_daily_history(session, ticker, end)
            if len(bars) < MIN_BARS:
                logger.info("evaluation: skipping %s (%d bars)", ticker, len(bars))
                continue
            f = asset_features(bars)
            y = asset_labels(bars, horizons)
            if start is not None:
                keep = bars.index >= pd.Timestamp(start)
                f, y = f[keep], y[keep]
            feats[ticker] = f
            labels[ticker] = y
    finally:
        if own:
            session.close()
    return _stack(feats), _stack(labels)


def build_panel(assets: list[str] | None = None, start: date | None = None,
                end: date | None = None, **kw) -> pd.DataFrame:
    """Feature panel indexed by (date, ticker). Labels: see ``build_dataset``."""
    return build_dataset(assets, start, end, **kw)[0]


def signal_columns(features: pd.DataFrame) -> list[str]:
    """Columns to evaluate as signals: scores, SIG_*, baselines (not REGIME*)."""
    return [c for c in features.columns if c not in CONDITIONING_COLUMNS]
