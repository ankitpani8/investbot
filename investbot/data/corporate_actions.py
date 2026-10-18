"""
Corporate actions (dividends and splits) stored point-in-time by ex-date.

`load_actions` is the only function that touches the DB. `dividends_on_bars`
is pure: it maps stored actions onto a bar index for the total-return labels.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from investbot.data.models import Asset, CorporateAction

ACTION_COLUMNS = ["action_type", "value"]


def load_actions(session: Session, ticker: str, end: date | None = None) -> pd.DataFrame:
    """
    Stored actions for `ticker`, indexed by `ex_date` (ascending), columns
    `action_type` and `value`. Only rows with `ex_date <= end` when `end` is given.
    """
    stmt = (
        select(CorporateAction.ex_date, CorporateAction.action_type, CorporateAction.value)
        .join(Asset, Asset.id == CorporateAction.asset_id)
        .where(Asset.ticker == ticker)
        .order_by(CorporateAction.ex_date, CorporateAction.action_type)
    )
    if end is not None:
        stmt = stmt.where(CorporateAction.ex_date <= end)
    rows = session.execute(stmt).all()
    out = pd.DataFrame(
        {"action_type": [r[1] for r in rows], "value": [float(r[2]) for r in rows]},
        index=pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows], name="ex_date"),
        columns=ACTION_COLUMNS,
    )
    return out


def dividends_on_bars(bar_index: pd.DatetimeIndex, actions: pd.DataFrame) -> np.ndarray:
    """
    Dividend per share credited on each bar (0.0 where none).

    An ex-date without a stored bar goes to the first stored bar on or after it,
    ex-dates after the last bar are dropped, and several dividends on one bar are
    summed. The value at bar t depends only on actions with ex-date <= t's session
    and on bars up to t, so removing later bars changes nothing at or before t.
    """
    out = np.zeros(len(bar_index), dtype=float)
    if len(bar_index) == 0 or actions is None or len(actions) == 0:
        return out
    divs = actions[actions["action_type"] == "dividend"]
    if divs.empty:
        return out
    bars = pd.DatetimeIndex(bar_index).normalize()
    pos = bars.searchsorted(pd.DatetimeIndex(divs.index).normalize(), side="left")
    keep = pos < len(bars)
    np.add.at(out, pos[keep], divs["value"].to_numpy(dtype=float)[keep])
    return out
