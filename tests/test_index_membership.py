"""Point-in-time index membership (in-memory SQLite, no network)."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from investbot.config.tickers import NIFTY50_TICKERS
from investbot.data import index_membership as im
from investbot.data.models import Asset, Base, Exchange, IndexMembership, PriceDaily
from investbot.data.seed import run_seed

D = date


def _changes(rows):
    return pd.DataFrame(rows, columns=["effective_date", "action", "ticker", "source"])


# Toy 5-name index, today {A, C, D, E, G}. Swaps: X -> E (2020-06-01), B -> C (2021-03-01),
# D -> F (2022-01-03), then F -> D (2022-09-01): D leaves and comes back.
TOY_CURRENT = ["A", "C", "D", "E", "G"]
TOY_CHANGES = _changes([
    (D(2020, 6, 1), "add", "E", "s1"),
    (D(2020, 6, 1), "remove", "X", "s1"),
    (D(2021, 3, 1), "add", "C", "s2"),
    (D(2021, 3, 1), "remove", "B", "s2"),
    (D(2022, 1, 3), "add", "F", "s3"),
    (D(2022, 1, 3), "remove", "D", "s3"),
    (D(2022, 9, 1), "add", "D", "s4"),
    (D(2022, 9, 1), "remove", "F", "s4"),
])
TOY_START = D(2020, 1, 1)


def _toy():
    return im.reconstruct_intervals(TOY_CURRENT, TOY_CHANGES, D(2023, 1, 1), history_start=TOY_START)


def test_reconstruct_toy_intervals():
    iv = _toy().set_index(["ticker", "effective_from"])
    # open members
    assert iv.loc[("A", TOY_START), "effective_to"] is None and not iv.loc[("A", TOY_START), "complete"]
    assert iv.loc[("D", D(2022, 9, 1)), "effective_to"] is None
    # D was a member from the start until 2022-01-02 and came back on 2022-09-01
    assert iv.loc[("D", TOY_START), "effective_to"] == D(2022, 1, 2)
    assert not iv.loc[("D", TOY_START), "complete"]
    # F was a member 2022-01-03 .. 2022-08-31
    assert iv.loc[("F", D(2022, 1, 3)), "effective_to"] == D(2022, 8, 31)
    assert iv.loc[("F", D(2022, 1, 3)), "complete"]
    # B and X left, C and E joined
    assert iv.loc[("B", TOY_START), "effective_to"] == D(2021, 2, 28)
    assert iv.loc[("X", TOY_START), "effective_to"] == D(2020, 5, 31)
    assert iv.loc[("C", D(2021, 3, 1)), "effective_to"] is None
    assert iv.loc[("E", D(2020, 6, 1)), "source"] == "s1"
    assert iv.loc[("B", TOY_START), "source"] == "s2"  # its end is sourced by the removal


@pytest.mark.parametrize("d, expected", [
    (D(2020, 5, 31), {"A", "B", "D", "G", "X"}),                  # day before the first swap
    (D(2020, 6, 1), {"A", "B", "D", "E", "G"}),                   # change day itself
    (D(2021, 2, 26), {"A", "B", "D", "E", "G"}),
    (D(2021, 3, 1), {"A", "C", "D", "E", "G"}),
    (D(2022, 1, 2), {"A", "C", "D", "E", "G"}),                   # day before the D -> F swap
    (D(2022, 1, 3), {"A", "C", "E", "F", "G"}),
    (D(2022, 8, 31), {"A", "C", "E", "F", "G"}),
    (D(2022, 9, 1), {"A", "C", "D", "E", "G"}),
    (D(2030, 1, 1), {"A", "C", "D", "E", "G"}),
])
def test_members_as_of_boundaries(d, expected):
    assert im.members_as_of(_toy(), d) == expected


def test_membership_matrix_matches_members_as_of():
    iv = _toy()
    dates = pd.bdate_range("2020-05-25", "2022-09-10")
    tickers = ["A", "B", "C", "D", "E", "F", "G", "X", "NEVER"]
    m = im.membership_matrix(iv, dates, tickers)
    assert m.dtypes.eq(bool).all() and list(m.columns) == tickers
    assert not m["NEVER"].any()
    for d in dates[::17]:
        assert {t for t in tickers if m.loc[d, t]} == im.members_as_of(iv, d.date())


def test_members_as_of_uses_only_changes_up_to_d():
    """Membership on d equals a forward replay of the changes dated <= d from the initial set."""
    iv = _toy()
    initial = im.members_as_of(iv, TOY_START)
    for d in pd.bdate_range("2020-01-01", "2023-01-01"):
        state = set(initial)
        known = TOY_CHANGES[TOY_CHANGES["effective_date"] <= d.date()]
        for c in known.sort_values("effective_date").itertuples():
            state.add(c.ticker) if c.action == "add" else state.remove(c.ticker)
        assert im.members_as_of(iv, d.date()) == state, d
    # a stint that starts after d never leaks into d
    future = pd.concat([iv, pd.DataFrame([{"ticker": "Z", "effective_from": D(2030, 1, 1),
                                           "effective_to": None, "source": "", "complete": True}])])
    assert "Z" not in im.members_as_of(future, D(2029, 12, 31))
    assert "Z" in im.members_as_of(future, D(2030, 1, 1))


def test_reconstruct_rejects_contradicting_changes():
    bad_add = _changes([(D(2021, 1, 4), "add", "Q", "s")])  # Q is not in today's list
    with pytest.raises(ValueError, match="add Q"):
        im.reconstruct_intervals(["A"], bad_add, D(2022, 1, 1), history_start=TOY_START)
    bad_remove = _changes([(D(2021, 1, 4), "remove", "A", "s")])
    with pytest.raises(ValueError, match="remove A"):
        im.reconstruct_intervals(["A"], bad_remove, D(2022, 1, 1), history_start=TOY_START)


def test_changes_after_as_of_are_ignored():
    # as of 2021-06-01 the list is {A, C, D, E, G}; the 2022 swaps do not exist yet
    iv = im.reconstruct_intervals(["A", "C", "D", "E", "G"], TOY_CHANGES, D(2021, 6, 1), history_start=TOY_START)
    assert im.members_as_of(iv, D(2021, 6, 1)) == {"A", "C", "D", "E", "G"}
    assert iv["effective_from"].max() == D(2021, 3, 1)


def test_completeness_flag():
    assert not im.is_complete(im.COMPLETE_FROM - pd.Timedelta(days=1).to_pytimedelta())
    assert im.is_complete(im.COMPLETE_FROM)
    mask = im.complete_mask(pd.bdate_range("2019-03-25", "2019-04-02"))
    assert list(mask) == [False, False, False, False, True, True, True]


# ──────────────────────────────────────────────────────────
# The real change list
# ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def changes():
    return im.load_changes()


@pytest.fixture(scope="module")
def intervals():
    return im.nifty50_intervals()


# (first, last) sessions with a temporary 51st member: a demerged company held at zero
# price under a dummy symbol (JIOFIN 2023, ITC Hotels 2025, TMCV 2025, Kwality Wall's 2025-26)
TEMPORARY_51ST = [
    (D(2023, 7, 20), D(2023, 9, 6)),
    (D(2025, 1, 6), D(2025, 2, 7)),
    (D(2025, 10, 14), D(2025, 11, 14)),
    (D(2025, 12, 5), D(2026, 2, 23)),
]


def test_fifty_members_on_every_business_day(intervals):
    days = pd.bdate_range(im.COMPLETE_FROM, im.CURRENT_AS_OF)
    tickers = sorted(set(intervals["ticker"]))
    counts = im.membership_matrix(intervals, days, tickers).sum(axis=1)
    expected = pd.Series(50, index=days)
    for first, last in TEMPORARY_51ST:
        expected[(days >= pd.Timestamp(first)) & (days <= pd.Timestamp(last))] = 51
    bad = counts[counts != expected]
    assert bad.empty, bad.head()
    assert (counts == 51).sum() > 0


def test_membership_today_is_the_current_list(intervals):
    assert im.members_as_of(intervals, im.CURRENT_AS_OF) == {t for t, *_ in NIFTY50_TICKERS}
    assert im.members_as_of(intervals, D(2026, 9, 29)) >= {"WIPRO.NS"}
    assert "WIPRO.NS" not in im.members_as_of(intervals, D(2026, 9, 30))
    assert "BSE.NS" not in im.members_as_of(intervals, D(2026, 9, 29))
    assert "BSE.NS" in im.members_as_of(intervals, D(2026, 9, 30))


def test_known_changes_on_boundary_dates(intervals):
    # HDFC merged into HDFC Bank: last session 2023-07-12, LTIMindtree first session 2023-07-13
    assert "HDFC.NS" in im.members_as_of(intervals, D(2023, 7, 12))
    assert "LTIM.NS" not in im.members_as_of(intervals, D(2023, 7, 12))
    assert "HDFC.NS" not in im.members_as_of(intervals, D(2023, 7, 13))
    assert "LTIM.NS" in im.members_as_of(intervals, D(2023, 7, 13))
    # the five exit dates the tickers file called unverified
    for t, last in [("BPCL.NS", D(2025, 3, 27)), ("BRITANNIA.NS", D(2025, 3, 27)),
                    ("DIVISLAB.NS", D(2024, 9, 29)), ("HEROMOTOCO.NS", D(2025, 9, 29)),
                    ("INDUSINDBK.NS", D(2025, 9, 29))]:
        row = intervals[(intervals["ticker"] == t) & intervals["effective_to"].notna()]
        assert list(row["effective_to"]) == [last], t
    # the continuing Tata Motors company never left; the CV arm was a temporary member
    assert "TMPV.NS" in im.members_as_of(intervals, D(2025, 11, 17))
    assert "TMCV.NS" in im.members_as_of(intervals, D(2025, 11, 14))
    assert "TMCV.NS" not in im.members_as_of(intervals, D(2025, 11, 17))


def test_csv_shape_and_sources(changes):
    assert list(changes.columns[:5]) == ["effective_date", "action", "ticker", "name", "source"]
    assert set(changes["action"]) == {"add", "remove"}
    assert not changes.duplicated(["effective_date", "ticker"]).any()
    assert changes["source"].str.startswith("https://").all()
    assert (changes["source"].str.strip() != "").all() and (changes["name"].str.strip() != "").all()
    assert changes["ticker"].str.endswith(".NS").all()
    assert changes["effective_date"].min() >= im.COMPLETE_FROM


def test_csv_changes_are_consistent_forward(changes):
    """No add of an existing member and no removal of a non-member (forward replay)."""
    iv = im.nifty50_intervals()
    state = im.members_as_of(iv, im.HISTORY_START)
    assert len(state) == 50
    for d, group in changes.sort_values("effective_date").groupby("effective_date", sort=True):
        for c in group.itertuples():
            if c.action == "add":
                assert c.ticker not in state, (d, c.ticker)
            else:
                assert c.ticker in state, (d, c.ticker)
        for c in group.itertuples():
            state.add(c.ticker) if c.action == "add" else state.discard(c.ticker)
    assert state == {t for t, *_ in NIFTY50_TICKERS}


def test_temporary_members_are_flagged_in_the_csv(changes):
    flagged = changes[changes["note"].str.startswith("temporary")]
    assert set(flagged["ticker"]) == {"JIOFIN.NS", "ITCHOTELS.NS", "TMCV.NS", "KWIL.NS"}
    assert set(flagged["action"]) == {"add", "remove"}


def test_only_complete_stints_start_after_the_first_change(intervals):
    assert (intervals.loc[~intervals["complete"], "effective_from"] == im.HISTORY_START).all()
    assert intervals.loc[intervals["complete"], "effective_from"].min() >= im.COMPLETE_FROM


# ──────────────────────────────────────────────────────────
# DB sync
# ──────────────────────────────────────────────────────────

@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


def _flags(session):
    return {a.ticker: a.is_active for a in session.query(Asset)}


def test_sync_is_idempotent_and_keeps_active_flags(session):
    run_seed(session)
    n_assets = session.query(Asset).count()
    n_rows = session.query(IndexMembership).count()
    flags = _flags(session)
    assert n_rows == len(im.nifty50_intervals())
    assert flags["TMPV.NS"] is True and flags["HDFC.NS"] is False

    stats = im.sync_index_membership(session)
    session.commit()
    assert stats == {"inserted": 0, "updated": 0, "deleted": 0, "assets_created": 0}
    assert session.query(Asset).count() == n_assets
    assert session.query(IndexMembership).count() == n_rows
    assert _flags(session) == flags


def test_sync_creates_past_members_inactive_and_never_activates(session):
    run_seed(session)
    hdfc = session.query(Asset).filter_by(ticker="HDFC.NS").one()
    assert (hdfc.asset_type, hdfc.sector, hdfc.industry, hdfc.is_active) == ("stock", None, None, False)
    assert hdfc.name == "Housing Development Finance Corporation"
    # a flag set by hand (either way) survives the next sync
    hdfc.is_active = True
    session.query(Asset).filter_by(ticker="TCS.NS").one().is_active = False
    session.commit()
    im.sync_index_membership(session)
    assert session.query(Asset).filter_by(ticker="HDFC.NS").one().is_active is True
    assert session.query(Asset).filter_by(ticker="TCS.NS").one().is_active is False
    assert session.query(PriceDaily).count() == 0         # nothing fetched


def test_sync_creates_missing_assets_inactive(session):
    run_seed(session)
    nse = session.query(Exchange).filter_by(code="NSE").one()
    session.query(IndexMembership).delete()
    session.query(Asset).filter_by(ticker="YESBANK.NS").delete()
    session.commit()
    stats = im.sync_index_membership(session)
    assert stats["assets_created"] == 1
    yes = session.query(Asset).filter_by(exchange_id=nse.id, ticker="YESBANK.NS").one()
    assert yes.is_active is False and yes.name == "Yes Bank"


def test_sync_rows_match_intervals_and_repair_drift(session):
    run_seed(session)
    iv = im.nifty50_intervals()
    ids = {a.ticker: a.id for a in session.query(Asset)}
    tickers = {i: t for t, i in ids.items()}
    rows = {(tickers[r.asset_id], r.effective_from): r for r in session.query(IndexMembership)}
    assert len(rows) == len(iv)
    r = rows[("HDFC.NS", im.HISTORY_START)]
    assert (r.effective_to, r.verified, r.index_code) == (D(2023, 7, 12), False, "NIFTY50")
    assert r.source.endswith("ind_prs04072023.pdf")
    assert rows[("LTIM.NS", D(2023, 7, 13))].verified is True

    # tamper: a stale stint and a wrong end date are repaired by the next sync
    session.add(IndexMembership(index_code="NIFTY50", asset_id=ids["TCS.NS"],
                                effective_from=D(2001, 1, 1), effective_to=None, source="x", verified=True))
    r.effective_to = D(2000, 1, 1)
    session.commit()
    stats = im.sync_index_membership(session)
    assert (stats["deleted"], stats["updated"], stats["inserted"]) == (1, 1, 0)
    assert session.query(IndexMembership).count() == len(iv)


def test_sync_unknown_index(session):
    run_seed(session)
    with pytest.raises(ValueError):
        im.sync_index_membership(session, "NIFTY999")


def test_membership_summary_counts_members_without_prices(session):
    run_seed(session)
    s = im.membership_summary(session)
    assert s["rows"] == len(im.nifty50_intervals())
    assert s["members"] == len(set(im.nifty50_intervals()["ticker"]))
    assert s["members_without_prices"] == s["members"]
    tcs = session.query(Asset).filter_by(ticker="TCS.NS").one()
    session.add(PriceDaily(asset_id=tcs.id, date=D(2024, 1, 2), open=1, high=1, low=1, close=1, volume=1))
    session.commit()
    assert im.membership_summary(session)["members_without_prices"] == s["members"] - 1
