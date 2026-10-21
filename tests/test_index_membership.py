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
    around = pd.bdate_range(im.COMPLETE_FROM - pd.Timedelta(days=4), im.COMPLETE_FROM + pd.Timedelta(days=4))
    mask = im.complete_mask(around)
    assert list(mask) == [d >= pd.Timestamp(im.COMPLETE_FROM) for d in around]
    assert mask.any() and not mask.all()


# ──────────────────────────────────────────────────────────
# The real change list
# ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def changes():
    return im.load_changes()


@pytest.fixture(scope="module")
def intervals():
    return im.nifty50_intervals()


# (first, last) sessions with a temporary 51st member: the Tata Motors DVR share class as an
# additional security (2016-04-01 .. 2017-09-28), then demerged companies held at zero price
# under a dummy symbol (JIOFIN 2023, ITC Hotels 2025, TMCV 2025, Kwality Wall's 2025-26)
TEMPORARY_51ST = [
    (D(2016, 4, 1), D(2017, 9, 28)),
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
    assert "LTM.NS" not in im.members_as_of(intervals, D(2023, 7, 12))
    assert "HDFC.NS" not in im.members_as_of(intervals, D(2023, 7, 13))
    assert "LTM.NS" in im.members_as_of(intervals, D(2023, 7, 13))
    # the five exit dates the tickers file called unverified
    for t, last in [("BPCL.NS", D(2025, 3, 27)), ("BRITANNIA.NS", D(2025, 3, 27)),
                    ("DIVISLAB.NS", D(2024, 9, 29)), ("HEROMOTOCO.NS", D(2025, 9, 29)),
                    ("INDUSINDBK.NS", D(2025, 9, 29))]:
        row = intervals[(intervals["ticker"] == t) & intervals["effective_to"].notna()]
        assert row["effective_to"].max() == last, t  # BPCL / Britannia / Hero also had earlier stints
    # the continuing Tata Motors company never left; the CV arm was a temporary member
    assert "TMPV.NS" in im.members_as_of(intervals, D(2025, 11, 17))
    assert "TMCV.NS" in im.members_as_of(intervals, D(2025, 11, 14))
    assert "TMCV.NS" not in im.members_as_of(intervals, D(2025, 11, 17))


def _on(intervals, d):
    return im.members_as_of(intervals, d)


def test_pre_2019_changes_match_the_releases(intervals):
    """Members the day before and on the first day of a change, as the press release lists it.

    ind_prs21022018.pdf (NIFTY 50, effective 2018-04-02): out AMBUJACEM, AUROPHARMA, BOSCHLTD;
    in BAJAJFINSV, GRASIM, TITAN.
    ind_prs22022016_2.pdf (effective 2016-04-01): out CAIRN, PNB, VEDL; in AUROPHARMA, INFRATEL,
    EICHERMOT and the Tata Motors DVR, so the index has 51 securities.
    ind_prs27082013.pdf (effective 2013-09-27, the start of the chain): RELINFRA out, WIPRO in.
    """
    out_2018 = {"AMBUJACEM.NS", "AUROPHARMA.NS", "BOSCHLTD.NS"}
    in_2018 = {"BAJAJFINSV.NS", "GRASIM.NS", "TITAN.NS"}
    before, on = _on(intervals, D(2018, 4, 1)), _on(intervals, D(2018, 4, 2))  # Sunday, Monday
    assert out_2018 <= before and not in_2018 & before and len(before) == 50
    assert in_2018 <= on and not out_2018 & on and len(on) == 50
    assert _on(intervals, D(2018, 3, 28)) == before  # nothing in between

    out_2016 = {"CAIRN.NS", "PNB.NS", "VEDL.NS"}
    in_2016 = {"AUROPHARMA.NS", "INDUSTOWER.NS", "EICHERMOT.NS", "TATAMTRDVR.NS"}
    before, on = _on(intervals, D(2016, 3, 31)), _on(intervals, D(2016, 4, 1))
    assert out_2016 <= before and not in_2016 & before and len(before) == 50
    assert in_2016 <= on and not out_2016 & on and len(on) == 51

    before, on = _on(intervals, D(2013, 9, 26)), _on(intervals, D(2013, 9, 27))
    assert "RELINFRA.NS" in before and "WIPRO.NS" not in before and len(before) == 50
    assert "WIPRO.NS" in on and "RELINFRA.NS" not in on and len(on) == 50
    assert im.is_complete(D(2013, 9, 27)) and im.is_complete(D(2013, 9, 26))


def test_pre_2013_changes_match_the_releases(intervals):
    """Members the day before and on the day of a change, as the press release lists it (task 440).

    ind_prs02091998.pdf (S&P CNX Nifty, effective 1998-10-07, the oldest release in the archive and
    the start of the chain): out BPCL, ASHOKLEY, INFOGULF, ANDRAVALLY, THERMAX, MRPL, PONDS; in
    INFOSYSTCH (INFY), NIIT, BANKINDIA, SMITHKBECH (GSKCONS), HEROHONDA (HEROMOTOCO), P&G (PGHH), CIPLA.
    ind_prs26022008.pdf + ind_prs31012008_2.pdf (effective 2008-03-14): out Bajaj Auto and
    GlaxoSmithKline Pharmaceuticals; in Power Grid and DLF.
    ind_prs14122001.pdf (effective 2002-01-17 and 2002-01-25, after the Cadbury decision was revised):
    out HCL Infosystems, Kochi Refineries, then Reckitt & Benckiser; in Sun Pharma, Wipro, then ICICI Bank.
    """
    assert im.COMPLETE_FROM == D(1998, 10, 7) and im.HISTORY_START == D(1998, 9, 1)
    out_1998 = {"BPCL.NS", "ASHOKLEY.NS", "INFOGULF.NS", "ANDRAVALLY.NS", "THERMAX.NS", "MRPL.NS", "PONDS.NS"}
    in_1998 = {"INFY.NS", "NIIT.NS", "BANKINDIA.NS", "GSKCONS.NS", "HEROMOTOCO.NS", "PGHH.NS", "CIPLA.NS"}
    before, on = _on(intervals, D(1998, 10, 6)), _on(intervals, D(1998, 10, 7))
    assert out_1998 <= before and not in_1998 & before and len(before) == 50
    assert in_1998 <= on and not out_1998 & on and len(on) == 50
    assert not im.is_complete(D(1998, 10, 6)) and im.is_complete(D(1998, 10, 7))

    before, on = _on(intervals, D(2008, 3, 13)), _on(intervals, D(2008, 3, 14))
    assert {"BAJAJ-AUTO.NS", "GLAXO.NS"} <= before and not {"POWERGRID.NS", "DLF.NS"} & before
    assert {"POWERGRID.NS", "DLF.NS"} <= on and not {"BAJAJ-AUTO.NS", "GLAXO.NS"} & on
    assert len(before) == len(on) == 50

    assert {"HCL-INSYS.NS", "KOCHIREFIN.NS", "RECKITTBEN.NS"} <= _on(intervals, D(2002, 1, 16))
    mid = _on(intervals, D(2002, 1, 17))     # first change in, the Reckitt swap not yet
    assert {"SUNPHARMA.NS", "WIPRO.NS", "RECKITTBEN.NS"} <= mid and not {"HCL-INSYS.NS", "KOCHIREFIN.NS", "ICICIBANK.NS"} & mid
    last = _on(intervals, D(2002, 1, 25))
    assert "ICICIBANK.NS" in last and "RECKITTBEN.NS" not in last and len(last) == 50

    # a company that came back: Reliance Capital left in 1999 and joined again in 2009
    assert "RELCAPITAL.NS" in _on(intervals, D(1999, 5, 25)) and "RELCAPITAL.NS" not in _on(intervals, D(1999, 5, 26))
    assert "RELCAPITAL.NS" not in _on(intervals, D(2009, 1, 9)) and "RELCAPITAL.NS" in _on(intervals, D(2009, 1, 12))
    # Sesa Goa / Vedanta is one ticker through the 2010 add and the 2016 removal
    assert "VEDL.NS" not in _on(intervals, D(2010, 9, 30)) and "VEDL.NS" in _on(intervals, D(2010, 10, 1))
    assert "VEDL.NS" in _on(intervals, D(2016, 3, 31)) and "VEDL.NS" not in _on(intervals, D(2016, 4, 1))


def test_members_as_of_pre_2013_uses_only_changes_up_to_d(changes):
    """Truncation: members_as_of(d) equals a forward replay of the changes dated <= d (pre-2013 part).

    Probed on both sides of every change before the 2013-08-27 release, from 1998 on. A change dated
    after d (e.g. the 2003-08-04 Nestle/SAIL swap, the 2013-04-01 swap) never shows up on d.
    """
    iv = im.nifty50_intervals()
    initial = im.members_as_of(iv, im.HISTORY_START)
    pre = changes[changes["effective_date"] < D(2013, 9, 27)]
    assert len(pre) > 100
    probes = {im.HISTORY_START, D(2013, 9, 26)}
    for d in set(pre["effective_date"]):
        probes |= {d - pd.Timedelta(days=1).to_pytimedelta(), d}
    for d in sorted(probes):
        state = set(initial)
        for c in changes[changes["effective_date"] <= d].sort_values("effective_date", kind="stable").itertuples():
            state.add(c.ticker) if c.action == "add" else state.remove(c.ticker)
        assert im.members_as_of(iv, d) == state, d
    assert "SAIL.NS" not in im.members_as_of(iv, D(2003, 8, 3)) and "SAIL.NS" in im.members_as_of(iv, D(2003, 8, 4))
    assert "NESTLEIND.NS" in im.members_as_of(iv, D(2003, 8, 3)) and "NESTLEIND.NS" not in im.members_as_of(iv, D(2003, 8, 4))


def test_members_as_of_pre_2019_uses_only_changes_up_to_d(changes):
    """Forward replay of the changes dated <= d from the initial set equals members_as_of(d).

    The same truncation property as the toy test, on the real pre-2019 chain: a later change
    can never leak into an earlier date. Checked on both sides of every pre-2019 change date.
    """
    iv = im.nifty50_intervals()
    initial = im.members_as_of(iv, im.HISTORY_START)
    pre = changes[changes["effective_date"] < D(2019, 1, 1)]
    assert len(pre) > 20
    probes = {D(2013, 8, 1), D(2018, 12, 31)}
    for d in set(pre["effective_date"]):
        probes |= {d - pd.Timedelta(days=1).to_pytimedelta(), d}
    for d in sorted(probes):
        state = set(initial)
        for c in changes[changes["effective_date"] <= d].sort_values("effective_date", kind="stable").itertuples():
            state.add(c.ticker) if c.action == "add" else state.remove(c.ticker)
        assert im.members_as_of(iv, d) == state, d


def test_csv_shape_and_sources(changes):
    assert list(changes.columns[:5]) == ["effective_date", "action", "ticker", "name", "source"]
    assert set(changes["action"]) == {"add", "remove"}
    assert not changes.duplicated(["effective_date", "ticker"]).any()
    assert changes["source"].str.startswith("https://www.niftyindices.com/Press_Release/ind_prs").all()
    assert changes["source"].str.endswith(".pdf").all()
    assert (changes["source"].str.strip() != "").all() and (changes["name"].str.strip() != "").all()
    assert changes["ticker"].str.endswith(".NS").all()
    assert changes["effective_date"].min() == im.COMPLETE_FROM
    # the first release read is dated in the month HISTORY_START opens
    assert im.HISTORY_START.day == 1 and im.HISTORY_START < im.COMPLETE_FROM
    assert (im.COMPLETE_FROM - im.HISTORY_START).days < 62


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
    assert set(flagged["ticker"]) == {"JIOFIN.NS", "ITCHOTELS.NS", "TMCV.NS", "KWIL.NS", "TATAMTRDVR.NS"}
    assert set(flagged["action"]) == {"add", "remove"}


def test_every_source_is_a_press_release_and_symbols_are_flagged(changes):
    """Rows added in task 440: official source, and a name-only symbol is never left unflagged."""
    old = changes[changes["effective_date"] < D(2013, 9, 27)]
    assert old["source"].str.match(r"^https://www\.niftyindices\.com/Press_Release/ind_prs\d{8}(_\d)?\.pdf$").all()
    assert (old["note"].str.strip() != "").sum() > 50
    assert not (old["note"].str.contains("assumed") & ~old["note"].str.contains("not verified")).any()


SYMBOL_CHANGES_CSV = im.CHANGES_DIR / "nse_symbol_changes.csv"
SYMBOL_CHANGES_URL = "https://nsearchives.nseindia.com/content/equities/symbolchange.csv"


@pytest.fixture(scope="module")
def symbol_changes():
    return pd.read_csv(SYMBOL_CHANGES_CSV, dtype=str, keep_default_na=False)


def test_nse_symbol_changes_shape_and_source(symbol_changes):
    assert list(symbol_changes.columns) == ["old_symbol", "new_symbol", "change_date", "company_name", "source"]
    assert len(symbol_changes) > 0
    assert (symbol_changes["source"] == SYMBOL_CHANGES_URL).all()
    assert (symbol_changes[["old_symbol", "new_symbol", "company_name"]].apply(lambda c: c.str.strip() != "")).all().all()
    assert pd.to_datetime(symbol_changes["change_date"], format="%Y-%m-%d").notna().all()
    assert not symbol_changes["old_symbol"].duplicated().any()


def test_nse_symbol_changes_are_applied_to_the_change_list(symbol_changes, changes):
    tickers = set(changes["ticker"])
    # an old symbol is gone from the change list, the new one is in it
    assert not {o + ".NS" for o in symbol_changes["old_symbol"]} & tickers
    assert {n + ".NS" for n in symbol_changes["new_symbol"]} <= tickers
    # every row of a renamed ticker cites the change in its note, and no longer claims it is unverified
    for r in symbol_changes.itertuples():
        rows = changes[changes["ticker"] == r.new_symbol + ".NS"]
        cite = f"NSE symbol change {r.old_symbol}\u2192{r.new_symbol} {r.change_date} (nse_symbol_changes.csv)"
        assert rows["note"].str.contains(cite, regex=False).all(), r.old_symbol
        assert not rows["note"].str.contains("current Yahoo symbol not verified").any(), r.old_symbol


def test_nse_symbol_changes_do_not_clash_with_the_current_list(symbol_changes):
    current = {t for t, *_ in NIFTY50_TICKERS}
    assert not {o + ".NS" for o in symbol_changes["old_symbol"]} & current


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
    assert rows[("LTM.NS", D(2023, 7, 13))].verified is True

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
