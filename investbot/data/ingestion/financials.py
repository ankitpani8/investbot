"""
Financial statement ingestion via yfinance.
Pulls income statement, balance sheet, and cash flow data.
Maps yfinance field names to our schema and upserts.
"""
import logging
from datetime import datetime

import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session

from investbot.data.models import Asset, Financial, FinancialDataAvailability
from investbot.scoring.fundamental.point_in_time import (
    SOURCE_LODR_LAG,
    available_from,
    fiscal_year_end_months,
)

logger = logging.getLogger(__name__)


# ── Field mappings: yfinance row labels → our column names ──
# yfinance returns financial statements as DataFrames with row labels as INDEX
# (not columns) and reporting dates as columns.

INCOME_STMT_MAP = {
    "Total Revenue": "revenue",
    "Operating Revenue": "revenue",
    "Cost Of Revenue": "cogs",
    "Cost of Revenue": "cogs",
    "Gross Profit": "gross_profit",
    "Selling General And Administration": "sga_expense",
    "Selling General And Administrative": "sga_expense",
    "Research And Development": "rnd_expense",
    "Research Development": "rnd_expense",
    "Depreciation": "depreciation",
    "Depreciation And Amortization In Income Statement": "depreciation",
    "Amortization": "amortization",
    "Amortization Of Intangibles Income Statement": "amortization",
    "Interest Expense": "interest_expense",
    "Interest Expense Non Operating": "interest_expense",
    "Restructuring And Mergern Acquisition": "restructuring",
    "Other Operating Expenses": "other_operating",
    "EBIT": "ebit",
    "Pretax Income": "pretax_income",
    "Tax Provision": "tax_paid",
    "Net Income": "net_income",
    "Net Income Common Stockholders": "net_income",
    "Basic EPS": "eps",
    "Diluted EPS": "eps",
    "Basic Average Shares": "shares_outstanding",
    "Diluted Average Shares": "shares_outstanding",
}

BALANCE_SHEET_MAP = {
    # Current Assets
    "Cash And Cash Equivalents": "cash_and_equivalents",
    "Cash Cash Equivalents And Short Term Investments": "cash_and_equivalents",
    "Other Short Term Investments": "short_term_investments",
    "Accounts Receivable": "accounts_receivable",
    "Net Receivables": "net_receivables",
    "Receivables": "net_receivables",
    "Inventory": "inventory",
    "Prepaid Assets": "prepaid_expenses",
    "Other Current Assets": "other_current_assets",
    "Current Assets": "total_current_assets",
    "Total Current Assets": "total_current_assets",
    # Non-Current Assets
    "Gross PPE": "property_plant_equipment",
    "Net PPE": "net_ppe",
    "Accumulated Depreciation": "accumulated_depreciation",
    "Goodwill": "goodwill",
    "Goodwill And Other Intangible Assets": "goodwill",
    "Other Intangible Assets": "intangible_assets",
    "Investments And Advances": "long_term_investments",
    "Long Term Equity Investment": "long_term_investments",
    "Other Non Current Assets": "other_noncurrent_assets",
    "Total Assets": "total_assets",
    # Current Liabilities
    "Accounts Payable": "accounts_payable",
    "Payables And Accrued Expenses": "accounts_payable",
    "Current Debt": "short_term_debt",
    "Current Debt And Capital Lease Obligation": "short_term_debt",
    "Current Deferred Liabilities": "deferred_revenue_current",
    "Other Current Liabilities": "other_current_liabilities",
    "Current Liabilities": "total_current_liabilities",
    "Total Current Liabilities": "total_current_liabilities",
    # Non-Current Liabilities
    "Long Term Debt": "long_term_debt",
    "Long Term Debt And Capital Lease Obligation": "long_term_debt",
    "Non Current Deferred Taxes Liabilities": "deferred_tax_liabilities",
    "Non Current Deferred Revenue": "deferred_revenue_noncurrent",
    "Non Current Pension And Other Postretirement Benefit Plans": "pension_obligations",
    "Other Non Current Liabilities": "other_noncurrent_liabilities",
    "Total Liabilities Net Minority Interest": "total_liabilities",
    # Shareholders Equity
    "Common Stock": "common_stock",
    "Capital Stock": "common_stock",
    "Preferred Stock": "preferred_stock",
    "Additional Paid In Capital": "additional_paid_in_capital",
    "Retained Earnings": "retained_earnings",
    "Treasury Shares Number": "treasury_stock",
    "Gains Losses Not Affecting Retained Earnings": "accumulated_other_comprehensive_income",
    "Minority Interest": "minority_interest",
    "Stockholders Equity": "total_shareholders_equity",
    "Total Equity Gross Minority Interest": "total_shareholders_equity",
}

CASH_FLOW_MAP = {
    # Operating
    "Operating Cash Flow": "operating_cash_flow",
    "Cash Flow From Continuing Operating Activities": "operating_cash_flow",
    "Depreciation And Amortization": "depreciation_amortization_cf",
    "Depreciation Amortization Depletion": "depreciation_amortization_cf",
    "Stock Based Compensation": "stock_based_compensation",
    "Change In Working Capital": "change_in_working_capital",
    "Change In Receivables": "change_in_receivables",
    "Changes In Account Receivables": "change_in_receivables",
    "Change In Inventory": "change_in_inventory",
    "Change In Payable": "change_in_payables",
    "Change In Payables And Accrued Expense": "change_in_payables",
    "Other Operating Activities": "other_operating_activities",
    "Other Non Cash Items": "other_operating_activities",
    # Investing
    "Capital Expenditure": "capital_expenditure",
    "Purchase Of Business": "acquisitions",
    "Net Business Purchase And Sale": "acquisitions",
    "Purchase Of Investment": "purchase_of_investments",
    "Sale Of Investment": "sale_of_investments",
    "Other Investing Activities": "other_investing_activities",
    "Investing Cash Flow": "investing_cash_flow",
    "Cash Flow From Continuing Investing Activities": "investing_cash_flow",
    # Financing
    "Issuance Of Debt": "debt_issued",
    "Long Term Debt Issuance": "debt_issued",
    "Repayment Of Debt": "debt_repaid",
    "Long Term Debt Payments": "debt_repaid",
    "Common Stock Dividend Paid": "dividends_paid",
    "Cash Dividends Paid": "dividends_paid",
    "Repurchase Of Capital Stock": "share_buybacks",
    "Common Stock Issuance": "shares_issued",
    "Other Financing Activities": "other_financing_activities",
    "Financing Cash Flow": "financing_cash_flow",
    "Cash Flow From Continuing Financing Activities": "financing_cash_flow",
    # Derived
    "Free Cash Flow": "free_cash_flow",
    "Changes In Cash": "net_change_in_cash",
    "Change In Cash Supplemental As Reported": "net_change_in_cash",
}


def _extract_column(df: pd.DataFrame, col_date, field_map: dict) -> dict:
    """
    Extract fields from one column of a financial statement DataFrame.
    df has row labels as index and dates as columns.
    Returns {our_col_name: float_value}.
    """
    result = {}
    if df is None or df.empty or col_date not in df.columns:
        return result

    series = df[col_date]  # Series indexed by row labels
    for yf_label, our_col in field_map.items():
        if yf_label in series.index:
            val = series[yf_label]
            if pd.notna(val):
                try:
                    result[our_col] = float(val)
                except (ValueError, TypeError):
                    pass
    return result


def ingest_financials_for_asset(session: Session, asset: Asset, period_type: str = "annual"):
    """Pull financial statements for one asset and upsert into financials table."""
    ticker = asset.ticker
    logger.info(f"Ingesting {period_type} financials for {ticker}...")

    try:
        stock = yf.Ticker(ticker)
        if period_type == "annual":
            income_stmt = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
        else:
            income_stmt = stock.quarterly_financials
            balance_sheet = stock.quarterly_balance_sheet
            cash_flow = stock.quarterly_cashflow
    except Exception as e:
        logger.error(f"Failed to fetch financials for {ticker}: {e}")
        return 0

    # Collect all reporting dates across the three statements
    all_dates = set()
    for df in [income_stmt, balance_sheet, cash_flow]:
        if df is not None and not df.empty:
            all_dates.update(df.columns)

    if not all_dates:
        logger.warning(f"No financial data available for {ticker}")
        return 0

    # Q4 gets the annual lag in every fiscal-year-end month of this company: March plus
    # the months of all its annual period ends (stored, plus the ones in this call).
    # Some of those annual rows are published after a given quarter, but the month a
    # company closes its books is static company metadata, not price or result data,
    # and the union with March can only make a row's date later, never earlier, so it
    # cannot add lookahead.
    annual_ends = [
        pe for (pe,) in session.query(Financial.period_end).filter(
            Financial.asset_id == asset.id,
            Financial.period_type == "annual",
            Financial.period_end.isnot(None),
        )
    ]
    if period_type == "annual":
        annual_ends += [pd.Timestamp(d).date() for d in all_dates]
    fy_end_months = fiscal_year_end_months(annual_ends)

    rows_added = 0
    for col_date in sorted(all_dates):
        # Statement columns are period-end dates, not publication dates.
        period_end = pd.Timestamp(col_date).date()
        fiscal_year = period_end.year

        # Skip if we already have this period (keyed on period_end: several quarters share a year)
        existing = session.query(Financial).filter_by(
            asset_id=asset.id, period_end=period_end, period_type=period_type
        ).first()
        if existing:
            continue

        # yfinance gives no publication date: filing_date stays NULL and availability
        # falls back to the SEBI LODR lag.
        avail, avail_source = available_from(
            period_end, period_type, fiscal_year_end_months=fy_end_months
        )

        # Build the row from all three statements
        data = {
            "asset_id": asset.id,
            "fiscal_year": fiscal_year,
            "period_type": period_type,
            "period_end": period_end,
            "filing_date": None,
            "available_from": avail,
            "available_from_source": avail_source,
        }

        has_income = income_stmt is not None and col_date in income_stmt.columns
        has_bs = balance_sheet is not None and col_date in balance_sheet.columns
        has_cf = cash_flow is not None and col_date in cash_flow.columns

        if has_income:
            data.update(_extract_column(income_stmt, col_date, INCOME_STMT_MAP))
        if has_bs:
            data.update(_extract_column(balance_sheet, col_date, BALANCE_SHEET_MAP))
        if has_cf:
            data.update(_extract_column(cash_flow, col_date, CASH_FLOW_MAP))

        # Derive free_cash_flow if not supplied (capex is typically negative)
        if "free_cash_flow" not in data and "operating_cash_flow" in data and "capital_expenditure" in data:
            data["free_cash_flow"] = data["operating_cash_flow"] + data["capital_expenditure"]

        try:
            session.add(Financial(**data))
            rows_added += 1
        except Exception as e:
            logger.warning(f"Failed to insert financials for {ticker} {period_end}: {e}")
            continue

        _update_availability(session, asset.id, fiscal_year, period_type, has_income, has_bs, has_cf)

    # A new annual month makes earlier-stored quarters of that month a Q4.
    rows_relagged = relag_quarterly_rows(session, asset.id, fy_end_months)

    if rows_added > 0 or rows_relagged > 0:
        session.commit()
    logger.info(f"{ticker}: added {rows_added} {period_type} financial records")
    if rows_relagged:
        logger.info(f"{ticker}: moved {rows_relagged} quarterly rows to the annual lag")
    return rows_added


def relag_quarterly_rows(session: Session, asset_id: int, fy_end_months) -> int:
    """Recompute ``available_from`` of the asset's stored ``lodr_lag`` quarterly rows.

    Only moves a date later, never earlier; ``published`` rows and rows without
    ``available_from`` are left alone. Does not commit. Returns the number of rows moved.
    """
    rows = session.query(Financial).filter(
        Financial.asset_id == asset_id,
        Financial.period_type == "quarterly",
        Financial.available_from_source == SOURCE_LODR_LAG,
        Financial.available_from.isnot(None),
        Financial.period_end.isnot(None),
    )
    moved = 0
    for row in rows:
        new, _ = available_from(row.period_end, "quarterly", fiscal_year_end_months=fy_end_months)
        if new > row.available_from:
            row.available_from = new
            moved += 1
    return moved


def _update_availability(session, asset_id, fiscal_year, period_type, has_income, has_bs, has_cf):
    """Update or create financial_data_availability record."""
    existing = session.query(FinancialDataAvailability).filter_by(
        asset_id=asset_id, fiscal_year=fiscal_year, period_type=period_type
    ).first()

    if existing:
        existing.income_stmt_available = has_income
        existing.balance_sheet_available = has_bs
        existing.cash_flow_available = has_cf
        existing.last_updated = datetime.utcnow()
    else:
        session.add(FinancialDataAvailability(
            asset_id=asset_id,
            fiscal_year=fiscal_year,
            period_type=period_type,
            income_stmt_available=has_income,
            balance_sheet_available=has_bs,
            cash_flow_available=has_cf,
            last_updated=datetime.utcnow(),
        ))


def ingest_all_financials(session: Session, asset_type: str = "stock"):
    """Ingest annual + quarterly financials for all active assets."""
    assets = session.query(Asset).filter_by(is_active=True, asset_type=asset_type).all()
    logger.info(f"Starting financial ingestion for {len(assets)} {asset_type} assets")

    stats = {"annual": 0, "quarterly": 0, "errors": []}
    for i, asset in enumerate(assets, 1):
        logger.info(f"[{i}/{len(assets)}] Financials for {asset.ticker}...")
        try:
            stats["annual"] += ingest_financials_for_asset(session, asset, "annual")
            stats["quarterly"] += ingest_financials_for_asset(session, asset, "quarterly")
        except Exception as e:
            logger.error(f"Error for {asset.ticker}: {e}")
            stats["errors"].append(asset.ticker)

    logger.info(
        f"Financial ingestion complete: {stats['annual']} annual, "
        f"{stats['quarterly']} quarterly records, {len(stats['errors'])} errors"
    )
    return stats
