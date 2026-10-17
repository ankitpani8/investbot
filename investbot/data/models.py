"""
SQLAlchemy ORM models for InvestBot.
Implements ALL tables from master plan v3 §4 — schema-complete from day one.
Tables without scoring logic yet still get their columns populated for future use.
"""
from datetime import datetime, date, time
from sqlalchemy import (
    Column, Integer, Float, String, Text, Boolean, Date, Time, DateTime,
    ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ──────────────────────────────────────────────────────────
# §4.1 — Core Tables (Asset-Class Agnostic)
# ──────────────────────────────────────────────────────────

class AssetClass(Base):
    __tablename__ = "asset_classes"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)                       # 'equity', 'crypto', 'commodity', 'forex'
    description = Column(Text)
    scoring_interval_minutes = Column(Integer, default=30)
    market_open = Column(Time)                                  # '09:15' for NSE, NULL for 24/7
    market_close = Column(Time)                                 # '15:30' for NSE, NULL for 24/7
    is_24x7 = Column(Boolean, default=False)

    exchanges = relationship("Exchange", back_populates="asset_class")


class Exchange(Base):
    __tablename__ = "exchanges"

    id = Column(Integer, primary_key=True)
    asset_class_id = Column(Integer, ForeignKey("asset_classes.id"), nullable=False)
    code = Column(String, nullable=False, unique=True)          # 'NSE', 'BSE', 'BINANCE'
    name = Column(String, nullable=False)
    country = Column(String)
    currency = Column(String, nullable=False)                   # 'INR', 'USD'

    asset_class = relationship("AssetClass", back_populates="exchanges")
    assets = relationship("Asset", back_populates="exchange")


class Asset(Base):
    __tablename__ = "assets"
    __table_args__ = (UniqueConstraint("exchange_id", "ticker"),)

    id = Column(Integer, primary_key=True)
    exchange_id = Column(Integer, ForeignKey("exchanges.id"), nullable=False)
    ticker = Column(String, nullable=False)                     # 'RELIANCE.NS', 'BTC-USD'
    name = Column(String, nullable=False)
    asset_type = Column(String)                                 # 'stock', 'etf', 'crypto'
    sector = Column(String)
    industry = Column(String)
    is_active = Column(Boolean, default=True)

    exchange = relationship("Exchange", back_populates="assets")


# ──────────────────────────────────────────────────────────
# §4.2 — Price Data
# ──────────────────────────────────────────────────────────

class PriceDaily(Base):
    __tablename__ = "price_daily"
    __table_args__ = (UniqueConstraint("asset_id", "date"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(Integer)


class PriceIntraday(Base):
    __tablename__ = "price_intraday"
    __table_args__ = (UniqueConstraint("asset_id", "timestamp", "interval"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    interval = Column(String, nullable=False)                   # '15m', '30m', '1h'
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)


# ──────────────────────────────────────────────────────────
# §4.3 — Fundamental Data: Equity Financials
# ──────────────────────────────────────────────────────────

class Financial(Base):
    __tablename__ = "financials"
    __table_args__ = (UniqueConstraint("asset_id", "fiscal_year", "period_type"),)

    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    fiscal_year = Column(Integer, nullable=False)
    period_type = Column(String, default="annual")              # 'annual' or 'quarterly'
    filing_date = Column(Date)

    # === INCOME STATEMENT ===
    revenue = Column(Float)
    cogs = Column(Float)
    gross_profit = Column(Float)
    sga_expense = Column(Float)
    rnd_expense = Column(Float)
    depreciation = Column(Float)
    amortization = Column(Float)
    interest_expense = Column(Float)
    restructuring = Column(Float)
    other_operating = Column(Float)
    ebit = Column(Float)
    pretax_income = Column(Float)
    tax_paid = Column(Float)
    net_income = Column(Float)
    eps = Column(Float)
    shares_outstanding = Column(Float)

    # === BALANCE SHEET ===
    # Current Assets
    cash_and_equivalents = Column(Float)
    short_term_investments = Column(Float)
    accounts_receivable = Column(Float)
    net_receivables = Column(Float)
    inventory = Column(Float)
    prepaid_expenses = Column(Float)
    other_current_assets = Column(Float)
    total_current_assets = Column(Float)
    # Non-Current Assets
    property_plant_equipment = Column(Float)
    accumulated_depreciation = Column(Float)
    net_ppe = Column(Float)
    goodwill = Column(Float)
    intangible_assets = Column(Float)
    long_term_investments = Column(Float)
    other_noncurrent_assets = Column(Float)
    total_assets = Column(Float)
    # Current Liabilities
    accounts_payable = Column(Float)
    short_term_debt = Column(Float)
    current_portion_lt_debt = Column(Float)
    accrued_liabilities = Column(Float)
    deferred_revenue_current = Column(Float)
    other_current_liabilities = Column(Float)
    total_current_liabilities = Column(Float)
    # Non-Current Liabilities
    long_term_debt = Column(Float)
    deferred_tax_liabilities = Column(Float)
    deferred_revenue_noncurrent = Column(Float)
    pension_obligations = Column(Float)
    other_noncurrent_liabilities = Column(Float)
    total_liabilities = Column(Float)
    # Shareholders Equity
    common_stock = Column(Float)
    preferred_stock = Column(Float)
    additional_paid_in_capital = Column(Float)
    retained_earnings = Column(Float)
    treasury_stock = Column(Float)
    accumulated_other_comprehensive_income = Column(Float)
    minority_interest = Column(Float)
    total_shareholders_equity = Column(Float)

    # === CASH FLOW STATEMENT ===
    # Operating Activities
    operating_cash_flow = Column(Float)
    depreciation_amortization_cf = Column(Float)
    stock_based_compensation = Column(Float)
    change_in_working_capital = Column(Float)
    change_in_receivables = Column(Float)
    change_in_inventory = Column(Float)
    change_in_payables = Column(Float)
    other_operating_activities = Column(Float)
    # Investing Activities
    capital_expenditure = Column(Float)
    acquisitions = Column(Float)
    purchase_of_investments = Column(Float)
    sale_of_investments = Column(Float)
    other_investing_activities = Column(Float)
    investing_cash_flow = Column(Float)
    # Financing Activities
    debt_issued = Column(Float)
    debt_repaid = Column(Float)
    dividends_paid = Column(Float)
    share_buybacks = Column(Float)
    shares_issued = Column(Float)
    other_financing_activities = Column(Float)
    financing_cash_flow = Column(Float)
    # Derived
    free_cash_flow = Column(Float)
    net_change_in_cash = Column(Float)


class FinancialDataAvailability(Base):
    __tablename__ = "financial_data_availability"
    __table_args__ = (UniqueConstraint("asset_id", "fiscal_year", "period_type"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    fiscal_year = Column(Integer, nullable=False)
    period_type = Column(String, nullable=False)
    income_stmt_available = Column(Boolean, default=False)
    balance_sheet_available = Column(Boolean, default=False)
    cash_flow_available = Column(Boolean, default=False)
    last_updated = Column(DateTime)


# ──────────────────────────────────────────────────────────
# §4.4 — Fundamental Data: Crypto On-Chain Metrics
# ──────────────────────────────────────────────────────────

class OnchainMetric(Base):
    __tablename__ = "onchain_metrics"
    __table_args__ = (UniqueConstraint("asset_id", "date"),)

    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    date = Column(Date, nullable=False)
    # Network Activity
    daily_active_addresses = Column(Integer)
    new_addresses = Column(Integer)
    transaction_count = Column(Integer)
    transaction_volume_usd = Column(Float)
    nvt_ratio = Column(Float)
    nvt_signal_90d = Column(Float)
    # Supply & Holders
    exchange_net_flow = Column(Float)
    exchange_reserve = Column(Float)
    hodl_wave_1yr_plus_pct = Column(Float)
    mvrv_ratio = Column(Float)
    sopr = Column(Float)
    supply_in_profit_pct = Column(Float)
    whale_concentration_top100 = Column(Float)
    stablecoin_supply_ratio = Column(Float)
    # Mining (PoW only)
    hash_rate = Column(Float)
    difficulty = Column(Float)
    miner_revenue_usd = Column(Float)
    fee_revenue_pct = Column(Float)
    puell_multiple = Column(Float)
    # DeFi / Ecosystem
    tvl_usd = Column(Float)
    tvl_market_cap_ratio = Column(Float)
    dex_volume_usd = Column(Float)
    protocol_revenue_usd = Column(Float)
    developer_commits_30d = Column(Integer)
    # Market Structure
    btc_dominance_pct = Column(Float)
    funding_rate = Column(Float)
    open_interest_usd = Column(Float)


# ──────────────────────────────────────────────────────────
# §4.5 — Scoring & Indicators
# ──────────────────────────────────────────────────────────

class IndicatorCache(Base):
    __tablename__ = "indicator_cache"
    __table_args__ = (UniqueConstraint("asset_id", "timestamp", "indicator"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    indicator = Column(String, nullable=False)                  # 'RSI_14', 'MACD', 'HURST_100'
    value = Column(Float)


class Score(Base):
    __tablename__ = "scores"
    __table_args__ = (UniqueConstraint("asset_id", "timestamp"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    technical_score = Column(Float)
    fundamental_score = Column(Float)
    sentiment_score = Column(Float)
    regime_confidence = Column(Float)
    combined_score = Column(Float)
    signal_reasons = Column(Text)                               # JSON


class TechnicalScoreRecord(Base):
    """One technical score per (asset, timeframe, bar), written by scoring/technical/scorer.py.

    Not in master plan v3: §4.5 only has the combined `scores` table (0-100
    sub-scores). This keeps the full technical breakdown on its native [-1, 1]
    scale; the combined layer maps it into `scores.technical_score`.
    """
    __tablename__ = "technical_scores"
    __table_args__ = (UniqueConstraint("asset_id", "timeframe", "timestamp"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    timeframe = Column(String, nullable=False)                  # '1d', '15m'
    timestamp = Column(DateTime, nullable=False)                # scored bar (daily: date 00:00; intraday: bar open)
    tech_score = Column(Float)                                  # -1 to +1, NULL during warmup
    score_trend = Column(Float)
    score_momentum = Column(Float)
    score_volatility = Column(Float)
    score_volume = Column(Float)
    score_pattern = Column(Float)
    score_sr = Column(Float)
    score_mtf = Column(Float)
    regime = Column(String)                                     # 'trend', 'range', 'chaotic'
    regime_confidence = Column(Float)                           # 0-1
    signal_reasons = Column(Text)                               # JSON list of str
    # Data quality
    status = Column(String, nullable=False)                     # 'ok', 'warmup', 'insufficient_data', 'error'
    bars_available = Column(Integer)
    warmup_satisfied = Column(Boolean)
    staleness_days = Column(Integer)                            # business days from bar to reference time, at compute time
    # Cache
    input_hash = Column(String)                                 # fingerprint of the OHLCV window scored
    scorer_version = Column(String)
    computed_at = Column(DateTime)


# ──────────────────────────────────────────────────────────
# §4.6 — Sentiment
# ──────────────────────────────────────────────────────────

class SentimentRaw(Base):
    __tablename__ = "sentiment_raw"

    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey("assets.id"))
    source = Column(String, nullable=False)                     # 'reddit', 'twitter'
    subreddit = Column(String)
    post_id = Column(String)
    timestamp = Column(DateTime, nullable=False)
    text = Column(Text)
    sentiment = Column(Float)                                   # -1 to +1
    magnitude = Column(Float)
    model_version = Column(String)


class SentimentAggregated(Base):
    __tablename__ = "sentiment_aggregated"
    __table_args__ = (UniqueConstraint("asset_id", "date", "source"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    date = Column(Date, nullable=False)
    source = Column(String, nullable=False)
    avg_sentiment = Column(Float)
    mention_count = Column(Integer)
    bullish_pct = Column(Float)
    bearish_pct = Column(Float)
    neutral_pct = Column(Float)
    contrarian_flag = Column(Boolean, default=False)


# ──────────────────────────────────────────────────────────
# §4.7 — Users
# ──────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active_at = Column(DateTime)

    # Identity & Account
    name = Column(String)
    email = Column(String)
    phone = Column(String)
    telegram_id = Column(String, unique=True)
    pan_number = Column(String)
    country_of_residence = Column(String, default="IN")
    tax_residency_status = Column(String, default="resident")
    age = Column(Integer)
    kyc_status = Column(String, default="pending")

    # Capital & Cash Flow
    initial_capital = Column(Float, default=0)
    monthly_contribution_amount = Column(Float, default=0)
    monthly_contribution_day = Column(Integer, default=1)
    monthly_withdrawal_amount = Column(Float, default=0)
    emergency_reserve_amount = Column(Float, default=0)
    cash_buffer_pct = Column(Float, default=5.0)

    # Time Horizon & Goals
    investment_horizon_years = Column(Integer, default=5)
    primary_goal = Column(String, default="wealth_growth")
    target_corpus = Column(Float)
    target_date = Column(Date)
    drawdown_tolerance_pct = Column(Float, default=20.0)

    # Risk & Strategy (Global Defaults)
    overall_risk_appetite = Column(Integer, default=50)
    max_single_position_pct = Column(Float, default=20.0)
    max_sector_concentration_pct = Column(Float, default=40.0)
    min_score_threshold = Column(Float, default=50.0)
    rebalance_sensitivity = Column(Integer, default=50)
    max_daily_loss_pct = Column(Float, default=5.0)

    # Factor Tilts
    factor_tilt_momentum = Column(Float, default=1.0)
    factor_tilt_value = Column(Float, default=1.0)
    factor_tilt_quality = Column(Float, default=1.0)
    factor_tilt_low_volatility = Column(Float, default=1.0)

    # Scoring Weight Preferences
    weight_technical = Column(Float, default=0.50)
    weight_fundamental = Column(Float, default=0.30)
    weight_sentiment = Column(Float, default=0.20)
    allow_contrarian_signals = Column(Boolean, default=True)

    # Tax Profile
    annual_income_slab = Column(String, default="30pct")
    filing_status = Column(String, default="individual")
    tax_optimization_aggressiveness = Column(Integer, default=50)
    auto_harvest_losses = Column(Boolean, default=True)
    prefer_ltcg_exemption_usage = Column(Boolean, default=True)
    tax_patience_window_days = Column(Integer, default=45)

    # Behavioral & Notification
    notification_channel = Column(String, default="telegram")
    notification_frequency = Column(String, default="real_time")
    notification_types_enabled = Column(String, default="trades,exits,rebalance_warnings,tax_events")
    quiet_hours_start = Column(Time, default=time(22, 0))
    quiet_hours_end = Column(Time, default=time(7, 0))
    timezone = Column(String, default="Asia/Kolkata")
    language_preference = Column(String, default="en")
    require_confirmation_above = Column(Float)

    # ESG Filters
    exclude_sectors = Column(Text)                              # JSON list
    exclude_tickers = Column(Text)                              # JSON list
    require_esg_rating_min = Column(Float)
    shariah_compliant_only = Column(Boolean, default=False)

    # Performance & Reporting
    benchmark_preference = Column(String, default="NIFTY50")
    report_delivery_frequency = Column(String, default="weekly")
    report_format = Column(String, default="dashboard_only")
    show_after_tax_returns = Column(Boolean, default=True)

    # System Configuration
    is_paper_trading = Column(Boolean, default=True)
    paper_trading_start_date = Column(Date)
    real_trading_allowed = Column(Boolean, default=False)
    kill_switch_active = Column(Boolean, default=False)

    # Experimental
    allows_leverage = Column(Boolean, default=False)
    allows_shorting = Column(Boolean, default=False)
    allows_derivatives = Column(Boolean, default=False)
    ml_model_version_preference = Column(String, default="latest")
    backtest_profile_id = Column(Integer)


# ──────────────────────────────────────────────────────────
# §4.8 — Per-Asset-Class Preferences
# ──────────────────────────────────────────────────────────

class UserAssetClassPreference(Base):
    __tablename__ = "user_asset_class_preferences"
    __table_args__ = (UniqueConstraint("user_id", "asset_class_id"),)

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    asset_class_id = Column(Integer, ForeignKey("asset_classes.id"), nullable=False)
    is_enabled = Column(Boolean, default=True)
    risk_appetite = Column(Integer, default=50)

    # PID gains (computed from risk_appetite, not user-set)
    kp = Column(Float, default=1.0)
    ki = Column(Float, default=0.5)
    kd = Column(Float, default=0.5)

    # Constraints
    trailing_sl_pct = Column(Float, default=10.0)
    max_class_allocation_pct = Column(Float, default=100.0)
    min_class_allocation_pct = Column(Float, default=0.0)
    rebalance_frequency_override = Column(Integer)

    # Tax behavior
    tax_patience_enabled = Column(Boolean, default=True)
    loss_harvesting_enabled = Column(Boolean, default=True)


# ──────────────────────────────────────────────────────────
# §4.9 — Wallets, Portfolios, Trading
# ──────────────────────────────────────────────────────────

class Wallet(Base):
    __tablename__ = "wallets"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    currency = Column(String, default="INR")
    balance = Column(Float, default=0.0)
    updated_at = Column(DateTime)


class Portfolio(Base):
    __tablename__ = "portfolios"
    __table_args__ = (UniqueConstraint("user_id", "asset_id", "is_paper"),)

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    quantity = Column(Float, nullable=False)
    avg_price = Column(Float, nullable=False)
    date_entered = Column(Date, nullable=False)
    trailing_sl = Column(Float)
    is_paper = Column(Boolean, default=True)


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    action = Column(String, nullable=False)                     # 'BUY', 'SELL'
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    combined_score = Column(Float)
    signal_reasons = Column(Text)                               # JSON
    is_paper = Column(Boolean, default=True)
    pnl = Column(Float)
    # Tax fields
    tax_type = Column(String)                                   # 'STCG', 'LTCG', 'FLAT_30', 'SLAB'
    tax_amount = Column(Float)
    lot_ids_sold = Column(Text)                                 # JSON
    stt_paid = Column(Float)
    stamp_duty_paid = Column(Float)
    transaction_costs = Column(Float)


class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    trade_id = Column(Integer, ForeignKey("trades.id"))
    message = Column(Text, nullable=False)
    sent_at = Column(DateTime)
    read_at = Column(DateTime)
    action_taken = Column(String)


# ──────────────────────────────────────────────────────────
# §4.10 — Tax Tables
# ──────────────────────────────────────────────────────────

class TaxRegime(Base):
    __tablename__ = "tax_regimes"

    id = Column(Integer, primary_key=True)
    country = Column(String, nullable=False)
    asset_class_id = Column(Integer, ForeignKey("asset_classes.id"))
    regime_name = Column(String, nullable=False)
    # Capital Gains
    stcg_rate = Column(Float)
    ltcg_rate = Column(Float)
    ltcg_holding_days = Column(Integer)
    ltcg_annual_exemption = Column(Float)
    flat_tax_rate = Column(Float)
    # Loss Rules
    loss_offset_allowed = Column(Boolean)
    stcl_offsets = Column(String)
    ltcl_offsets = Column(String)
    loss_carryforward_years = Column(Integer)
    wash_sale_days = Column(Integer, default=0)
    # Transaction Costs
    tds_rate = Column(Float)
    stt_buy_rate = Column(Float)
    stt_sell_rate = Column(Float)
    stamp_duty_rate = Column(Float)
    exchange_txn_charge_rate = Column(Float)
    sebi_turnover_fee_rate = Column(Float)
    gst_rate = Column(Float, default=0.18)
    # Validity
    effective_from = Column(Date)
    effective_to = Column(Date)


class UserTaxLedger(Base):
    __tablename__ = "user_tax_ledger"
    __table_args__ = (UniqueConstraint("user_id", "fiscal_year"),)

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    fiscal_year = Column(String, nullable=False)                # '2025-26'
    # Realized gains/losses
    realized_stcg = Column(Float, default=0)
    realized_ltcg = Column(Float, default=0)
    realized_stcl = Column(Float, default=0)
    realized_ltcl = Column(Float, default=0)
    # Exemptions
    ltcg_exemption_used = Column(Float, default=0)
    ltcg_exemption_remaining = Column(Float, default=125000)
    # Tax estimate
    estimated_tax_liability = Column(Float, default=0)
    stt_paid_ytd = Column(Float, default=0)
    stamp_duty_paid_ytd = Column(Float, default=0)
    total_transaction_costs_ytd = Column(Float, default=0)
    # Carried forward
    carried_forward_stcl = Column(Float, default=0)
    carried_forward_ltcl = Column(Float, default=0)
    cf_stcl_by_year = Column(Text)                              # JSON
    cf_ltcl_by_year = Column(Text)                              # JSON
    # Crypto-specific
    crypto_realized_gains = Column(Float, default=0)
    crypto_tds_paid = Column(Float, default=0)
    #
    last_updated = Column(DateTime)


class LotTracking(Base):
    __tablename__ = "lot_tracking"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    buy_trade_id = Column(Integer, ForeignKey("trades.id"))
    quantity = Column(Float, nullable=False)
    buy_price = Column(Float, nullable=False)
    buy_date = Column(Date, nullable=False)
    remaining_qty = Column(Float, nullable=False)
    unrealized_gain = Column(Float)
    # holding_days and tax_status computed in Python, not SQL generated columns
    # (SQLite generated columns have limitations with expressions)
