"""
NIFTY50 ticker list and metadata for seeding the database.
Tickers use yfinance format (.NS suffix for NSE).
Sector/industry data is approximate — will be refined from yfinance info.

Last updated: October 2026
"""

NIFTY50_TICKERS = [
    # Format: (ticker, name, sector, industry)
    ("RELIANCE.NS",     "Reliance Industries",         "Energy",           "Oil & Gas Refining"),
    ("TCS.NS",          "Tata Consultancy Services",   "Technology",       "IT Services"),
    ("HDFCBANK.NS",     "HDFC Bank",                   "Financial",        "Banks"),
    ("INFY.NS",         "Infosys",                     "Technology",       "IT Services"),
    ("ICICIBANK.NS",    "ICICI Bank",                  "Financial",        "Banks"),
    ("BHARTIARTL.NS",   "Bharti Airtel",               "Telecom",          "Telecom Services"),
    ("SBIN.NS",         "State Bank of India",         "Financial",        "Banks"),
    ("ITC.NS",          "ITC",                         "FMCG",             "Tobacco & FMCG"),
    ("HINDUNILVR.NS",   "Hindustan Unilever",          "FMCG",             "Personal Products"),
    ("LT.NS",           "Larsen & Toubro",             "Industrials",      "Construction & Engineering"),
    ("KOTAKBANK.NS",    "Kotak Mahindra Bank",         "Financial",        "Banks"),
    ("BAJFINANCE.NS",   "Bajaj Finance",               "Financial",        "Consumer Finance"),
    ("AXISBANK.NS",     "Axis Bank",                   "Financial",        "Banks"),
    ("ASIANPAINT.NS",   "Asian Paints",                "Materials",        "Paints"),
    ("MARUTI.NS",       "Maruti Suzuki",               "Automobile",       "Auto Manufacturers"),
    ("SUNPHARMA.NS",    "Sun Pharma",                  "Healthcare",       "Pharmaceuticals"),
    ("TITAN.NS",        "Titan Company",               "Consumer",         "Jewellery & Watches"),
    ("WIPRO.NS",        "Wipro",                       "Technology",       "IT Services"),
    ("ULTRACEMCO.NS",   "UltraTech Cement",            "Materials",        "Cement"),
    ("HCLTECH.NS",      "HCL Technologies",            "Technology",       "IT Services"),
    ("NTPC.NS",         "NTPC",                        "Energy",           "Power Generation"),
    ("POWERGRID.NS",    "Power Grid Corp",             "Energy",           "Power Transmission"),
    ("NESTLEIND.NS",    "Nestle India",                "FMCG",             "Food Products"),
    ("TMPV.NS",         "Tata Motors Passenger Vehicles", "Automobile",    "Auto Manufacturers"),
    ("BAJAJFINSV.NS",   "Bajaj Finserv",               "Financial",        "Financial Services"),
    ("M&M.NS",          "Mahindra & Mahindra",         "Automobile",       "Auto Manufacturers"),
    ("ONGC.NS",         "ONGC",                        "Energy",           "Oil & Gas Exploration"),
    ("TATASTEEL.NS",    "Tata Steel",                  "Materials",        "Steel"),
    ("TECHM.NS",        "Tech Mahindra",               "Technology",       "IT Services"),
    ("JSWSTEEL.NS",     "JSW Steel",                   "Materials",        "Steel"),
    ("ADANIENT.NS",     "Adani Enterprises",           "Conglomerate",     "Diversified"),
    ("ADANIPORTS.NS",   "Adani Ports",                 "Industrials",      "Port Operations"),
    ("COALINDIA.NS",    "Coal India",                  "Energy",           "Mining"),
    ("BPCL.NS",         "Bharat Petroleum",            "Energy",           "Oil & Gas Refining"),
    ("GRASIM.NS",       "Grasim Industries",           "Materials",        "Cement & Textiles"),
    ("DRREDDY.NS",      "Dr Reddy's Laboratories",     "Healthcare",       "Pharmaceuticals"),
    ("CIPLA.NS",        "Cipla",                       "Healthcare",       "Pharmaceuticals"),
    ("APOLLOHOSP.NS",   "Apollo Hospitals",            "Healthcare",       "Hospitals"),
    ("HEROMOTOCO.NS",   "Hero MotoCorp",               "Automobile",       "Two Wheelers"),
    ("EICHERMOT.NS",    "Eicher Motors",               "Automobile",       "Two Wheelers"),
    ("DIVISLAB.NS",     "Divis Laboratories",          "Healthcare",       "Pharmaceuticals"),
    ("SBILIFE.NS",      "SBI Life Insurance",          "Financial",        "Insurance"),
    ("HDFCLIFE.NS",     "HDFC Life Insurance",         "Financial",        "Insurance"),
    ("BRITANNIA.NS",    "Britannia Industries",        "FMCG",             "Food Products"),
    ("TATACONSUM.NS",   "Tata Consumer Products",      "FMCG",             "Food & Beverages"),
    ("INDUSINDBK.NS",   "IndusInd Bank",               "Financial",        "Banks"),
    ("HINDALCO.NS",     "Hindalco Industries",         "Materials",        "Aluminium"),
    ("BAJAJ-AUTO.NS",   "Bajaj Auto",                  "Automobile",       "Two Wheelers"),
    ("WIPRO.NS",        "Wipro",                       "Technology",       "IT Services"),  # duplicate guard handled in seed
    ("SHRIRAMFIN.NS",   "Shriram Finance",             "Financial",        "NBFC"),
    ("BEL.NS",          "Bharat Electronics",          "Industrials",      "Defence Electronics"),
    ("TRENT.NS",        "Trent",                       "Consumer",         "Retail"),
]

# Tickers no longer listed under this symbol. The seed marks existing rows
# is_active=False and keeps their price history. Never delete these rows.
#   TATAMOTORS.NS: renamed TMPV.NS after the Oct 2025 demerger. The CV arm
#   listed as TMCV.NS on 2025-11-12 and was excluded from NIFTY50 on 2025-11-17.
RETIRED_TICKERS = [
    "TATAMOTORS.NS",
]

# Crypto tickers for future Phase (schema seeded now, data later)
CRYPTO_TICKERS = [
    ("BTC-USD",   "Bitcoin",   None, None),
    ("ETH-USD",   "Ethereum",  None, None),
    ("SOL-USD",   "Solana",    None, None),
]
