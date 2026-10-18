"""
NIFTY50 ticker list and metadata for seeding the database.
Tickers use yfinance format (.NS suffix for NSE).
Sector/industry data is approximate — will be refined from yfinance info.

Constituent list as of 2026-10-18 (fetched that day from NSE's
ind_nifty50list.csv, 50 symbols, includes the 2026-09-15 reconstitution).
The CSV carries no effective date. Names, sector and industry are our own labels.
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
    ("GRASIM.NS",       "Grasim Industries",           "Materials",        "Cement & Textiles"),
    ("DRREDDY.NS",      "Dr Reddy's Laboratories",     "Healthcare",       "Pharmaceuticals"),
    ("CIPLA.NS",        "Cipla",                       "Healthcare",       "Pharmaceuticals"),
    ("APOLLOHOSP.NS",   "Apollo Hospitals",            "Healthcare",       "Hospitals"),
    ("EICHERMOT.NS",    "Eicher Motors",               "Automobile",       "Two Wheelers"),
    ("SBILIFE.NS",      "SBI Life Insurance",          "Financial",        "Insurance"),
    ("HDFCLIFE.NS",     "HDFC Life Insurance",         "Financial",        "Insurance"),
    ("TATACONSUM.NS",   "Tata Consumer Products",      "FMCG",             "Food & Beverages"),
    ("HINDALCO.NS",     "Hindalco Industries",         "Materials",        "Aluminium"),
    ("BAJAJ-AUTO.NS",   "Bajaj Auto",                  "Automobile",       "Two Wheelers"),
    ("SHRIRAMFIN.NS",   "Shriram Finance",             "Financial",        "NBFC"),
    ("BEL.NS",          "Bharat Electronics",          "Industrials",      "Defence Electronics"),
    ("TRENT.NS",        "Trent",                       "Consumer",         "Retail"),
    ("BSE.NS",         "BSE",                         "Financial",        "Capital Markets"),
    ("ETERNAL.NS",     "Eternal (Zomato)",            "Consumer",         "Online Food Delivery & Quick Commerce"),
    ("INDIGO.NS",      "InterGlobe Aviation (IndiGo)", "Industrials",     "Airlines"),
    ("JIOFIN.NS",      "Jio Financial Services",      "Financial",        "NBFC"),
    ("MAXHEALTH.NS",   "Max Healthcare Institute",    "Healthcare",       "Hospitals"),
]

# Tickers no longer listed under this symbol. The seed marks existing rows
# is_active=False and keeps their price history. Never delete these rows.
#   TATAMOTORS.NS: renamed TMPV.NS after the Oct 2025 demerger. The CV arm
#   listed as TMCV.NS on 2025-11-12 and was excluded from NIFTY50 on 2025-11-17.
# The rest were dropped from NIFTY50 and may still trade; deactivated only
# because they are no longer index members (history kept).
#   WIPRO.NS: removed in the 2026-09-15 reconstitution.
#   BPCL, BRITANNIA, DIVISLAB, HEROMOTOCO, INDUSINDBK: absent from the
#   2026-10-18 NSE list (reconstitution dates not verified).
RETIRED_TICKERS = [
    "TATAMOTORS.NS",
    "WIPRO.NS",
    "BPCL.NS",
    "BRITANNIA.NS",
    "DIVISLAB.NS",
    "HEROMOTOCO.NS",
    "INDUSINDBK.NS",
]

# Crypto tickers for future Phase (schema seeded now, data later)
CRYPTO_TICKERS = [
    ("BTC-USD",   "Bitcoin",   None, None),
    ("ETH-USD",   "Ethereum",  None, None),
    ("SOL-USD",   "Solana",    None, None),
]
