"""
Walk-forward forward-return evaluation harness.

    loader.py       the only DB access (universe, one asset's daily bars and corporate
                    actions, DB fingerprint)
    panel.py        build_panel / build_dataset: (date, ticker) features + separate labels;
                    universe="index" adds point-in-time IN_UNIVERSE
    labels.py       price labels FWD_RET_h / FWD_CC_RET_h / FWD_OO_RET_h and their total-return
                    versions FWD_TR_* (dividends on ex-dates); split-spanning periods NaN
    baselines.py    momentum, 12-1 momentum, reversal, low-vol
    walkforward.py  expanding-train / 6-month-test splits with embargo
    stats.py        Newey-West, BH q-values, Spearman
    metrics.py      IC, quintiles, turnover, long-only costs
    evaluate.py     every signal × horizon × subset → metrics table (PRIMARY_RETURN = "tr")
    report.py       markdown report + CSV (python main.py evaluate)
"""
