"""
Walk-forward forward-return evaluation harness.

    loader.py       the only DB access (universe, one asset's daily bars)
    panel.py        build_panel / build_dataset: (date, ticker) features + separate labels
    labels.py       FWD_RET_h (next-open entry) and FWD_CC_RET_h
    baselines.py    momentum, 12-1 momentum, reversal, low-vol
    walkforward.py  expanding-train / 6-month-test splits with embargo
    stats.py        Newey-West, BH q-values, Spearman
    metrics.py      IC, quintiles, turnover, long-only costs
    evaluate.py     every signal × horizon × subset → metrics table
    report.py       markdown report + CSV (python main.py evaluate)
"""
