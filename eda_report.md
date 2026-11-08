# Fresh-start EDA

All measurements below were produced by `eda_analysis.py`; the complete machine-level and pair-level data is in `eda_results.json`.

## Base rates and target size distributions

| machine type / signal | n | p0 | p(win) | positive payout shares (1,2,5,20) |
|---|---:|---:|---:|---|
| balanced/-1 | 80 | 0.5867 | 0.4133 | 0.3784, 0.4324, 0.1622, 0.0270 |
| balanced/0 | 71 | 0.4947 | 0.5053 | 0.3831, 0.4291, 0.1609, 0.0268 |
| balanced/1 | 83 | 0.4944 | 0.5056 | 0.3621, 0.4437, 0.1664, 0.0277 |
| jackpot/-1 | 75 | 0.8275 | 0.1725 | 0.2380, 0.2807, 0.3610, 0.1203 |
| jackpot/0 | 102 | 0.7568 | 0.2432 | 0.2343, 0.2821, 0.3627, 0.1209 |
| jackpot/1 | 88 | 0.6876 | 0.3124 | 0.2475, 0.2773, 0.3565, 0.1188 |
| steady/-1 | 92 | 0.3865 | 0.6135 | 0.6078, 0.3138, 0.0769, 0.0016 |
| steady/0 | 86 | 0.4695 | 0.5305 | 0.6115, 0.3108, 0.0761, 0.0016 |
| steady/1 | 91 | 0.3575 | 0.6425 | 0.6029, 0.3177, 0.0778, 0.0016 |

Positive-size vectors have 6 distinct clusters after rounding each conditional share to 3 decimals; cluster sizes are in `eda_results.json`.

## Lag effects

Each row uses every available within-machine history transition before the validation cutoff. MI is between the categorical lag (or lag-pair) payout and binary win; win-rate difference is max conditional win rate minus min conditional win rate. A sign conflict means both positive and negative ordinal trends occur among machines of that type.

| lag(s) | machines | MI | win-rate difference | sign conflict by type |
|---|---:|---:|---:|---|
| 1 | 755 | 0.016758 | 0.1856 | balanced, jackpot, steady |
| 2 | 755 | 0.016563 | 0.1841 | balanced, jackpot, steady |
| 3 | 757 | 0.015696 | 0.1775 | balanced, jackpot, steady |
| 4 | 753 | 0.017944 | 0.1869 | balanced, jackpot, steady |
| 5 | 752 | 0.015362 | 0.1715 | balanced, jackpot, steady |
| 6 | 753 | 0.016689 | 0.1864 | balanced, jackpot, steady |
| 7 | 751 | 0.018639 | 0.1922 | balanced, jackpot, steady |
| 8 | 752 | 0.016058 | 0.1811 | balanced, jackpot, steady |
| (1, 2) | 752 | 0.049261 | 0.4737 | balanced, jackpot, steady |
| (1, 3) | 752 | 0.053852 | 0.5107 | balanced, jackpot, steady |
| (1, 4) | 752 | 0.052067 | 0.4891 | balanced, jackpot, steady |
| (1, 5) | 752 | 0.049155 | 0.4837 | balanced, jackpot, steady |
| (1, 6) | 753 | 0.050152 | 0.4722 | balanced, jackpot, steady |
| (1, 7) | 753 | 0.053916 | 0.5129 | balanced, jackpot, steady |
| (1, 8) | 752 | 0.050548 | 0.4987 | balanced, jackpot, steady |
| (2, 3) | 751 | 0.048656 | 0.4623 | balanced, jackpot, steady |
| (2, 4) | 751 | 0.051349 | 0.4905 | balanced, jackpot, steady |
| (2, 5) | 752 | 0.054364 | 0.4757 | balanced, jackpot, steady |
| (2, 6) | 752 | 0.051869 | 0.4939 | balanced, jackpot, steady |
| (2, 7) | 751 | 0.052209 | 0.4965 | balanced, jackpot, steady |
| (2, 8) | 752 | 0.051723 | 0.5017 | balanced, jackpot, steady |
| (3, 4) | 751 | 0.050088 | 0.4759 | balanced, jackpot, steady |
| (3, 5) | 750 | 0.048456 | 0.4752 | balanced, jackpot, steady |
| (3, 6) | 752 | 0.049466 | 0.4805 | balanced, jackpot, steady |
| (3, 7) | 750 | 0.052016 | 0.4984 | balanced, jackpot, steady |
| (3, 8) | 752 | 0.049920 | 0.4797 | balanced, jackpot, steady |
| (4, 5) | 750 | 0.051119 | 0.4882 | balanced, jackpot, steady |
| (4, 6) | 750 | 0.052115 | 0.4881 | balanced, jackpot, steady |
| (4, 7) | 752 | 0.057615 | 0.5181 | balanced, jackpot, steady |
| (4, 8) | 749 | 0.052123 | 0.4917 | balanced, jackpot, steady |
| (5, 6) | 750 | 0.049716 | 0.4957 | balanced, jackpot, steady |
| (5, 7) | 749 | 0.050968 | 0.4931 | balanced, jackpot, steady |
| (5, 8) | 749 | 0.049845 | 0.4839 | balanced, jackpot, steady |
| (6, 7) | 748 | 0.052654 | 0.5018 | balanced, jackpot, steady |
| (6, 8) | 748 | 0.051005 | 0.4922 | balanced, jackpot, steady |
| (7, 8) | 746 | 0.053027 | 0.5026 | balanced, jackpot, steady |

## Non-stationarity, correlation, and missingness

Change points count adjacent observed ticks where binary win/loss changes; it is descriptive rather than a fitted detector. Missing rows are `cutoff - observed history rows`.

| condition | first win rate | second win rate | change | mean change points | mean missing | mean observed | cross-machine corr |
|---|---:|---:|---:|---:|---:|---:|---:|
| drift | 0.0782 | 0.0793 | 0.0011 | 31.57 | 0.00 | 226.21 | -0.0019 |
| shared_change | 0.0806 | 0.0815 | 0.0009 | 32.19 | 0.00 | 224.79 | 0.0021 |
| short_history | 0.0804 | 0.0737 | -0.0067 | 5.50 | 154.79 | 64.00 | -0.0016 |
| stable | 0.0745 | 0.0756 | 0.0011 | 29.55 | 0.00 | 221.42 | 0.0036 |

## Validation score ceilings

The first ceiling gives each target's true total win probability but uses train empirical positive-payout shares. The second gives the train empirical win rate but uses each target's true positive-payout shares.

| construction | points | R | condition ratios (stable, short_history, drift, shared_change) |
|---|---:|---:|---|
| true_win_empirical_size | 56.1258 | 0.0690 | 0.0686, 0.0650, 0.0798, 0.0628 |
| empirical_win_true_size | 30.3025 | 0.9800 | 0.9332, 1.0499, 0.9947, 0.9423 |
