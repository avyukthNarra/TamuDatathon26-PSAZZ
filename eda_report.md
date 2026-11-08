# Fresh-start EDA

All measurements below were produced by `eda_analysis.py`; the complete machine-level and pair-level data is in `eda_results.json`.

Win means any payout greater than zero (`1, 2, 5, 20`); only payout zero is a loss.

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
| 1 | 755 | 0.021259 | 0.3698 | balanced, jackpot, steady |
| 2 | 755 | 0.023050 | 0.3865 | balanced, jackpot, steady |
| 3 | 757 | 0.022204 | 0.3802 | balanced, jackpot, steady |
| 4 | 753 | 0.023746 | 0.3679 | balanced, jackpot, steady |
| 5 | 752 | 0.021731 | 0.3656 | balanced, jackpot, steady |
| 6 | 753 | 0.022434 | 0.3763 | balanced, jackpot, steady |
| 7 | 751 | 0.042543 | 0.4214 | balanced, jackpot, steady |
| 8 | 752 | 0.022157 | 0.3727 | balanced, jackpot, steady |
| (1, 2) | 752 | 0.080654 | 0.8788 | balanced, jackpot, steady |
| (1, 3) | 752 | 0.119197 | 0.9174 | balanced, jackpot, steady |
| (1, 4) | 752 | 0.082501 | 0.8809 | balanced, jackpot, steady |
| (1, 5) | 752 | 0.079302 | 0.8866 | balanced, jackpot, steady |
| (1, 6) | 753 | 0.079652 | 0.8771 | balanced, jackpot, steady |
| (1, 7) | 753 | 0.098930 | 0.8991 | balanced, jackpot, steady |
| (1, 8) | 752 | 0.080083 | 0.8901 | balanced, jackpot, steady |
| (2, 3) | 751 | 0.080279 | 0.8856 | balanced, jackpot, steady |
| (2, 4) | 751 | 0.081170 | 0.8901 | balanced, jackpot, steady |
| (2, 5) | 752 | 0.136346 | 0.8912 | balanced, jackpot, steady |
| (2, 6) | 752 | 0.092876 | 0.9017 | balanced, jackpot, steady |
| (2, 7) | 751 | 0.098994 | 0.9032 | balanced, jackpot, steady |
| (2, 8) | 752 | 0.079281 | 0.9019 | balanced, jackpot, steady |
| (3, 4) | 751 | 0.081524 | 0.8813 | balanced, jackpot, steady |
| (3, 5) | 750 | 0.077953 | 0.8892 | balanced, jackpot, steady |
| (3, 6) | 752 | 0.079541 | 0.8769 | balanced, jackpot, steady |
| (3, 7) | 750 | 0.097341 | 0.8974 | balanced, jackpot, steady |
| (3, 8) | 752 | 0.087519 | 0.8937 | balanced, jackpot, steady |
| (4, 5) | 750 | 0.095488 | 0.9032 | balanced, jackpot, steady |
| (4, 6) | 750 | 0.081512 | 0.8843 | balanced, jackpot, steady |
| (4, 7) | 752 | 0.147339 | 0.9196 | balanced, jackpot, steady |
| (4, 8) | 749 | 0.080879 | 0.8851 | balanced, jackpot, steady |
| (5, 6) | 750 | 0.077826 | 0.8832 | balanced, jackpot, steady |
| (5, 7) | 749 | 0.097148 | 0.8984 | balanced, jackpot, steady |
| (5, 8) | 749 | 0.080915 | 0.8902 | balanced, jackpot, steady |
| (6, 7) | 748 | 0.097763 | 0.9072 | balanced, jackpot, steady |
| (6, 8) | 748 | 0.080726 | 0.8936 | balanced, jackpot, steady |
| (7, 8) | 746 | 0.101100 | 0.8948 | balanced, jackpot, steady |

## Non-stationarity, correlation, and missingness

Change points count adjacent observed ticks where binary win/loss changes; it is descriptive rather than a fitted detector. Missing rows are `cutoff - observed history rows`.

| condition | first win rate | second win rate | change | mean change points | mean missing | mean observed | cross-machine corr |
|---|---:|---:|---:|---:|---:|---:|---:|
| drift | 0.4012 | 0.4093 | 0.0081 | 92.30 | 0.00 | 226.21 | 0.0047 |
| shared_change | 0.4006 | 0.4017 | 0.0011 | 91.56 | 0.00 | 224.79 | 0.0088 |
| short_history | 0.3921 | 0.3844 | -0.0076 | 16.86 | 154.79 | 64.00 | 0.0005 |
| stable | 0.4153 | 0.4251 | 0.0099 | 90.40 | 0.00 | 221.42 | 0.0051 |

## Validation score ceilings

The first ceiling gives each target's true total win probability but uses train empirical positive-payout shares. The model-size ceilings use the true win probability with the pooled or recent model's own conditional positive-payout distribution. The empirical-win ceiling uses the train empirical win rate with each target's true positive-payout shares.

| construction | points | R | condition ratios (stable, short_history, drift, shared_change) |
|---|---:|---:|---|
| true_win_empirical_size | 56.1258 | 0.0690 | 0.0686, 0.0650, 0.0798, 0.0628 |
| empirical_win_true_size | 30.3025 | 0.9800 | 0.9332, 1.0499, 0.9947, 0.9423 |
| true_win_pooled_size | 56.5033 | 0.0619 | 0.0498, 0.0748, 0.0588, 0.0642 |
| true_win_recent_size | 55.4426 | 0.0822 | 0.0710, 0.1080, 0.0679, 0.0819 |
