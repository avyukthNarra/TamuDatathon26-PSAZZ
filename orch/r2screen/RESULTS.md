# Round-two sign-invariant structure screen

Script: `orch/r2screen/walsh_screen.py` (numpy only, about 3 s per kit; run as
`python3 walsh_screen.py <kit_dir>`). Unit of analysis is each (episode, machine)
series, using train+validation+test histories. Per series,
E_m(S) = T c^2 / (pbar(1-pbar)) over the 92 lag subsets of size 1-3 of {1..8}.
Z-score = mean(E-1) / sqrt(2/n). Nothing was committed.

## Kit 1: public (`mainFiles/public`)

Expected: parity pairs (1,3), (2,5), (4,7).

Summary: all three expected pairs are in the top 10 overall, and (4,7), (2,5),
(1,3) together are the argmax for 55% of machines. But the top two overall are
triples ((4,5,7), (2,3,4)) with higher excess than any pair, and the lag-7
singleton (excess 7.7, z 303) is the strongest single lag. Lag 7 is strong on its
own, not only through (4,7). Triples such as (2,3,4) contain none of the three
expected pairs, so the triple excess is not explained by containment. I did not
verify whether this is a real three-lag term or leakage through the sequence
dependence of the process (see open questions).

### Overall (n_machines = 3072)

| rank | subset | size | mean E-1 | z |
|---|---|---|---|---|
| 1 | (4,5,7) | 3 | 18.621 | 729.8 |
| 2 | (2,3,4) | 3 | 18.457 | 723.4 |
| 3 | (1,2,5) | 3 | 18.400 | 721.1 |
| 4 | (1,4,6) | 3 | 18.072 | 708.3 |
| 5 | (2,3,8) | 3 | 17.745 | 695.5 |
| 6 | (2,5) | 2 | 15.451 | 605.5 |
| 7 | (1,3) | 2 | 13.859 | 543.2 |
| 8 | (4,7) | 2 | 13.840 | 542.4 |
| 9 | (2,5,8) | 3 | 8.678 | 340.1 |
| 10 | (1,7,8) | 3 | 8.669 | 339.7 |

Best per size: size 1: (7) mean 7.744 z 303.5; size 2: (2,5) mean 15.451 z 605.5; size 3: (4,5,7) mean 18.621 z 729.8.

### Type steady (n_machines = 1033)

| rank | subset | size | mean E-1 | z |
|---|---|---|---|---|
| 1 | (4,5,7) | 3 | 23.930 | 543.8 |
| 2 | (2,3,4) | 3 | 23.680 | 538.2 |
| 3 | (1,2,5) | 3 | 23.148 | 526.1 |
| 4 | (2,3,8) | 3 | 23.147 | 526.0 |
| 5 | (1,4,6) | 3 | 23.029 | 523.4 |
| 6 | (2,5) | 2 | 19.623 | 446.0 |
| 7 | (4,7) | 2 | 16.950 | 385.2 |
| 8 | (1,3) | 2 | 16.202 | 368.2 |
| 9 | (3,5,6) | 3 | 11.416 | 259.4 |
| 10 | (2,5,8) | 3 | 11.123 | 252.8 |

### Type balanced (n_machines = 1061)

| rank | subset | size | mean E-1 | z |
|---|---|---|---|---|
| 1 | (1,2,5) | 3 | 22.298 | 513.6 |
| 2 | (2,3,4) | 3 | 22.079 | 508.5 |
| 3 | (1,4,6) | 3 | 21.804 | 502.2 |
| 4 | (4,5,7) | 3 | 21.103 | 486.1 |
| 5 | (2,3,8) | 3 | 20.642 | 475.4 |
| 6 | (1,3) | 2 | 17.084 | 393.5 |
| 7 | (2,5) | 2 | 16.534 | 380.8 |
| 8 | (4,7) | 2 | 15.534 | 357.8 |
| 9 | (2,5,8) | 3 | 9.914 | 228.4 |
| 10 | (1,7,8) | 3 | 9.575 | 220.5 |

### Type jackpot (n_machines = 978)

| rank | subset | size | mean E-1 | z |
|---|---|---|---|---|
| 1 | (4,5,7) | 3 | 10.320 | 228.2 |
| 2 | (2,5) | 2 | 9.868 | 218.2 |
| 3 | (1,2,5) | 3 | 9.156 | 202.5 |
| 4 | (2,3,4) | 3 | 9.012 | 199.3 |
| 5 | (2,3,8) | 3 | 8.897 | 196.8 |
| 6 | (1,4,6) | 3 | 8.788 | 194.3 |
| 7 | (4,7) | 2 | 8.716 | 192.7 |
| 8 | (1,3) | 2 | 7.885 | 174.4 |
| 9 | (1,7,8) | 3 | 5.161 | 114.1 |
| 10 | (2,5,8) | 3 | 4.755 | 105.2 |

### Per-machine winner counts (argmax E_m(S); 3043 machines)

| subset | size | wins | share |
|---|---|---|---|
| (4,7) | 2 | 727 | 0.239 |
| (2,5) | 2 | 476 | 0.156 |
| (1,3) | 2 | 444 | 0.146 |
| (4,5,7) | 3 | 258 | 0.085 |
| (2,3,8) | 3 | 225 | 0.074 |
| (2,3,4) | 3 | 191 | 0.063 |
| (1,2,5) | 3 | 171 | 0.056 |
| (1,4,6) | 3 | 157 | 0.052 |
| (1) | 1 | 28 | 0.009 |
| (1,7,8) | 3 | 19 | 0.006 |
| (3,4) | 2 | 19 | 0.006 |
| (3) | 1 | 18 | 0.006 |
| (2) | 1 | 15 | 0.005 |
| (1,5,8) | 3 | 14 | 0.005 |
| (1,4,5) | 3 | 10 | 0.003 |

Singleton excess: (1)=1.177, (2)=1.322, (3)=1.170, (4)=1.411, (5)=0.993, (6)=0.838, (7)=7.744, (8)=0.807.

## Kit 2: simulated round2_shift (`worktrees/r2-transfer/r2/gen_round2_shift`)

Truth from the generator (`mainFiles/eval_suite/generator.py`,
`preset_config("round2_shift")`): `recurring_lags=(2,3,5,7)`, with 1-3 interactions
per machine drawn from single / xor / and / or over those lags only. The `_truth`
folder was not read by the screen.

Result: the screen recovers the true lag set. The four true lags are the top four
singletons ((5), (7), (2), (3); excess 4.1-4.8 overall, z 139-161). The top ten
overall pairs are all within {2,3,5,7}. Non-true lags 1, 4, 6, 8 have singleton
excess of only 0.7-1.1 and never reach the top ten. Triples within the set are
weaker than pairs.

### Overall (n_machines = 2304)

| rank | subset | size | mean E-1 | z |
|---|---|---|---|---|
| 1 | (5) | 1 | 4.751 | 161.2 |
| 2 | (7) | 1 | 4.484 | 152.2 |
| 3 | (2) | 1 | 4.403 | 149.5 |
| 4 | (3) | 1 | 4.092 | 138.9 |
| 5 | (3,5) | 2 | 2.584 | 87.7 |
| 6 | (3,7) | 2 | 2.546 | 86.4 |
| 7 | (2,5) | 2 | 2.371 | 80.5 |
| 8 | (5,7) | 2 | 2.315 | 78.6 |
| 9 | (2,3) | 2 | 2.280 | 77.4 |
| 10 | (2,7) | 2 | 2.167 | 73.6 |

Best per size: size 1: (5) mean 4.751 z 161.2; size 2: (3,5) mean 2.584 z 87.7; size 3: (2,5,7) mean 1.890 z 64.1.

### Type steady (n_machines = 864)

| rank | subset | size | mean E-1 | z |
|---|---|---|---|---|
| 1 | (7) | 1 | 5.332 | 110.8 |
| 2 | (5) | 1 | 4.684 | 97.4 |
| 3 | (2) | 1 | 4.645 | 96.5 |
| 4 | (3) | 1 | 4.059 | 84.4 |
| 5 | (3,5) | 2 | 2.698 | 56.1 |
| 6 | (3,7) | 2 | 2.636 | 54.8 |
| 7 | (2,7) | 2 | 2.342 | 48.7 |
| 8 | (2,5) | 2 | 2.258 | 46.9 |
| 9 | (2,3) | 2 | 2.174 | 45.2 |
| 10 | (5,7) | 2 | 1.990 | 41.4 |

### Type balanced (n_machines = 864)

| rank | subset | size | mean E-1 | z |
|---|---|---|---|---|
| 1 | (5) | 1 | 5.399 | 112.2 |
| 2 | (2) | 1 | 4.885 | 101.5 |
| 3 | (3) | 1 | 4.819 | 100.2 |
| 4 | (7) | 1 | 4.461 | 92.7 |
| 5 | (3,5) | 2 | 2.451 | 50.9 |
| 6 | (5,7) | 2 | 2.406 | 50.0 |
| 7 | (2,5) | 2 | 2.362 | 49.1 |
| 8 | (3,7) | 2 | 2.314 | 48.1 |
| 9 | (2,7) | 2 | 2.190 | 45.5 |
| 10 | (2,3) | 2 | 2.039 | 42.4 |

### Type jackpot (n_machines = 576)

| rank | subset | size | mean E-1 | z |
|---|---|---|---|---|
| 1 | (5) | 1 | 3.879 | 65.8 |
| 2 | (2) | 1 | 3.318 | 56.3 |
| 3 | (7) | 1 | 3.246 | 55.1 |
| 4 | (3) | 1 | 3.049 | 51.7 |
| 5 | (2,3) | 2 | 2.799 | 47.5 |
| 6 | (3,7) | 2 | 2.758 | 46.8 |
| 7 | (5,7) | 2 | 2.665 | 45.2 |
| 8 | (3,5) | 2 | 2.611 | 44.3 |
| 9 | (2,5) | 2 | 2.554 | 43.3 |
| 10 | (2,5,7) | 3 | 2.171 | 36.8 |

### Per-machine winner counts (argmax E_m(S); 2302 machines)

| subset | size | wins | share |
|---|---|---|---|
| (5) | 1 | 221 | 0.096 |
| (2) | 1 | 210 | 0.091 |
| (7) | 1 | 210 | 0.091 |
| (3) | 1 | 205 | 0.089 |
| (3,7) | 2 | 144 | 0.063 |
| (3,5) | 2 | 127 | 0.055 |
| (2,5) | 2 | 124 | 0.054 |
| (5,7) | 2 | 120 | 0.052 |
| (2,7) | 2 | 115 | 0.050 |
| (2,3) | 2 | 105 | 0.046 |
| (2,3,5) | 3 | 21 | 0.009 |
| (2,3,7) | 3 | 21 | 0.009 |
| (3,5,8) | 3 | 18 | 0.008 |
| (2,3,6) | 3 | 16 | 0.007 |
| (3,5,7) | 3 | 14 | 0.006 |

Singleton excess: (1)=0.672, (2)=4.403, (3)=4.092, (4)=1.089, (5)=4.751, (6)=0.965, (7)=4.484, (8)=0.785.

## Open questions

- Public kit triples outrank pairs. Under pure pair parity they should not. Next
  check: regress out the pair terms (or test triples on machines where the pair
  is absent) before concluding that three-lag rules exist.
- Round two could use single or triple lags. The shifted kit shows singles and pairs
  are picked up clearly. Triples inside the true set are weaker, so the screen
  alone may under-rank a pure three-lag parity.
