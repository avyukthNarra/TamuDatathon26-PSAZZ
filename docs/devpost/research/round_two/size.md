# Round-two size lags

**Verdict: YES. The shipped size split will adapt to the new lags.**

Size given a win is still `s XOR 1{payout(t−L) ≥ 2}`. The lag is no longer taken from `{1, 2, 4}`. It is the smaller member of the machine’s win pair, so the live set is `{1, 2, 3}`:

| Win pair | Size lag L | Query atom recovered |
|---|---|---|
| (1, 4) | 1 | 213/229 = 0.930 |
| (2, 6) | 2 | 226/254 = 0.890 |
| (3, 8) | 3 | 231/244 = 0.947 |
| all three | smaller member | 670/727 = 0.922 |

`L = 4` is the old partner of lag 1 and is not a size lag in this round. Indicators `payout ≥ 5` and `payout > 0` do not recover the query atom the way `≥ 2` does.

## How this was scored

Kit: `adaptation-participants/public/validation` (768 queries, one per machine). For every query the conditional size is `(p_1, p_2, p_5, p_20) / (1 − p_0)` from `targets.csv`. All 768 vectors match one of two atoms for that machine type (L1 distance to the nearer atom ≤ 1.2×10⁻⁶). There is no third mode and no 50/50 mixture at the cutoff.

History wins with `tick < cutoff` and payout in `{1, 2, 5, 20}` score a candidate rule. Polarity `s ∈ {0, 1}` is the one with higher likelihood. The rule then predicts the atom at the cutoff from `payout(cutoff − L)`, and that prediction is checked against the target atom. A lag tick was present at the cutoff for every machine in the table above. 727/768 machines had a win pair in `{(1,4), (2,6), (3,8)}` with both XOR cells at least 8 ticks and at least 8 wins. The other 41 are short histories (31 machines have fewer than 8 wins).

Same rule by condition (727 machines):

| Condition | Match |
|---|---|
| stable | 181/185 = 0.978 |
| drift | 181/191 = 0.948 |
| short_history | 145/159 = 0.912 |
| shared_change | 163/192 = 0.849 |

`shared_change` is the weak cell: one episode-wide flip makes a single polarity wrong for part of the history. It is still far above a coin flip.

Same rule by machine type:

| Type | (1, 4) → L=1 | (2, 6) → L=2 | (3, 8) → L=3 |
|---|---|---|---|
| steady | 56/59 = 0.949 | 68/73 = 0.932 | 66/69 = 0.957 |
| balanced | 86/89 = 0.966 | 92/101 = 0.911 | 100/102 = 0.980 |
| jackpot | 71/81 = 0.877 | 66/80 = 0.825 | 65/73 = 0.890 |

Jackpot is the noisiest because its two atoms are closer (L1 0.43, versus 0.57 balanced and 0.60 steady), so a finite win history identifies `s` less cleanly.

## Fraction of machines explained per lag

Denominator: 737 machines with at least 8 wins. For indicator `≥ 2`, each machine is assigned the lag in `1..8` with the highest history likelihood. “Explained” means that lag’s fitted polarity also predicts the target atom.

| L | MLE share | Explained (MLE and query match) |
|---|---|---|
| 1 | 181/737 = 0.246 | 165/737 = 0.224 |
| 2 | 165/737 = 0.224 | 148/737 = 0.201 |
| 3 | 190/737 = 0.258 | 180/737 = 0.244 |
| 4 | 48/737 = 0.065 | 30/737 = 0.041 |
| 5 | 38/737 = 0.052 | 23/737 = 0.031 |
| 6 | 45/737 = 0.061 | 25/737 = 0.034 |
| 7 | 42/737 = 0.057 | 24/737 = 0.033 |
| 8 | 28/737 = 0.038 | 19/737 = 0.026 |

`{1, 2, 3}` together: 493/737 = 0.669 explained. `{4, 5, 6, 7, 8}` together: 121/737 = 0.164, and when one of those wins the likelihood it predicts the query atom only about 60% of the time. When `L ∈ {1, 2, 3}` wins, the query match is 0.90–0.95.

Per type, the MLE lag under `≥ 2` (same 737) splits across 1, 2, and 3 and leaves the old lag 4:

| Type | n | L=1 | L=2 | L=3 | L=4 | L=5..8 |
|---|---|---|---|---|---|---|
| steady | 207 | 0.285 | 0.285 | 0.280 | 0.039 | 0.111 |
| balanced | 294 | 0.238 | 0.201 | 0.306 | 0.061 | 0.194 |
| jackpot | 236 | 0.220 | 0.199 | 0.178 | 0.093 | 0.309 |

Holding the lag fixed and ignoring which lag is the MLE, `≥ 2` predicts the query atom for 0.697 (L=1), 0.688 (L=2), 0.735 (L=3), and 0.56–0.57 for L=4..8. That is what a three-way split looks like: about one third of machines truly use a given lag (match ≈ 0.92) and the other two thirds land near chance.

## Indicator

On the 727 machines with a known pair, likelihood at the smaller lag prefers:

| Indicator | Best at that lag | And the prediction matches the query atom |
|---|---|---|
| `payout ≥ 2` | 459/727 = 0.631 | 441/459 = 0.961 |
| `payout ≥ 5` | 142/727 = 0.195 | 105/142 = 0.739 |
| `payout > 0` | 126/727 = 0.173 | 92/126 = 0.730 |

Raw L1 between the two empirical size distributions is slightly higher for `≥ 5` (about 0.45–0.50 vs 0.43–0.45 for `≥ 2`). That gap is the small `≥ 5` bucket, not a better rule: `≥ 5` does not predict the held-out target atom.

## Atoms per machine type

Conditional distribution on `(1x, 2x, 5x, 20x)`. `A0` is the atom with less mass on 5x+20x. Counts are validation queries sitting on that atom. Every query is one of these six vectors.

| Type | Atom | n | 1x | 2x | 5x | 20x |
|---|---|---|---|---|---|---|
| balanced | A0 | 166 | 0.499853 | 0.347928 | 0.130473 | 0.021746 |
| balanced | A1 | 129 | 0.214066 | 0.546737 | 0.205026 | 0.034171 |
| jackpot | A0 | 143 | 0.335146 | 0.244946 | 0.314931 | 0.104977 |
| jackpot | A1 | 123 | 0.120787 | 0.323921 | 0.416469 | 0.138823 |
| steady | A0 | 115 | 0.753989 | 0.196808 | 0.048218 | 0.000984 |
| steady | A1 | 92 | 0.455122 | 0.435902 | 0.106796 | 0.002180 |

## Shipped model

The blend’s size split is v13’s, and inside v13 the size numbers are v7c’s.

- `R2/adapted_entry_v17g_rules/members/hooks.py` line 29: `SIZE_MEMBER = "v13_short"`.
- `R2/adapted_entry_v17g_rules/vendor/v13/slot_forecasting/combo.py` lines 3–4 and line 103: `sizes = [x / wb for x in b[1:]]` takes the conditional size from v7c (`b`), after the win log-odds blend.
- `R2/adapted_entry_v17g_rules/vendor/v13/v7c_forecasting/forecast.py` lines 16–18: `rule = SizeRule.learn(queries, history)` on this round’s train. The gate can turn the rule off; it does not pin the lag list.
- `R2/adapted_entry_v17g_rules/vendor/v13/v7c_forecasting/size_rule.py`:
  - line 23: `CANDIDATE_LAGS = tuple(range(1, 9))`
  - lines 34–39: the bit is fixed, `1 if pay[index] >= 2 else 0`
  - lines 42–55 and line 518: `kept = select_lags(mass, lags)` keeps every candidate whose share of train posterior mass exceeds `1/8`

The lags are learned from train. They are not the constant `{1, 2, 4}`. The threshold `≥ 2` is a constant, and it is still the right threshold.

Train, scored with the validation atoms above, shows the same concentration the selector will see (733 machines with ≥8 wins): best rule is `≥ 2` at L=3 for 134, L=1 for 125, L=2 for 112, and L=4 for 13. A refit on this round’s train moves the lag set to `{1, 2, 3}`.

One hardcoded `{1, 2, 4}` list does exist and is not what this entry emits. `vendor/v13/slot_forecasting/size_rule_hmm.py` line 22 sets `CORE = [(2, k, d) for k in (1, 2, 4) ...]`. `strongest.py` lines 56–58 replaces that core with `core_from_train` when train supports other rules (`ALL_RULES` is thresholds 2 and 5, lags 1..8). `combo.combine` then drops the HMM size and writes v7c’s size, so the `CORE` tuple never reaches the submission.

If the train gate rejects `SizeRule`, the fallback v6 head discovers size keys from train over lags 1..8 (`vendor/v13/slot_forecasting/pair_mixer.py` `discover_size_keys`, lines 389–396). That path also moves when the size lags move. Its empty-signal fallback, `MODE_SIZE_KEYS` line 40, is `((), (1,), (2,), (3,), (4,))`, which already contains the new lag 3.
