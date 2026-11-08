# h2: independent verification of g5's size-atom rule (Haiku, 23:10)
Own implementation: `verify.py`, `results.json`.

| Variant | Points | stable | short | drift | shared |
|---|---:|---:|---:|---:|---:|
| v6 | 56.29 | .0541 | .1047 | .0488 | .0564 |
| Claim: target atoms, L ∈ {1,2,4} | 57.03 | .0436 | .0844 | .0379 | .0424 |
| Target atoms, L ∈ 1..8 | 56.74 | .0482 | .0906 | .0441 | .0471 |
| **Train-EM atoms, L ∈ {1,2,4}** (no validation targets) | **57.02** | .0438 | .0844 | .0382 | .0426 |

- Train EM recovers lags {1,2,4} (weight ≈ .99).
- Episode halves:
  - even: v6 56.93 → 57.26;
  - odd: v6 55.69 → 56.79.
