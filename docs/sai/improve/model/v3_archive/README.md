# Model-level improvements

All new population statistics use **train histories only**. Model selection and
1,000-replicate episode-bootstrap comparisons use **72 dev episodes only**.
No audit/all scoring is performed. Existing v3 size modes are still extracted
from **all validation targets**, as in TM_COMBINED3; the inherited dependency
is explicit in both runners. These experiments do not provide a new blind test.

Run from the repository root, with the already installed NumPy and SciPy:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
nice -n 10 python3 -m unittest discover -s docs/sai/improve/model -v
nice -n 10 python3 docs/sai/improve/model/experiments.py
nice -n 10 python3 docs/sai/improve/model/verify_model.py
```

`experiments.py` reproduces v3's win head in a batch implementation. It obtains
v3's raw sizes from the copied original `pranav_v3.py` and caches them in
`v3_raw.npy`; all experiments preserve these conditional sizes. Numerical
reproduction of the saved v3 baseline is checked before running the grid.
Delete this cache only when intentionally changing the dataset or original
size settings. It contains forecasts, not truth. Fixed seeds are in the source.

`train_structure.json` records train-derived priors and peer null tests.
`results.json` records all 93 candidate rows (31 settings x 3 rule weights),
full dev CIs, paired differences, condition ratios and two-fold cross-fit
selection. M5 cross-fitting reselects M1/M2/M4 settings on the opposite fold;
it does not hold globally selected component settings fixed. Component search
spaces and the decision to combine them still benefit from exploratory dev
work. Bootstraps on globally selected settings do not correct for selection.

`ImprovedPairMixer` in `improved_mixer.py` is a reusable fit/predict model:

```python
from improved_mixer import ImprovedPairMixer
model = ImprovedPairMixer(size_modes=original_v3_size_modes).fit(train_queries, train_history)
forecast = model.predict(queries, history)
```

Its selected M5 defaults use type-specific log pair-prior strength 4, type-level
win-cell pseudo-counts 1 for exactly 64 observations and .25 otherwise, and a
jackpot-only Beta mixture of strength 6 with expert loss learning rate 8.
The jackpot component means are learned from best-pair cell rates on train;
its fixed mixture weights are .45/.45/.10 with a weak middle component.
Rule forecasts remain external: combine .2 rule with .8 improved mixer.

M3's peer pooling did not improve matched type-only shrinkage, so it is excluded
from M5. Jackpot half-life alternatives were inferior to its Beta-mixture fix.
M1's small full-dev improvement did not survive fold selection on its own.

`verify_model.py` checks the reusable model against IMP_M5, runs it twice for
exact determinism, and validates IDs/probabilities for every saved CSV. It does
not score targets. `verification.json` contains errors and prediction hashes.
This is a research model using existing NumPy/SciPy, not a stdlib submission port.
No existing source, Git state, changelog or packaging files were changed.
