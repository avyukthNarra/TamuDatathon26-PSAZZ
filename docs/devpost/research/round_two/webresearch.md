# Three Brier edits on pair-BMA

Pair-BMA is Brier-optimal only if the posterior over the three lag pairs is calibrated. A shared changepoint and a gradual live/dead swap both fold opposite XOR polarities into one likelihood, so the true pair flattens and about 30% of machines stay ambiguous. Fix the window and the polarity, then shrink the two rates.

**1. One shared changepoint, product of evidences.** Grid τ on the last 40 ticks. For each machine, log-marginal-likelihood of the three pairs on post-τ ticks only; add the eight log evidences and softmax over τ. Forecast with pair-BMA fit on post-τ ticks, averaged over that posterior (keep a small “no change” atom). Eight machines identify a τ one cannot, and pre-change bits stop cancelling the true pair.
https://arxiv.org/pdf/1609.06805

**2. Two-state polarity filter; enumerate pairs, do not use particles.** After τ, the hidden state is which XOR level is live, with Bernoulli emissions and a sticky stay probability ρ. Forward-filter each pair; pick ρ in {0.95, 0.98, 0.995} by total log evidence on the episode. Emit the mixture over 3 pairs × 2 states. A swap flips the state and keeps the pair, which static BMA cannot do. 2×3 is closed form (Rao-Blackwell).
https://www.stats.ox.ac.uk/~doucet/doucet_godsill_andrieu_sequentialmontecarloforbayesfiltering.pdf

**3. Shrink p_high and p_low across the eight machines.** Weight each machine’s high-cell and low-cell counts by its pair posterior; method-of-moments a shared mean μ and prior strength κ; emit (κμ + wins) / (κ + n). About twenty trials per cell makes the raw rate’s variance a direct Brier cost (beta-binomial Efron–Morris).
https://carleton-stat340.github.io/notes2022/19-hierarchical-beta-binom

Skip η<1 tempering: it helps only when weights are already over-separated, and it blurs the ambiguous third (https://doi.org/10.1093/biomet/asaf034). Skip stacking: it loses to BMA when the true pair is on the list (https://jmlr.org/papers/volume4/clarke03a/clarke03a.pdf), which it is once (1) and (2) remove the regime misspecification.
