# Sai prior and combination findings

Pinned source: `origin/sai/combo-top10` at `a373345749af89612d0d8571ff1657dcca949ad9`, especially [`docs/sai/COMBOS_TOP10.md`](https://github.com/avyukthNarra/TamuDatathon26-PSAZZ/blob/a373345749af89612d0d8571ff1657dcca949ad9/docs/sai/COMBOS_TOP10.md).

R0 PairMixer at `68e5bd6` matches v4 at `39e5757` except for a log string. The already-tested v4 results do not support repeating a population lag-pair prior, type-level win-cell shrinkage, peer pooling, win-logit recalibration, or gated blending as the next experiment. Reported v4 changes were +0.05 for the pair prior and approximately 0.00 for type shrinkage and peer pooling; logit recalibration was noise, while the gated +0.52 came from including Mixer v4. The peer analysis says episodes share changepoints, not reusable machine rules. The source reports a cross-fit prior score of 55.35 versus 55.34 in-sample, but that does not establish absence of generalization risk.

Keep the jackpot references distinct: the document's jackpot error/reference ratio .149 and 69% win-probability diagnosis describe the older v3 55.90 run; the .149→.121 ratio describes short-history performance changing from v3 to v4. Sai's dev fold also differs from this report's fixed tuning fold. Neither pair of figures is a matched control for our equal-episode jackpot scores.
