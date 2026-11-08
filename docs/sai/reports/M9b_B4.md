# M9b_B4
- Ran: gate between episode-pooled BOCPD survival-weighted fit and B4 holdout memory (fb=B4 memory); 164 cells on dev: A 140 (pool surv/surv*b3 x h 1/200,1/5000 x pool-only, soft1, hard1, soft2, kofn, plus B4-memory-only ref; lam .05), B 24 (lam .5/2; kofn W80 k2,3 + pool-only); B4 = 50.89 (memory-only ref reproduces 50.89).
- Best {"lam":0.05,"pool":"surv*b3","h":1/5000,"fb":"B4 memory","gate":"kofn","W":80,"k":2}: 51.22 CI [50.22,52.15].
- Condition ratios: stable 0.105, short_history 0.216, drift 0.197 (B4 0.211), shared_change 0.168 (0.183).
- Error shares: p_0 .480 p_1 .254 p_2 .191 p_5 .068 p_20 .007.
- vs B4 standalone +0.327 [+0.143,+0.531]; blend_gain 51.22, vs_base +0.327 [+0.143,+0.531], w 1.0.
- Per-condition paired diff vs B4 (ratio): stable -0.001 [-0.002,0.000], drift -0.014 [-0.029,-0.002], shared_change -0.015 [-0.030,-0.004].
- Sensitivity (pool surv*b3, h 1/5000, lam .05): kofn W80 k=1..4 = 51.08/51.22/51.17/51.02; W40 k=1..4 = 50.79/50.91/50.89/50.89; hard1 th3 51.16. Pool-only ungated: 47.5/47.2 (surv), 50.9/51.1 (surv*b3). lam .5 best 51.18, lam 2 51.00. pool=surv best 50.90.
- Selection: best of 164 dev cells; neighbours (k=3, hard1 th3, lam .5) are 51.16-51.18 so a ~+0.25..0.3 gain is consistent. Runtime ~0.1s per predict after cached posteriors; determinism 0.0; stdlib-only no.
