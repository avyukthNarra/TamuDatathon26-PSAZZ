"""A8: leave-one-out episode / episode-type shrinkage. Size prior = (1-lam)*train type prior + lam*peer share,
peer share = (pooled hl-64 weighted payout counts of the OTHER machines in the episode [scope all] or of same-type
peers [scope type] + 8*type prior) normalised; the query machine is excluded (leave-one-out), peers use their own
ticks < cutoff only.

Pre-declared grid: scope in {all, type} x lam in {0,0.25,0.5,0.75,1} x alpha in {8,24,64}, ridge mode (27 cells,
lam 0 not duplicated across scopes), plus freq mode for scope type, lam in {0.5,1}, alpha 8.
"""
import itertools
import _freq as f

BASE = [dict(mode="ridge", prior="self", alpha=8, hl=64), dict(mode="ridge", prior="type", alpha=8, hl=64), dict(mode="freq", prior="type", alpha=8, hl=64)]
cells = list(BASE)
for sc, lam, a in itertools.product(("all", "type"), (0.25, 0.5, 0.75, 1.0), (8, 24, 64)):
    cells.append(dict(mode="ridge", prior="type", alpha=a, hl=64, a8=dict(scope=sc, lam=lam)))
for a in (24, 64):
    cells.append(dict(mode="ridge", prior="type", alpha=a, hl=64))
for lam in (0.5, 1.0):
    cells.append(dict(mode="freq", prior="type", alpha=8, hl=64, a8=dict(scope="type", lam=lam)))

if __name__ == "__main__":
    f.run_ids("A8", cells, "leave-one-out episode shrinkage")
