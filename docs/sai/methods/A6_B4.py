"""A6_B4: prior strength scaled by effective sample size atop B4: alpha_eff = a0*(W/Wmed)^g.
Pre-declared: g {-1,-0.5,0,0.5,1} x a0 {8,24,64} x prior {self, type} = 30 cells; cell0 = B4 repro (self, a0 8, g 0)."""
import itertools
import _b4size as b
cells = [dict(mode="ridge", prior=pr, alpha=a, hl=64, g=g) for pr, g, a in itertools.product(("self", "type"), (0.0, -1, -0.5, 0.5, 1), (8, 24, 64))]
if __name__ == "__main__":
    b.run_b4("A6_B4", cells, "ESS-scaled prior atop B4")
