"""A1_B4: alpha grid {8,24,64,128} for the size split atop B4's win model (type prior from train, and self prior as in B4).
Pre-declared: prior {self (B4's), type (train)} x alpha {8,24,64,128} = 8 cells; cell 0 = B4 repro (self, 8). hl 64. """
import itertools
import _b4size as b
cells = [dict(mode="ridge", prior=pr, alpha=a, hl=64) for pr, a in itertools.product(("self", "type"), (8, 24, 64, 128))]
if __name__ == "__main__":
    b.run_b4("A1_B4", cells, "alpha grid on size split atop B4")
