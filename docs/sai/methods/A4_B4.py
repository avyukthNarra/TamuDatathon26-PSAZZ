"""A4_B4: class-specific half-life/alpha for the size split atop B4 (prior type from train, norm=True).
Pre-declared: cell0 = B4 repro (self prior, hl 64, alpha 8); then hl_frequent(p1,p2) {16,32,64} x hl_rare(p5,p20) {64,128,inf}
x alpha_rare {8,24,64} x alpha_frequent {8,24} = 54 cells (+1 repro = 55)."""
import itertools
import _b4size as b
cells = [dict(mode="ridge", prior="self", alpha=8, hl=64)]
for hf, hr, ar, af in itertools.product((16, 32, 64), (64, 128, 1e9), (8, 24, 64), (8, 24)):
    cells.append(dict(mode="ridge", prior="type", hl=[hf, hf, hr, hr], alpha=[af, af, ar, ar], norm=True))
if __name__ == "__main__":
    b.run_b4("A4_B4", cells, "class-specific half-life/alpha atop B4")
