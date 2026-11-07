"""S5_B4 predeclared:16 feature cells x standardized Ridge alpha{10,100,1000,10000} x strengths{0,.1,.25,.5,1}:320 cells. Six whole-episode OOF folds on72 dev episodes only. Selected fixed-hyper nested blend diagnostic separates learner/weight fits; global selection remains optimistic. Exact B4 base; smoke24 dev episodes first; never audit fit/score."""
from _w4_b4_learners import run

if __name__ == "__main__":
    run("S5_B4")
