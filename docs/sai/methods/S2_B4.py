"""S2_B4 predeclared:16 feature cells x CatBoost iterations{25,75},depth{3,5},l2=10 or LightGBM iterations{25,75},leaves{7,15},lambda=10 x strengths{0,.1,.25,.5,1}:640 cells. Both optional runtimes already installed in isolated folder. Exact B4 base; smoke each family24 dev episodes; audit excluded."""
from _w4_b4_learners import run

if __name__ == "__main__":
    run("S2_B4")
