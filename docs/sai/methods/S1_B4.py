"""S1_B4 predeclared: 16 signal/pairs/prior/density feature cells x HistGB (iterations,leaves,l2) {(20,7,10),(20,15,100),(60,7,100),(20,3,100)} x correction strengths {0,.1,.25,.5,1}:320 cells. Base win model/memory remain exact B4. Smoke24 dev episodes before full grid; audit excluded."""
from _w4_b4_learners import run

if __name__ == "__main__":
    run("S1_B4")
