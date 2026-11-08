import sys, json
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mainFiles"))
from eval_suite.scoring import official_score
W = str(Path(__file__).resolve().parent)  # the r2/ folder holding the generated round-two data
T = f"{W}/gen_round2_shift/_truth/test"
Q = f"{W}/data_round2_shift/test/queries.csv"
for name in sys.argv[1:]:
    pred = f"{W}/{name}/submission.csv"
    r = official_score(pred, f"{T}/targets.csv", Q, f"{T}/conditions.csv", f"{T}/reference.csv", 40)
    print(name, json.dumps({"points": round(r["points"], 3), "R": round(r["R"], 4),
          "ratio": {k: round(v, 4) for k, v in r["ratio"].items()},
          "E": {k: round(v, 5) for k, v in r["E"].items()}, "B": {k: round(v, 5) for k, v in r["B"].items()},
          "n": r["n_episodes"]}))
