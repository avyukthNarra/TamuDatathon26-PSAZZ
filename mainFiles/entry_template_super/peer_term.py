"""v21 add-on: machines win more when the episode's other machines just won (the "peer" effect).

None of Sai's three members look at other machines, and in v20 the effect only reaches the blend
inside the v19 member (1/4 weight). This adds coef x (peer share - centre) to the blended win
log-odds, where peer share = the other machines' win share over the last 5 ticks before the cutoff
(vendor/v13/slot_forecasting/peers.py) and the centre is the mean share over the split being
forecast. Uses only history.csv / queries.csv, never answers.

Switched on by peer.json ({"coef": c}) written by run_super.py --extra-peer; absent = off (v20).
Evidence for round two (adaptation kit), docs/SUPER_BRANCH.md: the logistic peer coefficient
against a lag-pair baseline is 1.14 [0.65, 1.58] in test history, 0.54 [0.00, 1.13] validation,
0.27 [-0.53, 0.76] train (main round 1.03 [0.35, 1.73]); coef 0.75 on v20 raises both cross-fit
validation halves (38.34 / 38.81 -> 38.38 / 38.87, 40-point scale).
"""
import json
import math
import sys
from pathlib import Path

ENTRY = Path(__file__).resolve().parent
MIN_PROBABILITY = 1e-4


def load_coef():
    path = ENTRY / "peer.json"
    return float(json.loads(path.read_text(encoding="utf-8"))["coef"]) if path.exists() else 0.0


def peer_shares(split_dir):
    sys.path.insert(0, str(ENTRY / "vendor" / "v13"))
    from slot_forecasting.io import load_inputs
    from slot_forecasting.peers import _by_machine, peer_share
    queries, history = load_inputs(split_dir)
    by_machine = _by_machine(queries, history)
    return {q.query_id: peer_share((q.episode_id, q.machine_id), q.cutoff, by_machine) for q in queries}


def apply(rows, split_dir, coef):
    """rows: {query_id: [p_0, p_1, p_2, p_5, p_20]} -> same with the peer term on the win log-odds."""
    if not coef:
        return rows
    shares = peer_shares(split_dir)
    known = [s for s in shares.values() if s is not None]
    center = math.fsum(known) / len(known) if known else 0.5
    out = {}
    for qid, row in rows.items():
        share = shares.get(qid)
        win = 1.0 - row[0]
        if share is None or win <= 0.0:
            out[qid] = row
            continue
        win = min(max(win, 1e-5), 1.0 - 1e-5)
        logit = math.log(win / (1.0 - win)) + coef * (share - center)
        new_win = 1.0 / (1.0 + math.exp(-logit))
        size_total = math.fsum(row[1:])
        raw = [1.0 - new_win] + [new_win * v / size_total for v in row[1:]]
        raw = [max(v, MIN_PROBABILITY) for v in raw]
        scale = math.fsum(raw)
        out[qid] = [v / scale for v in raw]
    return out
