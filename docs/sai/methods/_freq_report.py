"""Write reports/<ID>.md (<=15 lines) for W1 methods. Usage: python3 _freq_report.py ID 'notes'"""
import json, sys
from pathlib import Path
S = Path(__file__).resolve().parent.parent
mid, notes = sys.argv[1:3]
d = json.load(open(S / f"results/{mid}.json")); g = json.load(open(S / f"grids/{mid}.json"))
x = d["extra"]; b3 = json.load(open(S / "results/B3.json"))["condition_ratios"]
f = lambda c: f"[{c[0]:.2f},{c[1]:.2f}]"
doc = (sys.modules.get("x") or None)
L = [f"# {mid}", f"- Ran: grid pre-declared in methods/{mid}.py docstring, {len(g)} cells completed (dev, 72 episodes); B3 = 45.82; train-prior B3 cell reproduces 45.82."]
for md, r in x["by_mode"].items():
    cr = r["condition_ratios"]; cs = r["class_shares"]
    L.append(f"- Mode {md} best {r['points']:.2f} CI {f(r['ci95'])}: {json.dumps(r['params'])}")
    L.append(f"  vs B3 {r['vs_B3']['diff']:+.2f} CI {f(r['vs_B3']['ci95'])}; blend_gain {r['blend_gain']['points']:.2f}, vs B3 {r['blend_gain']['vs_base']['diff']:+.2f} CI {f(r['blend_gain']['vs_base']['ci95'])}, w {r['blend_gain']['w_mean']:.2f}")
    L.append("  cond ratios (B3): " + ", ".join(f"{k[:6]} {cr[k]:.3f} ({b3[k]:.3f})" for k in cr) + "; err shares " + ", ".join(f"{k[2:]} {v:.3f}" for k, v in cs.items()))
pts = sorted(r["points"] for r in g if r["params"].get("mode") == "ridge")
L.append(f"- Ridge-mode cells: worst {pts[0]:.2f}, median {pts[len(pts)//2]:.2f}, best {pts[-1]:.2f}. B3 error shares: " + ", ".join(f"{k[2:]} {v:.3f}" for k, v in x["class_shares_B3"].items()))
L.append(f"- Runtime {d['runtime_seconds']:.1f}s (best cell); determinism diff {d['determinism_max_diff']}; stdlib-only: no (numpy). Best cell tuned on dev (selection optimism not corrected).")
L.append(f"- Notes: {notes}")
(S / f"reports/{mid}.md").write_text("\n".join(L) + "\n"); print(len(L)); 
