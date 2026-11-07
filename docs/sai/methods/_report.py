"""Write reports/<ID>.md (<=15 lines) from results/<ID>.json + grids/<ID>.json. Usage: python3 _report.py ID 'what was run' 'surprises/followups'"""
import json, sys
from pathlib import Path
S = Path(__file__).resolve().parent.parent
mid, ran, notes = sys.argv[1:4]
d = json.load(open(S / f"results/{mid}.json")); g = json.load(open(S / f"grids/{mid}.json"))
x = d["extra"]; cr = d["condition_ratios"]; b3 = json.load(open(S / "results/B3.json"))["condition_ratios"]
f = lambda c: f"[{c[0]:.2f},{c[1]:.2f}]"
pts = sorted(g, key=lambda r: -r["points"])
L = [f"# {mid}",
 f"- Ran: {ran} ({len(g)} cells completed, scope dev, 72 episodes; B3 = 45.82).",
 f"- Best cell: {json.dumps(d['params'])}",
 f"- Points {d['points']:.2f} CI {f(d['ci95'])}; worst cell {pts[-1]['points']:.2f}; median cell {pts[len(pts)//2]['points']:.2f}.",
 "- Condition ratios (method vs B3): " + ", ".join(f"{k} {cr[k]:.3f} vs {b3[k]:.3f}" for k in cr),
 "- Class error shares: " + ", ".join(f"{k} {v:.3f}" for k, v in d["class_error_shares"].items()),
 f"- vs B3 standalone: {x['vs_B3']['diff']:+.2f} CI {f(x['vs_B3']['ci95'])}",
 f"- blend_gain with B3: {x['blend_gain']['points']:.2f} CI {f(x['blend_gain']['ci95'])}; vs B3 {x['blend_gain']['vs_base']['diff']:+.2f} CI {f(x['blend_gain']['vs_base']['ci95'])}; mean weight on method {x['blend_gain']['w_mean']:.2f}",
 f"- Runtime {d['runtime_seconds']:.1f}s (best cell, dev+val 768 queries); determinism diff {d['determinism_max_diff']}; stdlib-only: no (numpy)",
 f"- Notes: {notes}"]
(S / f"reports/{mid}.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
