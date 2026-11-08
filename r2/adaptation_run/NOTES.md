# Round-two (adaptation) run, 2026-11-08

Kit: organizer `adaptation-participants` (metadata `round: adaptation`, `max_points: 20`, old 80/20 scoring lib).
Our `mainFiles/slot_forecasting/rounds.py` (60/40) rejects 80/20 kits, so ranking tools ran on a copy whose
metadata says `max_points: 40` (rescales scores only). The entry itself runs fine on the real kit.

- Structure moved: win pairs (1,3),(2,5),(4,7) -> (3,8),(1,4),(2,6); original-pair share 0.00. Rule families: parity only.
- `gate_members.py`: keeps all three members (v13_short, exact, sysid), size member v13_short.
  Cross-fitted validation **38.59 / 40** (= ~19.29 / 20 on the real kit scale).
- `adapted_entry_v17g_rules/gate.json` is the adaptation gate. `bundled/validation/` = the kit's `validation/` (not committed).
- Ranking (RANKING.md) and final CSV/ZIP SHA-256 will be added when done.
