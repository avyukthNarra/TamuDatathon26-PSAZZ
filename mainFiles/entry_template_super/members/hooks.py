"""Swap hook for s1-adapt.

Called by entry/predict.py (REGISTRY, SIZE_MEMBER) and by
entry/members/v13_short.py and entry/members/exact_filter.py (PAIRS_OVERRIDE).
No existing entry file is a member registry. wt/orch/blend_final.py only
averages saved CSVs. This module does not read or write data files.

Drop in a replacement member by editing REGISTRY (one script per model).
Each script is run as its own process:

    python members/<name>.py --data DATA --split SPLIT --out CSV --protocol submission|crossfit

and writes query_id,p_0,p_1,p_2,p_5,p_20.

PAIRS_OVERRIDE: None means each member learns its lag pairs from train.
Set it to a list of (lag_a, lag_b) to force the exact filter, the short-history
patch, and the jackpot patch. Sysid ignores it: that member learns its rule
library from train.
"""

REGISTRY = (
    {"name": "v13_short", "script": "members/v13_short.py", "role": "win+size"},
    {"name": "exact", "script": "members/exact_filter.py", "role": "win"},
    {"name": "sysid", "script": "members/sysid_rule.py", "role": "win"},
    # pranav/super: Pranav's v19 as a 4th equal win member (cross-fit 57.973 -> 57.986 on saved
    # member forecasts; v19 carries the peer, signal, flip-transfer and gate terms the others lack).
    {"name": "pranav_v19", "script": "members/pranav_v19.py", "role": "win"},
)

# Conditional size given a win comes from this member (v13, after the short patch,
# which does not change the size split).
SIZE_MEMBER = "v13_short"

# None means "use each member's published pair source".
PAIRS_OVERRIDE = None
