"""Aggregate AI-usage stats for the TamuDatathon26-PSAZZ hackathon (Claude Code, Codex, Pane, git). Prints JSON."""
import collections
import datetime as dt
import glob
import json
import os
import re
import sqlite3
import subprocess

HOME = os.path.expanduser("~")
REPO = f"{HOME}/Desktop/Coding/TamuDatathon26-PSAZZ"
START = dt.datetime(2026, 11, 7, 6, 0, tzinfo=dt.timezone.utc)  # Sat 00:00 CST
CST = dt.timezone(dt.timedelta(hours=-6))


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def active_hours(times, gap_min=15):
    times = sorted(times)
    tot = 0.0
    for a, b in zip(times, times[1:]):
        d = (b - a).total_seconds()
        if d <= gap_min * 60:
            tot += d
    return tot / 3600


# ---------------- Claude Code ----------------
claude_files = []
for d in glob.glob(f"{HOME}/.claude/projects/*"):
    if "TamuDatathon26-PSAZZ" in d:
        claude_files += glob.glob(f"{d}/**/*.jsonl", recursive=True)

C = collections.defaultdict(collections.Counter)
tools, bash_heads, models, skills, mcp = (collections.Counter() for _ in range(5))
hours = collections.Counter()
edited_files = set()
seen_msg = set()
session_info = {}
user_prompts, prompt_chars = 0, 0
all_times, max_ctx = [], 0
interrupts = 0
subagent_files = 0
agent_types = collections.Counter()
for f in claude_files:
    is_sub = "/subagents/" in f
    subagent_files += is_sub
    sid = os.path.basename(f).split(".")[0]
    times = []
    for line in open(f, errors="ignore"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        t = d.get("timestamp")
        if not t:
            continue
        tt = ts(t)
        if tt < START:
            continue
        times.append(tt)
        typ = d.get("type")
        if typ == "user" and not is_sub and not d.get("isMeta"):
            m = d.get("message", {})
            c = m.get("content")
            if isinstance(c, str) and not c.startswith("<"):
                user_prompts += 1
                prompt_chars += len(c)
                if "[Request interrupted" in c:
                    interrupts += 1
            elif isinstance(c, list):
                for b in c:
                    if b.get("type") == "text" and not b.get("text", "").startswith("<"):
                        user_prompts += 1
                        prompt_chars += len(b["text"])
        if typ != "assistant":
            continue
        m = d.get("message", {})
        mid = m.get("id")
        for b in m.get("content", []) or []:
            if b.get("type") == "tool_use":
                n = b.get("name", "?")
                tools[n] += 1
                inp = b.get("input", {}) or {}
                if n == "Bash":
                    cmd = inp.get("command", "").strip()
                    head = re.split(r"[\s;|&]+", cmd)[0] if cmd else ""
                    if head in ("cd",):
                        parts = re.split(r"&&|;", cmd)
                        head = re.split(r"\s+", parts[1].strip())[0] if len(parts) > 1 else "cd"
                    bash_heads[os.path.basename(head)] += 1
                if n in ("Edit", "Write", "NotebookEdit"):
                    edited_files.add(inp.get("file_path", ""))
                if n == "Skill":
                    skills[inp.get("skill", "?")] += 1
                if n == "Agent":
                    agent_types[inp.get("subagent_type") or "general"] += 1
                if n.startswith("mcp__"):
                    mcp[n.split("__")[1]] += 1
        u = m.get("usage")
        if not u or mid in seen_msg:
            continue
        seen_msg.add(mid)
        mod = m.get("model", "?")
        models[mod] += 1
        k = "sub" if is_sub else "main"
        for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            C[k][key] += u.get(key, 0) or 0
            C["model:" + mod][key] += u.get(key, 0) or 0
        C[k]["messages"] += 1
        ctx = (u.get("input_tokens", 0) or 0) + (u.get("cache_read_input_tokens", 0) or 0) + (u.get("cache_creation_input_tokens", 0) or 0)
        max_ctx = max(max_ctx, ctx)
        hours[tt.astimezone(CST).strftime("%a %H")] += 1
    if times and not is_sub:
        session_info[sid] = (min(times), max(times), len(times), active_hours(times))
    all_times += times

claude = {
    "transcript_files": len(claude_files), "main_sessions_in_window": len(session_info), "subagent_transcripts": subagent_files,
    "tokens": {k: dict(v) for k, v in C.items()}, "models_by_message": dict(models), "tool_calls": dict(tools.most_common()),
    "bash_top": dict(bash_heads.most_common(15)), "skills": dict(skills), "mcp_servers": dict(mcp), "agent_spawns": dict(agent_types),
    "files_edited_or_written": len(edited_files), "user_prompts": user_prompts,
    "avg_prompt_chars": round(prompt_chars / max(user_prompts, 1)), "interrupts": interrupts, "max_context_tokens": max_ctx,
    "active_hours_union": round(active_hours(all_times), 2),
    "session_active_hours_sum": round(sum(v[3] for v in session_info.values()), 2),
    "longest_sessions_h": sorted([round(v[3], 2) for v in session_info.values()], reverse=True)[:5],
    "busiest_hours_cst": dict(hours.most_common(6)),
    "msgs_by_hour_cst": dict(sorted(hours.items())),
}

# ---------------- Codex ----------------
codex = collections.Counter()
ctools = collections.Counter()
cmodels = collections.Counter()
ctimes, cdirs, max_rl = [], collections.Counter(), 0
for f in glob.glob(f"{HOME}/.codex/sessions/**/*.jsonl", recursive=True):
    meta_cwd, last_total, times = None, None, []
    for line in open(f, errors="ignore"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        t, p = d.get("type"), d.get("payload", {}) or {}
        if t == "session_meta":
            meta_cwd = p.get("cwd", "")
        if d.get("timestamp"):
            tt = ts(d["timestamp"])
            if tt >= START:
                times.append(tt)
        if t == "event_msg" and p.get("type") == "token_count" and p.get("info"):
            last_total = p["info"].get("total_token_usage")
            rl = (d.get("payload", {}).get("rate_limits") or {}).get("primary") or {}
            max_rl = max(max_rl, rl.get("used_percent", 0) or 0)
        if t == "response_item" and p.get("type") in ("custom_tool_call", "function_call"):
            ctools[p.get("name", "?")] += 1
        if t == "turn_context":
            cmodels[p.get("model", "?")] += 1
        if t == "event_msg" and p.get("type") == "task_started":
            codex["turns"] += 1
        if t == "event_msg" and p.get("type") == "turn_aborted":
            codex["aborted_turns"] += 1
        if t == "compacted":
            codex["compactions"] += 1
    if not meta_cwd or "TamuDatathon26-PSAZZ" not in meta_cwd or not times:
        continue
    codex["sessions"] += 1
    cdirs[meta_cwd.replace(REPO, "<repo>")] += 1
    ctimes += times
    if last_total:
        for k, v in last_total.items():
            codex[k] += v or 0
codex_out = {"stats": dict(codex), "tool_calls": dict(ctools.most_common(10)), "models_by_turn": dict(cmodels),
             "dirs": dict(cdirs), "active_hours_union": round(active_hours(ctimes), 2), "max_5h_rate_limit_used_pct": max_rl}

# ---------------- overlap: hours where Claude and Codex were both working ----------------
def bins(times):
    return {t.replace(minute=(t.minute // 10) * 10, second=0, microsecond=0) for t in times}
cb, xb = bins(all_times), bins(ctimes)
overlap = {"claude_10min_bins": len(cb), "codex_10min_bins": len(xb), "both_10min_bins": len(cb & xb),
           "any_ai_hours": round(len(cb | xb) / 6, 1)}

# ---------------- git ----------------
def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout

log = git("log", "--remotes=origin", "--no-merges", "--format=%H%x1f%an%x1f%aI%x1f%B%x1e")
names = {"Saibot913": "Sai", "pkhanna29": "Pranav", "zmaknojia20": "Zhill", "ZhillMaknojia": "Zhill",
         "Avyukth Narra": "Avyukth", "avyukthNarra": "Avyukth", "avyTamuGit": "Avyukth"}
G = collections.defaultdict(collections.Counter)
seen = set()
sai_hours = collections.Counter()
body_lens = collections.defaultdict(list)
types = collections.defaultdict(collections.Counter)
for rec in log.split("\x1e"):
    rec = rec.strip()
    if not rec:
        continue
    h, a, when, body = rec.split("\x1f", 3)
    who = names.get(a, a)
    subj = body.splitlines()[0] if body else ""
    key = (who, subj)
    if key in seen:
        continue
    seen.add(key)
    G[who]["commits"] += 1
    if re.search(r"Co-Authored-By: Claude", body, re.I):
        G[who]["claude_coauthored"] += 1
    if re.search(r"codex", body, re.I) and re.search(r"co-authored|generated", body, re.I):
        G[who]["codex_marked"] += 1
    if "Claude-Session:" in body:
        G[who]["claude_session_link"] += 1
    m = re.match(r"(\w+)(\(|!|:)", subj)
    types[who][m.group(1) if m else "other"] += 1
    body_lens[who].append(len(body) - len(subj))
    if who == "Sai":
        sai_hours[dt.datetime.fromisoformat(when).astimezone(CST).strftime("%a %H")] += 1
for who, L in body_lens.items():
    L.sort()
    G[who]["median_body_chars"] = L[len(L) // 2]
num = git("log", "--remotes=origin", "--no-merges", "--author=Saibot913", "--numstat", "--format=")
add = dele = 0
for line in num.splitlines():
    p = line.split("\t")
    if len(p) == 3 and p[0].isdigit():
        add += int(p[0]); dele += int(p[1])
gh = lambda *a: subprocess.run(["gh", *a], capture_output=True, text=True, cwd=REPO).stdout
prs = json.loads(gh("pr", "list", "--state", "all", "--limit", "200", "--json", "number,author,state,createdAt") or "[]")
pr_by = collections.Counter((p["author"]["login"], p["state"]) for p in prs)
git_out = {"per_person": {k: dict(v) for k, v in G.items()}, "commit_types": {k: dict(v) for k, v in types.items()},
           "sai_lines_added": add, "sai_lines_deleted": dele, "sai_commits_by_hour_cst": dict(sorted(sai_hours.items())),
           "prs_total": len(prs), "prs_by_author_state": {f"{a}|{s}": n for (a, s), n in pr_by.items()},
           "remote_branches_now": len(git("ls-remote", "--heads", "origin").splitlines())}

# ---------------- Pane ----------------
pane = {}
try:
    db = sqlite3.connect(f"{HOME}/.pane/sessions.db")
    pane["panes_for_repo"] = db.execute("select count(*) from sessions s join projects p on s.project_id=p.id where p.path like '%TamuDatathon26-PSAZZ%'").fetchone()[0]
    pane["tool_panels_by_type"] = dict(db.execute("select type,count(*) from tool_panels group by 1").fetchall())
except Exception as e:
    pane["error"] = str(e)

print(json.dumps({"claude": claude, "codex": codex_out, "overlap": overlap, "git": git_out, "pane": pane}, indent=1, default=str))
