"""Write report.json and report.md from the records collected by cli.py.

A record is one check outcome: {"id", "covers", "status", "message", "source", ...}.
`covers` lists the requirement check IDs (requirements.py) the record evidences, so
the coverage table is derived from what ran, never typed by hand. Every requirement
in requirements.REQUIREMENTS gets a row, including the ones that did not run.
"""

from __future__ import annotations

import json
from pathlib import Path

from .requirements import REQUIREMENTS

STATUS_ORDER = ("FAIL", "WARN", "NOT RUN", "PASS", "INFO")
_PRIORITY = {status: index for index, status in enumerate(STATUS_ORDER)}
CONDITIONS = ("stable", "short_history", "drift", "shared_change")
SLICE_DIMENSIONS = ("condition", "machine_type", "signal", "history_kind", "cutoff_bin")


def _cell(text) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def _short(text, limit=160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def requirement_rows(records, not_run_reasons, level, mode):
    """One row per requirement: status is the worst status of its check IDs."""
    by_id = {}
    for record in records:
        for check_id in record["covers"]:
            by_id.setdefault(check_id, []).append(record)

    rows = []
    for requirement in REQUIREMENTS:
        check_ids = requirement["check_ids"]
        row = {
            "id": requirement["id"], "category": requirement["category"],
            "testable_by": requirement["testable_by"], "check_ids": check_ids,
            "source": requirement["source"], "quote": requirement["quote"],
        }
        if requirement["testable_by"] == "manual" or not check_ids:
            row.update(status="MANUAL", evidence=[], reason=requirement.get("note", "no automated check"))
            rows.append(row)
            continue

        evidence, statuses, reasons = [], [], []
        for check_id in check_ids:
            recs = by_id.get(check_id, [])
            if not recs:
                reason = not_run_reasons.get(check_id) or _default_reason(check_id, level, mode)
                statuses.append("NOT RUN")
                reasons.append(f"{check_id}: {reason}")
                evidence.append({"check_id": check_id, "status": "NOT RUN", "message": reason})
                continue
            worst = min((r["status"] for r in recs), key=_PRIORITY.__getitem__)
            statuses.append(worst)
            for record in recs:
                evidence.append({"check_id": check_id, "status": record["status"],
                                 "message": record["message"], "source": record["source"]})
                if record["status"] == "NOT RUN":
                    reasons.append(f"{check_id}: {record['message']}")
        status = min(statuses, key=_PRIORITY.__getitem__)
        if status == "NOT RUN" and any(s != "NOT RUN" for s in statuses):
            reasons.insert(0, "partly verified; the rest did not run")
        row.update(status=status, evidence=evidence, reason="; ".join(reasons))
        rows.append(row)
    return rows


def _default_reason(check_id, level, mode):
    if level == "quick":
        return "not part of the quick suite (run --suite full)"
    return f"not produced by this run (level {level}, mode {mode}); see caveats"


def counts(rows, records):
    tally = {status: 0 for status in (*STATUS_ORDER, "MANUAL")}
    for row in rows:
        tally[row["status"]] += 1
    record_fail = sum(1 for r in records if r["status"] == "FAIL")
    return tally, record_fail


def verdict_line(tally, record_fail, ctx) -> str:
    if tally["FAIL"] or record_fail:
        head = "FAIL"
    elif tally["WARN"]:
        head = "PASS WITH WARNINGS"
    else:
        head = "PASS"
    authority = ("NON-AUTHORITATIVE local run (entry runs on the host and could read answer files; "
                 "only docker mode isolates them)" if ctx["mode"] == "local"
                 else "docker run (VM memory and quotas as reported)")
    errors = f"; {len(ctx['errors'])} suite error(s)" if ctx["errors"] else ""
    return (f"VERDICT: {head} - {tally['FAIL']} requirement(s) FAIL, {tally['WARN']} WARN, "
            f"{tally['PASS']} PASS, {tally['INFO']} INFO, {tally['NOT RUN']} NOT RUN, "
            f"{tally['MANUAL']} MANUAL{errors}. Level {ctx['level']}, {authority}.")


def _fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _points_table(lines, scores):
    lines.append("| Dataset | Points | R | 95% bootstrap CI | Half-split A mean (sd, range) | "
                 "Half-split B mean (sd, range) | " + " | ".join(f"ratio {c}" for c in CONDITIONS) + " |")
    lines.append("|" + "---|" * (6 + len(CONDITIONS)))
    for name, bundle in scores.items():
        if bundle.get("status") == "NOT RUN":
            lines.append(f"| {_cell(name)} | NOT RUN: {_cell(bundle['reason'])} |" + " |" * (4 + len(CONDITIONS)))
            continue
        halves = bundle["half_splits"]
        half = lambda d: f"{d['mean']:.2f} ({d['sd']:.2f}, {d['min']:.2f}-{d['max']:.2f})"
        ratios = " | ".join(f"{bundle['ratio'][c]:.4f}" for c in CONDITIONS)
        lines.append(f"| {_cell(name)} | {bundle['points']:.4f} / {bundle['max_points']} | {bundle['R']:.4f} | "
                     f"[{bundle['ci_low']:.3f}, {bundle['ci_high']:.3f}] | {half(halves['half_a'])} | "
                     f"{half(halves['half_b'])} | {ratios} |")


def _evidence_text(evidence):
    """One phrase per check ID and status, with the first message for FAIL/WARN rows."""
    seen, parts = {}, []
    for item in evidence:
        key = (item["check_id"], item["status"])
        seen[key] = seen.get(key, 0) + 1
        if key in [k for k, _ in parts]:
            continue
        note = f" ({_short(item['message'], 110)})" if item["status"] in ("FAIL", "WARN") else ""
        parts.append((key, f"{item['check_id']}: {item['status']}{note}"))
    result = []
    for key, text in parts:
        count = seen[key]
        result.append(text + (f" x{count}" if count > 1 and key[1] == "PASS" else ""))
    return result


def _slices_section(lines, slices_data):
    for dimension in SLICE_DIMENSIONS:
        table = slices_data.get(dimension)
        if not table:
            continue
        lines.append(f"\n**By {dimension}**\n")
        lines.append("| Value | Queries | MSE | Reference MSE | Ratio | Win-prob error |")
        lines.append("|---|---|---|---|---|---|")
        for row in table:
            lines.append(f"| {_cell(row['value'])} | {row['n_queries']} | {row['mse']:.5f} | "
                         f"{row['reference_mse']:.5f} | {row['ratio']:.4f} | {row['win_prob_error']:.5f} |")


def render_markdown(ctx, rows, tally, record_fail) -> str:
    lines = [f"# eval_suite report: {ctx['entry_name']}", "", f"**{verdict_line(tally, record_fail, ctx)}**", ""]
    lines.append(f"Entry `{ctx['entry']}`"
                 + (f" compared with `{ctx['compare']}`" if ctx.get("compare") else "")
                 + f". Seed {ctx['seed']}, k={ctx['k']}, bootstrap repeats {ctx['repeats']}. "
                 f"Wall time {ctx['elapsed_s']:.1f}s. Generated {ctx['started']}.")
    lines.append("")

    lines.append("## 1. Requirement coverage")
    lines.append(f"{tally['PASS']} PASS, {tally['FAIL']} FAIL, {tally['WARN']} WARN, {tally['INFO']} INFO, "
                 f"{tally['NOT RUN']} NOT RUN, {tally['MANUAL']} MANUAL of {len(rows)} requirements.\n")
    lines.append("| ID | Category | Status | Check IDs | Source | Reason / evidence |")
    lines.append("|---|---|---|---|---|---|")
    for row in rows:
        if row["status"] == "MANUAL":
            detail = row["reason"]
        elif row["status"] == "NOT RUN":
            detail = row["reason"]
        else:
            detail = "; ".join(_evidence_text(row["evidence"]))
        ids = ", ".join(f"`{c}`" for c in row["check_ids"]) or "-"
        lines.append(f"| {row['id']} | {row['category']} | {row['status']} | {ids} | {_cell(row['source'])} | "
                     f"{_cell(_short(detail, 400))} |")
    lines.append("")

    lines.append("## 2. Scores")
    lines.append("Points = max_points / (1 + R), where R is the mean over conditions of E[c] / B[c]. "
                 "Lower ratios are better. The CI is a condition-stratified episode bootstrap. "
                 "Half-split rows show points on random halves of the episodes.\n")
    real = {k: v for k, v in ctx["scores"].items() if not k.startswith("preset:")}
    presets = {k[len("preset:"):]: v for k, v in ctx["scores"].items() if k.startswith("preset:")}
    lines.append("### Real validation (k-fold, each episode scored with answers hidden from its entry run)")
    _points_table(lines, real or {"validation k-fold": {"status": "NOT RUN", "reason": "not produced"}})
    lines.append("\n### Generator presets (spec-derived approximations, scored against _truth)")
    if presets:
        _points_table(lines, presets)
    else:
        lines.append("No presets were scored at this level.")
    lines.append("")

    lines.append("## 3. Slices (real validation, k-fold stitched)")
    validation = ctx["scores"].get("validation-kfold")
    if validation and "slices" in validation:
        _slices_section(lines, validation["slices"])
    else:
        lines.append("NOT RUN: no stitched validation predictions.")
    lines.append("")

    lines.append("## 4. Paired comparison")
    if ctx["paired"]:
        lines.append("| Dataset | Points A | Points B | A - B | 95% CI of difference | Share of resamples A better |")
        lines.append("|---|---|---|---|---|---|")
        for name, row in ctx["paired"].items():
            if row.get("status") == "NOT RUN":
                lines.append(f"| {_cell(name)} | NOT RUN: {_cell(row['reason'])} | | | | |")
                continue
            lines.append(f"| {_cell(name)} | {row['points_a']:.4f} | {row['points_b']:.4f} | {row['mean_diff']:+.4f} | "
                         f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] | {row['share_a_better']:.3f} |")
    else:
        lines.append("No --compare entry was given." if not ctx.get("compare") else "NOT RUN: no paired results.")
    lines.append("")

    lines.append("## 5. Runtime and memory")
    lines.append("| Run | Entry | Mode | Exit | Wall s | Limit s | Peak MB | Output |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for run in ctx["runs"]:
        lines.append(f"| {_cell(run['label'])} | {'B' if run['entry'] != ctx['entry'] else 'A'} | {run['mode']} | "
                     f"{run['exit_code']} | {run['wall_s']:.2f} | {run['limit_s']} | "
                     f"{_fmt(run['peak_mem_mb'], 1)} | {'csv' if run['csv'] else 'none'} |")
    slowest = max(ctx["runs"], key=lambda r: r["wall_s"], default=None)
    if slowest:
        lines.append(f"\nSlowest run: {slowest['label']} at {slowest['wall_s']:.2f}s "
                     f"({100 * slowest['wall_s'] / slowest['limit_s']:.2f}% of the {slowest['limit_s']}s limit).")
    lines.append("")

    lines.append("## 6. Caveats")
    for caveat in ctx["caveats"]:
        lines.append(f"- {caveat}")
    if ctx["errors"]:
        lines.append("")
        lines.append("Suite errors (checks that could not run because the suite itself failed):")
        for error in ctx["errors"]:
            lines.append(f"- {_cell(_short(error, 300))}")
    lines.append("")
    return "\n".join(lines)


def write_report(out_dir, ctx) -> dict:
    """Write report.json and report.md into out_dir; return the computed rows and tallies."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = requirement_rows(ctx["records"], ctx["not_run_reasons"], ctx["level"], ctx["mode"])
    tally, record_fail = counts(rows, ctx["records"])
    payload = {
        "verdict": verdict_line(tally, record_fail, ctx),
        "level": ctx["level"], "mode": ctx["mode"], "entry": ctx["entry"], "compare": ctx.get("compare"),
        "kit": ctx["kit"], "seed": ctx["seed"], "k": ctx["k"], "bootstrap_repeats": ctx["repeats"],
        "elapsed_s": ctx["elapsed_s"], "started": ctx["started"],
        "tally": tally, "requirements": rows, "checks": ctx["records"], "scores": ctx["scores"],
        "paired": ctx["paired"], "runs": ctx["runs"], "datasets_checked": ctx["datasets"],
        "notes": ctx["notes"], "errors": ctx["errors"], "caveats": ctx["caveats"],
    }
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                                         encoding="utf-8")
    (out_dir / "report.md").write_text(render_markdown(ctx, rows, tally, record_fail), encoding="utf-8")
    return {"rows": rows, "tally": tally, "record_fail": record_fail}
