#!/usr/bin/env python3
"""Summarize observed paired runs without claiming general quality or token savings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def tokens(record):
    usage = record.get("usage", {})
    return {
        "input": usage.get("input_tokens", 0),
        "cached": usage.get("cached_input_tokens", 0),
        "uncached": max(
            0, usage.get("input_tokens", 0) - usage.get("cached_input_tokens", 0)
        ),
        "output": usage.get("output_tokens", 0),
    }


def summarize(root, *, recheck=False):
    planned = (
        json.loads((root / "planned.json").read_text())
        if (root / "planned.json").exists()
        else {}
    )
    cases = {case["skill"]: case for case in planned.get("cases", [])}
    records = []
    for path in sorted(root.glob("*/*/result.json")):
        record = json.loads(path.read_text())
        if recheck:
            from codex_fixtures import check_artifacts

            case = cases[record["skill"]]
            protected = dict(record["source_hashes"])
            # Tests may be extended during a coding task. Scientific inputs stay
            # immutable and the independent numerical contract still runs.
            if case["kind"] == "weighted":
                protected.pop("test_weighted.py", None)
            answer_path = path.parent / "answer.md"
            checks = check_artifacts(
                path.parent / "project",
                case,
                answer_path.read_text() if answer_path.exists() else "",
                protected,
            )
            record["recorded_checks"] = record["checks"]
            checks.update(
                {
                    key: value
                    for key, value in record["checks"].items()
                    if key in {"model_completed", "skill_read", "actual_mcp_call"}
                }
            )
            record["checks"] = checks
            record["checks_passed"] = all(checks.values())
        record["outcome_screen_passed"] = all(
            value
            for key, value in record.get("checks", {}).items()
            if key not in {"skill_read", "actual_mcp_call"}
        ) and bool(record.get("completed"))
        records.append(record)
    by_skill = {}
    for record in records:
        by_skill.setdefault(record["skill"], {})[record["mode"]] = record
    pairs = []
    for skill, modes in by_skill.items():
        if not {"baseline", "skill"} <= modes.keys():
            continue
        a, b = modes["baseline"], modes["skill"]
        comparable = (
            a.get("prompt_sha256") == b.get("prompt_sha256")
            and a.get("source_hashes") == b.get("source_hashes")
            and a.get("model") == b.get("model")
            and a.get("servers") == b.get("servers")
            and a.get("server_config_sha256") == b.get("server_config_sha256")
            and a.get("invocation", "automatic")
            == b.get("invocation", "automatic")
            == "automatic"
            and a.get("completed")
            and b.get("completed")
            and not a.get("runtime_blocked")
            and not b.get("runtime_blocked")
        )
        ta, tb = tokens(a), tokens(b)
        pairs.append(
            {
                "skill": skill,
                "package": b.get("package") or b.get("installation", {}).get("package"),
                "not_installed": b.get("installation", {}).get("not_installed", []),
                "baseline_outcome_screen": a["outcome_screen_passed"],
                "skill_outcome_screen": b["outcome_screen_passed"],
                "comparable": bool(comparable),
                "skill_read": b.get("checks", {}).get("skill_read", False),
                "baseline_uncached_input_output": ta["uncached"] + ta["output"],
                "skill_uncached_input_output": tb["uncached"] + tb["output"],
                "delta_uncached_input_output": tb["uncached"]
                + tb["output"]
                - ta["uncached"]
                - ta["output"],
                "baseline_checks": a.get("checks", {}),
                "skill_checks": b.get("checks", {}),
                "quality": "needs evidence review; no model self-score used",
            }
        )
    missing = [
        f"{c['skill']}/{mode}"
        for c in planned.get("cases", [])
        for mode in planned.get("modes", "").split(",")
        if mode not in by_skill.get(c["skill"], {})
    ]
    totals = {
        key: sum(tokens(r)[key] for r in records)
        for key in ("input", "cached", "uncached", "output")
    }
    deltas = [
        p["delta_uncached_input_output"]
        for p in pairs
        if p["comparable"] and p["skill_read"]
    ]
    return {
        "model": planned.get("model"),
        "runs": len(records),
        "artifact_recheck": recheck,
        "outcome_screens_passed": sum(r["outcome_screen_passed"] for r in records),
        "skills_read": sum(
            bool(r.get("checks", {}).get("skill_read"))
            for r in records
            if r["mode"] == "skill"
        ),
        "plugins": {
            r["installation"]["package"]: {
                "route": "portable MCPs and skills",
                "not_installed": r["installation"].get("not_installed", []),
            }
            for r in records
            if r.get("installation")
        },
        "harness_errors": [
            r
            for r in json.loads((root / "results.json").read_text())
            if r.get("harness_error")
        ]
        if (root / "results.json").exists()
        else [],
        "completed": sum(
            bool(r.get("checks", {}).get("model_completed", r.get("completed")))
            for r in records
        ),
        "checks_passed": sum(bool(r.get("checks_passed")) for r in records),
        "runtime_blocked": sum(bool(r.get("runtime_blocked")) for r in records),
        "tokens": totals,
        "missing": missing,
        "pairs": pairs,
        "median_observed_delta": statistics.median(deltas) if deltas else None,
        "interpretation": "One paired case per skill is exploratory. Cached input is reported separately; token counts are not dollar costs or subscription quota units. Boundary/draft cases do not establish live delivery. Tool/skill activation is separate from task correctness.",
    }


def render(summary):
    lines = [
        "# Live Codex evaluation",
        "",
        summary["interpretation"],
        "",
        f"Model: {summary['model']}. Recorded runs: {summary['runs']}; completed: {summary['completed']}; automated checks passed: {summary['checks_passed']}; runtime-blocked: {summary['runtime_blocked']}. Scientific quality still requires review.",
        "",
        f"Task-outcome screens passed: {summary['outcome_screens_passed']}. Expected skill reads: {summary['skills_read']}. Artifact recheck: {summary['artifact_recheck']}. Outcome screens do not require a baseline to use MCP instead of another correct method.",
        "",
        f"Observed tokens: {summary['tokens']}",
        "",
        "| Skill | Comparable pair | Skill read | Baseline uncached input + output | With skill | Difference |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for p in summary["pairs"]:
        lines.append(
            f"| {p['skill']} | {p['comparable']} | {p['skill_read']} | {p['baseline_uncached_input_output']} | {p['skill_uncached_input_output']} | {p['delta_uncached_input_output']:+} |"
        )
    if summary["plugins"]:
        lines += ["", "Plugin coverage (portable component installation):"]
        for name, item in sorted(summary["plugins"].items()):
            lines.append(
                f"- {name}: {item['route']}; not installed: {', '.join(item['not_installed']) or 'none'}"
            )
    if summary["harness_errors"]:
        lines += ["", "Harness errors: " + json.dumps(summary["harness_errors"])]
    if summary["missing"]:
        lines += ["", "Not completed: " + ", ".join(summary["missing"])]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--recheck-artifacts",
        action="store_true",
        help="Recheck trusted fixture outputs with current rules (may execute generated fixture code); preserve raw records",
    )
    args = parser.parse_args()
    report = summarize(args.output, recheck=args.recheck_artifacts)
    stem = "rechecked-summary" if args.recheck_artifacts else "summary"
    (args.output / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / f"{stem}.md").write_text(render(report))
    print(
        f"Recorded {report['runs']} runs, {len(report['pairs'])} pairs; {len(report['missing'])} missing. See {args.output / f'{stem}.md'}"
    )
