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


def summarize(root):
    records = [json.loads(p.read_text()) for p in sorted(root.glob("*/*/result.json"))]
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
            and a.get("completed")
            and b.get("completed")
            and not a.get("runtime_blocked")
            and not b.get("runtime_blocked")
        )
        ta, tb = tokens(a), tokens(b)
        pairs.append(
            {
                "skill": skill,
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
    planned = (
        json.loads((root / "planned.json").read_text())
        if (root / "planned.json").exists()
        else {}
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
        "completed": sum(bool(r.get("completed")) for r in records),
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
        f"Observed tokens: {summary['tokens']}",
        "",
        "| Skill | Comparable pair | Skill read | Baseline uncached input + output | With skill | Difference |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for p in summary["pairs"]:
        lines.append(
            f"| {p['skill']} | {p['comparable']} | {p['skill_read']} | {p['baseline_uncached_input_output']} | {p['skill_uncached_input_output']} | {p['delta_uncached_input_output']:+} |"
        )
    if summary["missing"]:
        lines += ["", "Not completed: " + ", ".join(summary["missing"])]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = summarize(args.output)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "summary.md").write_text(render(report))
    print(
        f"Recorded {report['runs']} runs, {len(report['pairs'])} pairs; {len(report['missing'])} missing. See {args.output / 'summary.md'}"
    )
