"""Read-only, bounded verification for an explicitly prepared dataset report."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

MAX_BYTES = 64 * 1024 * 1024
MAX_ROWS = 100_000


def digest(path):
    if path.stat().st_size > MAX_BYTES:
        raise ValueError(f"Verification limit: {path} exceeds 64 MiB")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(path):
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Evidence manifest exceeds 1 MiB")
    evidence = json.loads(path.read_text())
    if not isinstance(evidence, dict) or evidence.get("schema") != 1:
        raise ValueError("Expected dataset-report schema 1")
    files = {key: Path(evidence[key]) for key in ("source", "csv", "figure", "report")}
    if not all(file.is_absolute() for file in files.values()):
        raise ValueError("Evidence paths must be absolute")
    if len({file.resolve() for file in files.values()}) != 4:
        raise ValueError("Source and output paths must be distinct")
    if digest(files["source"]) != evidence["source_sha256"]:
        raise ValueError("Source changed since preparation")
    csv_hash = digest(files["csv"])
    with files["csv"].open(newline="") as stream:
        reader = csv.DictReader(stream)
        column = evidence["column"]
        if not reader.fieldnames or reader.fieldnames.count(column) != 1:
            raise ValueError("Expected exactly one selected numeric column")
        values = []
        for row in reader:
            if len(values) >= MAX_ROWS:
                raise ValueError("Verification limit: CSV exceeds 100000 rows")
            value = float(row[column])
            if not math.isfinite(value):
                raise ValueError("Selected column contains non-finite values")
            values.append(value)
    if not values:
        raise ValueError("Selected column is empty")
    actual = {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }
    claimed = evidence["statistics"]
    if not isinstance(claimed, dict):
        raise ValueError("statistics must be an object")
    for key, value in actual.items():
        claim = claimed.get(key)
        if type(claim) not in (int, float) or not math.isfinite(claim):
            raise ValueError(f"Missing or invalid statistic: {key}")
        if not math.isclose(claim, value, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"Statistic mismatch: {key}; calculated {value}, claimed {claim}"
            )
    if files["figure"].stat().st_size > MAX_BYTES:
        raise ValueError("Figure exceeds verification limit")
    with files["figure"].open("rb") as stream:
        if stream.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError("Figure does not have a PNG signature")
    if not files["report"].stat().st_size:
        raise ValueError("Report is empty")
    return {
        "status": "PASS",
        "statistics": actual,
        "csv_sha256": csv_hash,
        "source_sha256": evidence["source_sha256"],
        "scope": "CSV statistics, unchanged source and output presence only; not plot semantics, CSV provenance or report interpretation.",
    }


def result(path):
    try:
        return verify(path)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        return {"status": "FAIL", "reason": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--column", required=True)
    check = sub.add_parser("check")
    check.add_argument("manifest", type=Path)
    sub.add_parser("hook")
    args = parser.parse_args()
    if args.action == "prepare":
        source = args.source.resolve()
        folder = args.output.resolve()
        folder.mkdir(parents=True, exist_ok=True)
        evidence = {
            "schema": 1,
            "source": str(source),
            "source_sha256": digest(source),
            "csv": str(folder / "data.csv"),
            "figure": str(folder / "plot.png"),
            "report": str(folder / "dataset-report.md"),
            "column": args.column,
            "statistics": {},
        }
        path = folder / "clio-dataset-report.json"
        # Do not silently replace the original source baseline on retries.
        with path.open("x") as stream:
            json.dump(evidence, stream, indent=2)
        print(path)
    elif args.action == "check":
        output = result(args.manifest)
        print(json.dumps(output))
        sys.exit(0 if output["status"] == "PASS" else 1)
    else:
        event = json.load(sys.stdin)
        changed = Path(event.get("tool_input", {}).get("file_path", ""))
        if not changed.is_absolute():
            changed = Path(event.get("cwd", ".")) / changed
        if changed.name not in {"clio-dataset-report.json", "dataset-report.md"}:
            return
        manifest = changed.parent / "clio-dataset-report.json"
        if not manifest.is_file():
            return
        output = result(manifest)
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PostToolUse",
                        "additionalContext": "CLIO_DATASET_REPORT_CHECK "
                        + json.dumps(output),
                    }
                }
            )
        )


if __name__ == "__main__":
    main()
