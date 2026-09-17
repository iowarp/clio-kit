"""Exercise report verification against real files and corrupt evidence."""

import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "plugins/clio-dataset-report/skills/creating-dataset-report/scripts/verify_report.py"
)
checker = ModuleType("report_verification")
exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), checker.__dict__)


@pytest.fixture
def report(tmp_path):
    source = tmp_path / "source.h5"
    source.write_bytes(b"fixture source, never interpreted by verifier")
    output = tmp_path / "output"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "prepare",
            "--source",
            str(source),
            "--output",
            str(output),
            "--column",
            "signal",
        ],
        check=True,
        capture_output=True,
    )
    path = output / "clio-dataset-report.json"
    evidence = json.loads(path.read_text())
    evidence["statistics"] = {"count": 3, "mean": 4, "median": 4, "min": 2, "max": 6}
    path.write_text(json.dumps(evidence))
    (output / "data.csv").write_text("time,signal\n0,2\n1,4\n2,6\n")
    # The checker explicitly checks the signature, not image semantics.
    (output / "plot.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (output / "dataset-report.md").write_text("Report awaiting interpretation review")
    return path, source


def test_valid_evidence_and_no_baseline_overwrite(report):
    path, source = report
    before = path.read_bytes()
    assert checker.result(path)["status"] == "PASS"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "prepare",
            "--source",
            str(source),
            "--output",
            str(path.parent),
            "--column",
            "signal",
        ],
        capture_output=True,
    )
    assert result.returncode != 0
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "corruption",
    ["mean", "missing-statistic", "source", "png", "nan", "empty", "row-limit"],
)
def test_incorrect_or_incomplete_evidence_fails(report, corruption, monkeypatch):
    path, source = report
    evidence = json.loads(path.read_text())
    if corruption == "mean":
        evidence["statistics"]["mean"] = 22
    elif corruption == "missing-statistic":
        evidence["statistics"].pop("median")
    elif corruption == "source":
        source.write_bytes(b"changed")
    elif corruption == "png":
        Path(evidence["figure"]).unlink()
    elif corruption == "nan":
        Path(evidence["csv"]).write_text("signal\nnan\n")
    elif corruption == "empty":
        Path(evidence["report"]).write_text("")
    else:
        monkeypatch.setattr(checker, "MAX_ROWS", 2)
    path.write_text(json.dumps(evidence))
    assert checker.result(path)["status"] == "FAIL"


def test_hook_reports_actual_result_and_ignores_unrelated_writes(report):
    path, source = report
    original = source.read_bytes()

    def hook(changed):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "hook"],
            text=True,
            input=json.dumps(
                {"cwd": str(path.parent), "tool_input": {"file_path": changed}}
            ),
            capture_output=True,
            check=True,
        )
        return result.stdout

    assert hook("unrelated.md") == ""
    assert "CLIO_DATASET_REPORT_CHECK" in hook("dataset-report.md")
    assert '\\"status\\": \\"PASS\\"' in hook("dataset-report.md")
    evidence = json.loads(path.read_text())
    evidence["statistics"]["mean"] = 22
    path.write_text(json.dumps(evidence))
    assert "Statistic mismatch" in hook(str(path))
    assert source.read_bytes() == original


@pytest.mark.parametrize("content", ["[]", "null", "{", '{"schema": 1}'])
def test_malformed_manifest_returns_failure(report, content):
    path, _ = report
    path.write_text(content)
    assert checker.result(path)["status"] == "FAIL"
