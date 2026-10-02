"""How a recorded tool result is judged to have failed.

This number is what the eval report presents as evidence about the servers, so
a scoring rule that misreads a result publishes a false claim about working
code. The rule it replaced scanned the whole payload for words like "error",
"not found" and "could not", which cannot distinguish a server reporting an
error from a server reporting *on* errors.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

from scoring import REPO_ROOT, anonymise, call_failed  # noqa: E402


def test_a_log_analyser_reporting_errors_is_not_itself_failing() -> None:
    """The regression: a log full of ERROR lines is a successful analysis."""
    result = json.dumps(
        {
            "total_lines": 200,
            "valid_entries": 200,
            "statistics": {"levels": {"ERROR": 12}, "sample": "file not found"},
        }
    )
    assert not call_failed(False, result)


def test_a_server_reporting_an_error_key_failed() -> None:
    assert call_failed(False, json.dumps({"error": "File not found: x"}))
    assert call_failed(False, json.dumps({"success": False, "error": "Unknown"}))


def test_a_failure_reported_inside_an_ordinary_field_is_still_a_failure() -> None:
    """Not every server has an error key; hdf5 answers in `result`."""
    assert call_failed(False, json.dumps({"result": "Error: Path not found"}))


def test_a_plain_success_is_not_a_failure() -> None:
    assert not call_failed(False, json.dumps({"success": True, "rows": 3}))
    assert not call_failed(False, "Wrote 3 rows to runs_cleaned.csv")


def test_the_protocols_own_error_flag_is_authoritative() -> None:
    """A message with no error vocabulary at all can still be an error."""
    assert call_failed(True, "scientific catalog is not configured")


def test_unparseable_output_falls_back_to_how_it_opens() -> None:
    assert call_failed(False, "Error: something broke")
    assert not call_failed(False, "{not json at all")


def test_redaction_preserves_nested_inputs_and_json_escaping() -> None:
    calls = [
        {
            "input": {
                "query": 'Read "/home/alice/data.csv" and summarize it.',
                "items": ["/Users/alice/data.csv\nNext line", 3, True, None],
                "nested": {"path": f"{REPO_ROOT}/evals/fixtures/runs.csv"},
            },
            "output": 'Read "/home/alice/data.csv" successfully.',
        }
    ]

    result = anonymise(calls)

    assert result[0]["input"] == {
        "query": 'Read "<home>" and summarize it.',
        "items": ["<home>\nNext line", 3, True, None],
        "nested": {"path": "<repo>/evals/fixtures/runs.csv"},
    }
    assert result[0]["output"] == 'Read "<home>" successfully.'
    assert json.loads(json.dumps(result)) == result
    assert anonymise(result) == result
