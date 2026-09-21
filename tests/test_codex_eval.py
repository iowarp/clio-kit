"""Evaluation must not turn resource discovery or backend failures into success."""

import importlib.util
from pathlib import Path
import sys

EVALS = Path(__file__).resolve().parents[1] / "evals"
sys.path.insert(0, str(EVALS))
spec = importlib.util.spec_from_file_location("clio_live_eval", EVALS / "codex_eval.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def event(item):
    return {"type": "item.completed", "item": item}


def test_actual_server_calls_exclude_discovery_and_count_structured_errors():
    events = [
        event(
            {
                "type": "mcp_tool_call",
                "server": "codex",
                "tool": "list_mcp_resources",
                "result": {},
            }
        ),
        event(
            {
                "type": "mcp_tool_call",
                "server": "pandas",
                "tool": "load_data",
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": '{"success":false,"error":"bad input"}',
                        }
                    ]
                },
            }
        ),
    ]
    result = runner.summarize_events(events, ["pandas"])
    assert result["mcp_calls"] == 1
    assert result["mcp_failed"] == 1
    assert result["mcp_tools"] == ["pandas/load_data"]


def test_failed_skill_read_is_not_invocation_evidence():
    result = runner.summarize_events(
        [
            event(
                {
                    "type": "command_execution",
                    "command": "cat skills/example/SKILL.md",
                    "exit_code": 1,
                }
            )
        ]
    )
    assert result["skill_reads"] == []


def test_usage_is_observed_not_estimated_from_text():
    result = runner.summarize_events(
        [
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 80,
                    "output_tokens": 20,
                },
            }
        ]
    )
    assert result["usage"] == {
        "input_tokens": 100,
        "cached_input_tokens": 80,
        "output_tokens": 20,
    }
    assert result["completed"]


def test_mcp_error_envelopes():
    assert runner.tool_failed({"result": {"isError": True}})
    assert runner.tool_failed({"error": {"message": "approval denied"}})
    assert runner.tool_failed({"result": {"structured_content": {"success": False}}})
    assert not runner.tool_failed(
        {
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": '{"success":true,"message":"found ERROR logs"}',
                    }
                ]
            }
        }
    )


def test_cases_cover_current_inventory_without_duplicate_ids():
    cases = runner.CASES
    assert len(cases) == len({c["skill"] for c in cases})
    assert {c["skill"] for c in cases} == set(runner.skill_records())


def test_large_data_check_accepts_exact_or_qualified_sample_not_unqualified_mean(
    tmp_path,
):
    from codex_fixtures import check_artifacts, git

    git(tmp_path, "init", "-q")
    case = {"skill": "large-data-read", "kind": "mcp", "facts": []}
    for answer, expected in (
        ("Mean 1.0000142714285714 across all 70,000,000 values.", True),
        ("Approximate mean 1.000000 from a 1% sample.", True),
        ("Exact mean 1.000000 for the complete file.", False),
    ):
        checks = check_artifacts(tmp_path, case, answer, {})
        assert checks["qualified_mean_and_coverage"] is expected


def test_task_verification_accepts_report_beside_its_task(tmp_path):
    from codex_fixtures import check_artifacts, git

    git(tmp_path, "init", "-q")
    report = tmp_path / ".research/tasks/task-02/verification.md"
    report.parent.mkdir(parents=True)
    report.write_text("Recomputed mean is 2. Statistical significance is unsupported.")
    case = {
        "skill": "clio-kit-materio-task-verifier",
        "kind": "verification",
        "facts": [],
    }
    assert check_artifacts(tmp_path, case, "", {})["incorrect_mean_rejected"]
