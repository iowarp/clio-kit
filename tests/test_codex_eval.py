"""Evaluation must not turn resource discovery or backend failures into success."""

import importlib.util
import json
from pathlib import Path
import sys

EVALS = Path(__file__).resolve().parents[1] / "evals"
sys.path.insert(0, str(EVALS))
spec = importlib.util.spec_from_file_location("clio_live_eval", EVALS / "codex_eval.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def event(item):
    return {"type": "item.completed", "item": item}


def test_native_skill_injection_requires_exact_installed_content(tmp_path):
    home, project = tmp_path / "home", tmp_path / "project"
    source = project / ".agents/skills/example/SKILL.md"
    source.parent.mkdir(parents=True)
    source.write_text("---\nname: example\n---\nInstructions.\n")
    sessions = home / "sessions"
    sessions.mkdir(parents=True)
    for role, content, expected in (
        ("user", source.read_text(), ["example"]),
        ("assistant", source.read_text(), []),
        ("user", "description only", []),
    ):
        (sessions / "run.jsonl").write_text(
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "role": role,
                        "content": [
                            {
                                "text": f"<skill>\n<name>example</name>\n<path>{source}</path>\n{content}\n</skill>"
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        assert runner.injected_skill_names(home, project) == expected


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


def test_handoff_check_rejects_secret_in_nested_output(tmp_path):
    from codex_fixtures import SECRET, check_artifacts, git

    git(tmp_path, "init", "-q")
    report = tmp_path / ".claude/handoffs/research.md"
    report.parent.mkdir(parents=True)
    report.write_text(SECRET)
    case = {"skill": "clio-kit-context-handoff", "kind": "handoff", "facts": []}
    assert not check_artifacts(tmp_path, case, "Credential redacted.", {})[
        "secret_not_reproduced"
    ]


def test_skill_activation_requires_content_not_only_listing():
    listing = event(
        {
            "type": "command_execution",
            "command": "rg --files .agents | rg SKILL.md",
            "exit_code": 0,
            "aggregated_output": ".agents/skills/example/SKILL.md",
        }
    )
    reading = event(
        {
            "type": "command_execution",
            "command": "cat .agents/skills/example/SKILL.md",
            "exit_code": 0,
            "aggregated_output": "---\nname: example\ndescription: Read the inputs.\n---\nInstructions",
        }
    )
    assert runner.summarize_events([listing])["skill_names_read"] == []
    assert runner.summarize_events([reading])["skill_names_read"] == ["example"]


def test_run_consumes_installed_project_settings_and_cleans_auth(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import tomli_w

    auth_home = tmp_path / "original-auth"
    auth_home.mkdir()
    (auth_home / "auth.json").write_text("{}")
    monkeypatch.setenv("CODEX_HOME", str(auth_home))

    def prepare(project, case):
        project.mkdir(parents=True)
        return {}

    monkeypatch.setattr(runner, "prepare", prepare)
    native = {
        "command": "lab-executable",
        "args": ["--profile", "research"],
        "env": {"LAB_PATH": "/site/lab"},
    }

    def install(root, name, client, project, **kwargs):
        config = project / ".codex/config.toml"
        config.parent.mkdir()
        config.write_text(tomli_w.dumps({"mcp_servers": {"clio-lab": native}}))
        return {
            "package": name,
            "servers": ["clio-lab"],
            "not_installed": ["lab: hooks"],
        }

    monkeypatch.setattr(runner, "install_for_client", install)
    monkeypatch.setattr(
        runner, "check_artifacts", lambda *args: {"source_preserved": True}
    )

    class Process:
        returncode = 0

        def __init__(self, command, **kwargs):
            config = runner.tomllib.loads(
                (kwargs["cwd"] / ".codex/config.toml").read_text()
            )
            settings = config["mcp_servers"]["clio-lab"]
            assert all(settings[k] == v for k, v in native.items())
            assert "mcp_servers" not in runner.tomllib.loads(
                (Path(kwargs["env"]["CODEX_HOME"]) / "config.toml").read_text()
            )
            kwargs["stdout"].write(
                json.dumps({"type": "turn.completed", "usage": {}}) + "\n"
            )

        def communicate(self, prompt, timeout):
            pass

    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    args = SimpleNamespace(
        output=tmp_path / "runs",
        model="test",
        network_access=False,
        invoke_skill=False,
        mcp_overrides={},
        sandbox="workspace-write",
        timeout=1,
        max_uncached_tokens=100000,
    )
    record = runner.run(
        {
            "skill": "lab-task",
            "package": "lab",
            "servers": ["lab"],
            "kind": "reasoning",
            "prompt": "Inspect",
            "facts": [],
        },
        "baseline",
        args,
    )
    assert record["config_route"] == "project"
    assert record["servers"] == ["clio-lab"]
    assert record["installation"]["not_installed"] == ["lab: hooks"]
    assert not (args.output / "lab-task/baseline/codex/auth.json").is_symlink()
    assert (auth_home / "auth.json").read_text() == "{}"


def test_auth_link_cleaned_on_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pytest

    auth = tmp_path / "auth.json"
    auth.write_text("{}")
    link = tmp_path / "case/skill/codex/auth.json"

    def fail(*args):
        link.parent.mkdir(parents=True)
        link.symlink_to(auth)
        raise RuntimeError("failed after login setup")

    monkeypatch.setattr(runner, "execute_run", fail)
    with pytest.raises(RuntimeError):
        runner.run({"skill": "case"}, "skill", SimpleNamespace(output=tmp_path))
    assert not link.is_symlink()
    assert auth.exists()


def test_backend_override_rejects_non_stdio_and_bad_environment(tmp_path):
    import pytest

    config = tmp_path / "backends.toml"
    for text in (
        '[mcp_servers.lab]\nurl="https://example.org"\n',
        "[mcp_servers.lab.env]\nLAB_PORT=9000\n",
    ):
        config.write_text(text)
        with pytest.raises(ValueError):
            runner.load_mcp_overrides(config)
    config.write_text('[mcp_servers.lab.env]\nLAB_PATH="/site/lab"\n')
    assert runner.load_mcp_overrides(config) == {
        "lab": {"env": {"LAB_PATH": "/site/lab"}}
    }


def test_report_separates_task_outcome_from_activation_and_backend_comparability(
    tmp_path,
):
    import json
    from codex_report import summarize

    record = {
        "skill": "example",
        "completed": True,
        "checks_passed": False,
        "servers": ["lab"],
        "model": "test",
        "prompt_sha256": "same",
        "source_hashes": {"input": "same"},
        "server_config_sha256": "first",
        "checks": {
            "source_preserved": True,
            "model_completed": True,
            "actual_mcp_call": False,
        },
    }
    for mode in ("baseline", "skill"):
        path = tmp_path / "example" / mode / "result.json"
        path.parent.mkdir(parents=True)
        data = {**record, "mode": mode}
        if mode == "skill":
            data["server_config_sha256"] = "different"
            data["checks"] = {
                "source_preserved": True,
                "model_completed": True,
                "skill_read": True,
            }
        path.write_text(json.dumps(data))
    report = summarize(tmp_path)
    assert report["outcome_screens_passed"] == 2
    assert report["pairs"][0]["baseline_checks"]["actual_mcp_call"] is False
    assert report["pairs"][0]["comparable"] is False
    assert report["median_observed_delta"] is None
