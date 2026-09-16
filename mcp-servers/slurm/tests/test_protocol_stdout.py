"""Diagnostic output must not share MCP's stdout transport."""

from slurm_mcp.implementation.job_submission import submit_slurm_job


def test_submission_leaves_stdout_available_for_json_rpc(tmp_path, capsys):
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/sh\necho finished\n")
    result = submit_slurm_job(str(script), cores=1)
    assert result["job_id"] == "12345"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "12345" in captured.err
