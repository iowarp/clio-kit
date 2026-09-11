"""Portable installation preserves complete skills and existing user content."""

import json

import pytest
import yaml
from click.testing import CliRunner

from clio_kit.skill_cli import (
    install_skills,
    selected_skills,
    skill_group,
    skill_inventory,
)
from clio_kit.skills import SkillProblem


def test_shipped_skills_use_standard_metadata():
    inventory = skill_inventory()
    assert len(inventory) == 20
    for name, source in inventory.items():
        fields = yaml.safe_load((source / "SKILL.md").read_text().split("---", 2)[1])
        assert set(fields) <= {
            "name",
            "description",
            "metadata",
            "license",
            "compatibility",
            "allowed-tools",
        }
        assert fields["name"] == name
        assert all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in fields["metadata"].items()
        )


def test_selection_and_cli_errors(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        skill_group, ["list", "--bundle", "clio-scientific-io", "--json"]
    )
    assert result.exit_code == 0, result.output
    assert len(json.loads(result.output)) == 3
    result = runner.invoke(
        skill_group, ["install", "unknown-skill", "--target", str(tmp_path / "target")]
    )
    assert result.exit_code != 0 and "Unknown skills" in result.output
    assert not (tmp_path / "target").exists()
    with pytest.raises(SkillProblem, match="do not belong"):
        selected_skills(
            ("choosing-the-right-chart", "writing-slurm-job-scripts"), "clio-hpc"
        )


def test_complete_copy_idempotence_and_explicit_replace(tmp_path):
    skills = selected_skills(("choosing-the-right-chart",), None)
    name, source = next(iter(skills.items()))
    target = tmp_path / "skills"
    target.mkdir()
    unrelated = target / "my-notes.txt"
    unrelated.write_text("keep me")
    install_skills(skills, target, False)
    for file in source.rglob("*"):
        if file.is_file():
            assert (
                target / name / file.relative_to(source)
            ).read_bytes() == file.read_bytes()
    installed = target / name / "SKILL.md"
    before = installed.stat().st_mtime_ns
    install_skills(skills, target, False)
    assert installed.stat().st_mtime_ns == before
    installed.write_text("my local changes")
    with pytest.raises(SkillProblem, match="different files"):
        install_skills(skills, target, False)
    assert installed.read_text() == "my local changes"
    install_skills(skills, target, True)
    assert installed.read_bytes() == (source / "SKILL.md").read_bytes()
    assert unrelated.read_text() == "keep me"


def test_conflicts_preflight_before_any_skill_is_written(tmp_path):
    skills = selected_skills((), "clio-scientific-io")
    names = list(skills)
    conflict = tmp_path / names[-1]
    conflict.mkdir()
    (conflict / "SKILL.md").write_text("local work")
    with pytest.raises(SkillProblem, match="different files"):
        install_skills(skills, tmp_path, False)
    assert not (tmp_path / names[0]).exists()
    assert (conflict / "SKILL.md").read_text() == "local work"


def test_rejects_linked_destination_and_nested_source(tmp_path):
    skills = selected_skills(("choosing-the-right-chart",), None)
    name, source = next(iter(skills.items()))
    (tmp_path / name).symlink_to(source, target_is_directory=True)
    with pytest.raises(SkillProblem, match="not a regular directory"):
        install_skills(skills, tmp_path, True)
    with pytest.raises(SkillProblem, match="inside a source"):
        install_skills(skills, source / "nested", False)


def test_standalone_validation_requires_no_plugin_manifest():
    source = next(iter(selected_skills(("choosing-the-right-chart",), None).values()))
    result = CliRunner().invoke(skill_group, ["validate", str(source)])
    assert result.exit_code == 0, result.output


def test_linked_contents_rejected_before_replacing_other_skills(tmp_path):
    skills = selected_skills((), "clio-scientific-io")
    names = list(skills)
    conflict = tmp_path / names[-1]
    conflict.mkdir()
    (conflict / "linked-file").symlink_to(skills[names[-1]] / "SKILL.md")
    with pytest.raises(SkillProblem, match="linked skill content"):
        install_skills(skills, tmp_path, True)
    assert not (tmp_path / names[0]).exists()
    assert (conflict / "linked-file").is_symlink()
