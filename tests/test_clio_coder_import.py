"""The imported library must survive individual and wheel installation."""

import hashlib
import json
import importlib.util
from pathlib import Path
import subprocess

import pytest

from clio_kit.federation import compile_catalogue
from clio_kit.skill_cli import install_skills, selected_skills

ROOT = Path(__file__).resolve().parents[1]
COLLECTION = ROOT / "skills" / "clio-coder-skills"


def test_imported_files_match_reviewed_manifest_and_individual_install(tmp_path):
    lock = json.loads((COLLECTION / "import-lock.json").read_text())
    files = {
        p.relative_to(COLLECTION).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in COLLECTION.rglob("*")
        if p.is_file() and p.name != "import-lock.json"
    }
    assert files == lock["files"]
    skills = selected_skills((), "clio-coder")
    assert set(skills) == {record["name"] for record in lock["skills"]}
    upstream_names = {record["upstream_name"] for record in lock["skills"]}
    assert not set(skills) & upstream_names
    assert all(
        record["name"] == "clio-kit-" + record["upstream_name"]
        for record in lock["skills"]
    )
    install_skills(skills, tmp_path, False)
    for name, source in skills.items():
        for path in source.rglob("*"):
            if path.is_file():
                assert (
                    tmp_path / name / path.relative_to(source)
                ).read_bytes() == path.read_bytes()
        if name.startswith("clio-kit-materio-"):
            assert (tmp_path / name / "assets/references/research-policy.md").is_file()
            assert "../../assets/" not in (tmp_path / name / "SKILL.md").read_text()


def test_federation_accepts_marketplace_defined_skill_packages(tmp_path):
    skill = tmp_path / "library" / "scientific-debugging"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: scientific-debugging\ndescription: Diagnose scientific failures.\n---\nInspect evidence.\n"
    )
    catalogue = {
        "name": "coder",
        "plugins": [
            {
                "name": "scientific-debugging",
                "description": "Scientific debugging",
                "source": "./library/scientific-debugging",
            }
        ],
    }
    result = compile_catalogue(
        catalogue,
        url="https://example.org/library.git",
        revision="abc",
        checkout=tmp_path,
    )
    assert result[0]["source"]["path"] == "library/scientific-debugging"
    assert result[0]["source"]["sha"] == "abc"
    (skill / "agents").mkdir()
    with pytest.raises(ValueError, match="Native components"):
        compile_catalogue(
            catalogue,
            url="https://example.org/library.git",
            revision="abc",
            checkout=tmp_path,
        )

    (skill / "agents").rmdir()
    (skill / "SKILL.md").unlink()
    with pytest.raises(ValueError, match="manifest missing"):
        compile_catalogue(
            catalogue,
            url="https://example.org/library.git",
            revision="abc",
            checkout=tmp_path,
        )


def test_refresh_preserves_local_changes_and_checks_source_revision(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "skill_import", ROOT / "scripts/import_clio_coder_skills.py"
    )
    importer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(importer)
    source = tmp_path / "upstream"
    skill = source / "library/skills/research/example"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: example\ndescription: Inspect experimental inputs.\nlicense: MIT\n---\nRead references/input.md.\n"
    )
    (skill / "references").mkdir()
    (skill / "references/input.md").write_text("Retain input identity.\n")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@localhost",
            "commit",
            "-qm",
            "Skill",
        ],
        cwd=source,
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    target = tmp_path / "imported"
    with pytest.raises(ValueError, match="Expected"):
        importer.generate(source, target, "wrong-revision")
    first = importer.generate(source, target, revision)
    assert importer.generate(source, target, revision) == first
    copied = target / "skills/clio-kit-example/references/input.md"
    copied.write_text("Local adaptation")
    with pytest.raises(ValueError, match="local changes"):
        importer.generate(source, target, revision)
    assert copied.read_text() == "Local adaptation"


def test_adapted_invocations_preserve_program_and_native_agent_names():
    spec = importlib.util.spec_from_file_location(
        "skill_import", ROOT / "scripts/import_clio_coder_skills.py"
    )
    importer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(importer)
    original = "/skill tdd; /skill:ship pr; requires skill:tdd; the `tdd` skill. Run `herdr`; dispatch materio-task-executor."
    assert importer.adapt_references(
        original, {"tdd", "ship", "herdr", "materio-task-executor"}
    ) == (
        "/skill clio-kit-tdd; /skill:clio-kit-ship pr; requires skill:clio-kit-tdd; "
        "the `clio-kit-tdd` skill. Run `herdr`; dispatch materio-task-executor."
    )
