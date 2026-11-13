"""Tests for the shipped clio-kit skill collection (`skills/`).

Covers discovery/packaging (the skill is a real directory clio-kit ships and
can list/print/install), the SKILL.md shape clio-agent's own runtime parser
expects, and a structural lint of every JSON tool-call example embedded in
`author-a2ui-surfaces/SKILL.md` -- the concrete rules that skill documents
(exactly one `root`, containers reference children by id, `clio.time-
series.v1` takes exactly one of `series`/`dataUri`) checked directly against
the example payloads, so a future edit that breaks one of them fails here
rather than silently shipping a broken worked example. The examples were
additionally validated once, out of band, against clio-agent's real compiled
`clio-workspace` catalog validator (schema + safety walk) -- see the PR
description for that evidence; clio-agent is not a dependency of this
repository, so that check does not run as part of this suite.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

import clio_kit.cli_entry  # noqa: F401 - import side effect attaches skill commands to `main`
from clio_kit import main
from clio_kit.skills import (
    SKILL_INSTALL_SCOPES,
    discover_skills,
    format_install_result,
    format_skill_listing,
    get_skills_path,
    install_skills,
    parse_frontmatter,
    skill_field,
    validate_skill,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_SKILLS_ROOT = REPOSITORY_ROOT / "skills"
A2UI_SKILL_DIR = SHIPPED_SKILLS_ROOT / "author-a2ui-surfaces"
A2UI_SKILL_MD = A2UI_SKILL_DIR / "SKILL.md"


# ---- packaging / discovery ---------------------------------------------


def test_shipped_skills_directory_exists() -> None:
    assert SHIPPED_SKILLS_ROOT.is_dir()
    assert A2UI_SKILL_MD.is_file()


def test_get_skills_path_finds_the_dev_checkout() -> None:
    assert get_skills_path() == SHIPPED_SKILLS_ROOT


def test_discover_skills_finds_the_a2ui_skill() -> None:
    found = discover_skills(SHIPPED_SKILLS_ROOT)
    assert "author-a2ui-surfaces" in found
    assert found["author-a2ui-surfaces"] == A2UI_SKILL_MD


def test_discover_skills_on_missing_root_is_empty(tmp_path: Path) -> None:
    assert discover_skills(tmp_path / "nope") == {}


def test_pyproject_ships_skills_as_shared_data_and_sdist_member() -> None:
    text = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"skills" = "skills"' in text
    assert '"skills/"' in text


# ---- frontmatter shape (what clio-agent's own SkillCatalog parses) ------


def test_parse_frontmatter_reads_the_a2ui_skill() -> None:
    text = A2UI_SKILL_MD.read_text(encoding="utf-8")
    fields = parse_frontmatter(text)
    assert fields["name"] == "author-a2ui-surfaces"
    assert fields["title"]
    assert "Use when" in fields["description"]


def test_parse_frontmatter_empty_without_leading_delimiter() -> None:
    assert parse_frontmatter("# just a heading\nno frontmatter here") == {}


def test_skill_field_missing_file_is_empty(tmp_path: Path) -> None:
    assert skill_field(tmp_path / "missing.md", "name") == ""


def test_validate_skill_a2ui_skill_is_sound() -> None:
    assert validate_skill(A2UI_SKILL_MD) == []


def test_validate_skill_catches_directory_name_mismatch(tmp_path: Path) -> None:
    skill_dir = tmp_path / "actual-dir-name"
    skill_dir.mkdir()
    manifest = skill_dir / "SKILL.md"
    manifest.write_text(
        "---\nname: different-name\ntitle: T\ndescription: D. Use when X.\n---\nBody.\n",
        encoding="utf-8",
    )
    problems = validate_skill(manifest)
    assert any("does not match directory" in problem for problem in problems)


def test_validate_skill_catches_missing_frontmatter_fields(tmp_path: Path) -> None:
    skill_dir = tmp_path / "bare"
    skill_dir.mkdir()
    manifest = skill_dir / "SKILL.md"
    manifest.write_text("---\nname: bare\n---\nBody.\n", encoding="utf-8")
    problems = validate_skill(manifest)
    assert any("'title'" in problem for problem in problems)
    assert any("'description'" in problem for problem in problems)


def test_validate_skill_catches_empty_body(tmp_path: Path) -> None:
    skill_dir = tmp_path / "empty-body"
    skill_dir.mkdir()
    manifest = skill_dir / "SKILL.md"
    manifest.write_text(
        "---\nname: empty-body\ntitle: T\ndescription: D. Use when X.\n---\n\n",
        encoding="utf-8",
    )
    problems = validate_skill(manifest)
    assert any("body is empty" in problem for problem in problems)


# ---- CLI-facing formatting ----------------------------------------------


def test_format_skill_listing_names_the_a2ui_skill() -> None:
    lines = format_skill_listing(SHIPPED_SKILLS_ROOT)
    joined = "\n".join(lines)
    assert "author-a2ui-surfaces" in joined


def test_format_skill_listing_empty_root(tmp_path: Path) -> None:
    assert format_skill_listing(tmp_path) == ["No skills found."]


# ---- CLI wiring (clio_kit.cli_entry attaches these to `main` at import) --


def test_cli_entry_attaches_skill_commands_to_main() -> None:
    for name in ("skills", "skill", "skills-install", "skills-validate"):
        assert name in main.commands


def test_main_skills_command_lists_the_a2ui_skill() -> None:
    result = CliRunner().invoke(main, ["skills"])
    assert result.exit_code == 0, result.output
    assert "author-a2ui-surfaces" in result.output


def test_main_skill_command_prints_the_a2ui_skill_body() -> None:
    result = CliRunner().invoke(main, ["skill", "author-a2ui-surfaces"])
    assert result.exit_code == 0, result.output
    assert "name: author-a2ui-surfaces" in result.output
    assert "create_a2ui_surface" in result.output


def test_main_skill_command_unknown_name_exits_nonzero() -> None:
    result = CliRunner().invoke(main, ["skill", "does-not-exist"])
    assert result.exit_code != 0
    assert "Unknown skill" in result.output


def test_main_skills_validate_command_passes() -> None:
    result = CliRunner().invoke(main, ["skills-validate"])
    assert result.exit_code == 0, result.output
    assert "OK:   author-a2ui-surfaces" in result.output


# ---- install ------------------------------------------------------------


def test_install_skills_copies_the_a2ui_skill(tmp_path: Path) -> None:
    destination = tmp_path / ".claude" / "skills"
    installed = install_skills(SHIPPED_SKILLS_ROOT, destination)
    assert installed == ["author-a2ui-surfaces"]
    assert (destination / "author-a2ui-surfaces" / "SKILL.md").is_file()


def test_install_skills_does_not_clear_unrelated_existing_files(tmp_path: Path) -> None:
    """Installing replaces only the skills it ships -- a user's own skill survives."""
    destination = tmp_path / ".claude" / "skills"
    own_skill = destination / "my-own-skill"
    own_skill.mkdir(parents=True)
    (own_skill / "SKILL.md").write_text(
        "---\nname: my-own-skill\ntitle: Mine\ndescription: D. Use when Y.\n---\nBody.\n",
        encoding="utf-8",
    )

    install_skills(SHIPPED_SKILLS_ROOT, destination)

    assert (own_skill / "SKILL.md").is_file()
    assert (destination / "author-a2ui-surfaces" / "SKILL.md").is_file()


def test_install_skills_replaces_a_stale_copy_wholesale(tmp_path: Path) -> None:
    destination = tmp_path / ".claude" / "skills"
    stale_dir = destination / "author-a2ui-surfaces"
    stale_dir.mkdir(parents=True)
    (stale_dir / "SKILL.md").write_text("stale", encoding="utf-8")
    (stale_dir / "leftover.txt").write_text("should not survive", encoding="utf-8")

    install_skills(SHIPPED_SKILLS_ROOT, destination)

    assert not (stale_dir / "leftover.txt").exists()
    assert "author-a2ui-surfaces" in (stale_dir / "SKILL.md").read_text(
        encoding="utf-8"
    )


def test_install_skills_on_empty_source_installs_nothing(tmp_path: Path) -> None:
    assert install_skills(tmp_path / "no-skills-here", tmp_path / "dest") == []


def test_format_install_result_user_scope_notes_no_auto_declaration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "home" / ".claude" / "skills"
    monkeypatch.setitem(SKILL_INSTALL_SCOPES, "user", destination)
    lines = format_install_result(SHIPPED_SKILLS_ROOT, "user")
    joined = "\n".join(lines)
    assert "author-a2ui-surfaces" in joined
    assert "not auto-declared" in joined


# ---- worked-example structural lint -------------------------------------
#
# Every fenced ```json block in the A2UI skill that carries a top-level
# "components" key is a real create/update tool-call payload. These checks
# encode, directly against the shipped text, the exact structural rules the
# skill's prose claims cause validation failures -- so an edit that breaks a
# worked example (a nested child, a time-series with both series and
# dataUri, a second/missing root) fails this suite instead of shipping.


def _json_blocks_with_components() -> list[dict]:
    text = A2UI_SKILL_MD.read_text(encoding="utf-8")
    raw_blocks = re.findall(r"```json\n(.*?)\n```", text, flags=re.DOTALL)
    assert raw_blocks, "expected at least one fenced json block in the A2UI skill"
    payloads = [json.loads(raw) for raw in raw_blocks]
    return [payload for payload in payloads if "components" in payload]


def test_skill_md_json_blocks_are_valid_json() -> None:
    payloads = _json_blocks_with_components()
    assert len(payloads) >= 3


def test_create_surface_example_has_exactly_one_root() -> None:
    """The `create_a2ui_surface`-shaped payloads (those with `catalog_id` or
    `data_model`) must carry exactly one root; a bare component-upsert payload
    (`update_a2ui_components`) has none, matching the skill's own claim that the
    root requirement is create-only."""
    for payload in _json_blocks_with_components():
        components = payload["components"]
        root_count = sum(1 for c in components if c.get("id") == "root")
        is_create_shaped = "catalog_id" in payload or "data_model" in payload
        if is_create_shaped:
            assert root_count == 1, payload["surface_id"]
        else:
            assert root_count == 0, payload["surface_id"]


def test_container_children_are_id_strings_never_inline_objects() -> None:
    container_kinds = {"Row", "Column", "Grid", "List", "Frame", "Tabs", "Modal"}
    for payload in _json_blocks_with_components():
        for component in payload["components"]:
            if component.get("component") not in container_kinds:
                continue
            children = component.get("children", [])
            assert isinstance(children, list)
            for child in children:
                assert isinstance(child, str), (
                    f"container {component['id']!r} nests an inline object instead "
                    "of an id reference"
                )


def test_every_component_has_a_unique_id_within_its_surface() -> None:
    for payload in _json_blocks_with_components():
        ids = [c["id"] for c in payload["components"]]
        assert len(ids) == len(set(ids)), payload["surface_id"]


def test_time_series_examples_set_exactly_one_of_series_or_datauri() -> None:
    found_a_time_series = False
    for payload in _json_blocks_with_components():
        for component in payload["components"]:
            if component.get("component") != "clio.time-series.v1":
                continue
            found_a_time_series = True
            has_series = "series" in component
            has_data_uri = "dataUri" in component
            assert has_series != has_data_uri, component["id"]
    assert found_a_time_series, (
        "expected the dashboard example to cover clio.time-series.v1"
    )


def test_metric_examples_carry_label_and_value() -> None:
    found_a_metric = False
    for payload in _json_blocks_with_components():
        for component in payload["components"]:
            if component.get("component") != "clio.metric.v1":
                continue
            found_a_metric = True
            assert "label" in component
            assert "value" in component
    assert found_a_metric, "expected the dashboard example to cover clio.metric.v1"


def test_map_example_never_carries_a_tile_or_style_url() -> None:
    for payload in _json_blocks_with_components():
        for component in payload["components"]:
            if component.get("component") != "clio.map.v1":
                continue
            forbidden = {"tileUrl", "styleUrl", "tileUrlTemplate", "mapStyle"}
            assert not (forbidden & component.keys()), component["id"]
            for point in component.get("points", []):
                assert {"id", "label", "latitude", "longitude"} <= point.keys()


def test_data_table_example_puts_the_title_on_a_sibling_text() -> None:
    for payload in _json_blocks_with_components():
        for component in payload["components"]:
            if component.get("component") != "clio.data-table.v1":
                continue
            assert "title" not in component
            assert "caption" not in component


@pytest.mark.parametrize("scope", sorted(SKILL_INSTALL_SCOPES))
def test_every_declared_scope_is_a_real_skill_search_root_shape(scope: str) -> None:
    """Both scopes install under a `.claude/skills` layout -- the same shape
    clio-agent's own `SkillCatalog._skill_search_roots` scans (workspace when
    relative to a project cwd, global under the user's home)."""
    root = SKILL_INSTALL_SCOPES[scope]
    assert root.parts[-2:] == (".claude", "skills")
