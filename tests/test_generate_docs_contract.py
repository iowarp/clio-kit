"""Regression tests for generated MCP site contract metadata."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_generator() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "generate_docs.py"
    spec = importlib.util.spec_from_file_location("clio_kit_generate_docs", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load documentation generator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator()
DocusaurusGenerator = GENERATOR.DocusaurusGenerator


def test_generation_from_website_directory_keeps_docs_at_repository_root(
    tmp_path, monkeypatch
):
    site = tmp_path / "website"
    site.mkdir()
    monkeypatch.chdir(site)
    DocusaurusGenerator(Path(".")).generate_all_docs({})
    assert (tmp_path / "docs/mcps").is_dir()
    assert not (site / "src/data/mcpData.js").exists()
    assert not (site / "docs").exists()


def test_generated_page_replaces_stale_description(tmp_path: Path) -> None:
    """A contract upgrade must not preserve an old generated description."""
    server = tmp_path / "server"
    server.mkdir()
    (server / "README.md").write_text("# Slurm\n", encoding="utf-8")
    output = tmp_path / "site"
    page = output.parent / "docs" / "mcps" / "slurm.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        '<MCPDetail description="stale v1 description" />\n', encoding="utf-8"
    )

    DocusaurusGenerator(output)._generate_mcp_markdown(
        {
            "name": "Slurm",
            "slug": "slurm",
            "category": "System Management",
            "description": "Fresh v3 agent contract",
            "icon": "scheduler",
            "version": "3.0.0",
            "actions": ["slurm_submit"],
            "platforms": ["claude"],
            "keywords": ["slurm"],
            "license": "BSD-3-Clause",
            "tools": [{"name": "slurm_submit", "description": "Submit one job."}],
            "path": str(server),
        }
    )

    rendered = page.read_text(encoding="utf-8")
    assert 'description="Fresh v3 agent contract"' in rendered
    assert "stale v1 description" not in rendered
    assert 'actions={["slurm_submit"]}' in rendered


def test_reference_generation_is_deterministic_without_legacy_catalogue(tmp_path):
    server = tmp_path / "server"
    server.mkdir()
    source_data = {
        "spack": {
            "name": "Spack",
            "slug": "spack",
            "category": "System Management",
            "description": "Authoritative Spack description",
            "icon": "packages",
            "version": "2.0.1",
            "updated": "2026-07-13",
            "actions": ["spack_install"],
            "platforms": ["claude"],
            "keywords": ["spack"],
            "license": "BSD-3-Clause",
            "tools": [],
            "path": str(server),
        }
    }
    first, second = tmp_path / "first/site", tmp_path / "second/site"
    for site in (first, second):
        DocusaurusGenerator(site).generate_all_docs(source_data)
        assert not (site / "src/data/mcpData.js").exists()
    assert (first.parent / "docs/mcps/spack.md").read_bytes() == (
        second.parent / "docs/mcps/spack.md"
    ).read_bytes()


@pytest.mark.parametrize("value", [None, "2026-7-13", "not-a-date"])
def test_documentation_date_must_be_explicit_and_canonical(value: object) -> None:
    """Wall-clock fallback cannot make generated docs drift between runs."""
    inventory = {"documentation": {"updated": value}}

    with pytest.raises(ValueError, match="documentation.updated"):
        GENERATOR.read_documentation_updated(inventory)


def test_regeneration_preserves_reviewed_usage_but_updates_contract(tmp_path):
    server = tmp_path / "server"
    server.mkdir()
    output = tmp_path / "site"
    page = output.parent / "docs" / "mcps" / "crystal.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "{/* clio-kit:usage:start */}\n\nReviewed usage and limits.\n\n"
        "{/* clio-kit:usage:end */}\n"
    )
    data = dict(
        name="Crystal",
        slug="crystal",
        category="Scientific",
        description="Updated",
        icon="",
        version="2.0.0",
        actions=["inspect"],
        platforms=["claude"],
        path=str(server),
    )
    generator = DocusaurusGenerator(output)
    generator._generate_mcp_markdown(data)
    first = page.read_text()
    assert 'version="2.0.0"' in first
    assert 'actions={["inspect"]}' in first
    assert "Reviewed usage and limits." in first
    assert "perform_operation" not in first
    generator._generate_mcp_markdown(data)
    assert page.read_text() == first


@pytest.mark.parametrize("runtime", ["node", "go"])
def test_docs_discover_descriptor_project_without_python_metadata(
    tmp_path, monkeypatch, runtime
):
    server = tmp_path / "crystal"
    server.mkdir()
    (server / "clio-server.toml").write_text(
        f'name = "crystal"\nruntime = "{runtime}"\nentry = "server"\n'
        'description = "Crystal analysis"\nversion = "1.0.0"\n'
    )
    extractor = GENERATOR.MCPDataExtractor({"crystal": "2.0.0"}, "2026-09-10")
    monkeypatch.setattr(
        extractor, "_extract_tools_from_server", lambda _: [{"name": "inspect"}]
    )
    data = extractor.extract_mcp_data(tmp_path)["crystal"]
    assert data["description"] == "Crystal analysis"
    assert data["version"] == "2.0.0"
    assert data["actions"] == ["inspect"]


def test_python_descriptor_preserves_project_documentation_metadata(
    tmp_path, monkeypatch
):
    server = tmp_path / "pandas"
    server.mkdir()
    (server / "clio-server.toml").write_text('name = "pandas"\nruntime = "python"\n')
    (server / "pyproject.toml").write_text(
        '[project]\nname = "pandas-mcp"\ndescription = "Scientific analysis"\n'
        'license = "BSD-3-Clause"\nkeywords = ["data-analysis"]\n'
    )
    extractor = GENERATOR.MCPDataExtractor({"pandas": "2.2.4"}, "2026-09-10")
    monkeypatch.setattr(extractor, "_extract_tools_from_server", lambda _: [])
    data = extractor._extract_single_mcp_data(server)
    assert data["description"] == "Scientific analysis"
    assert data["license"] == "BSD-3-Clause"
    assert data["keywords"] == ["data-analysis"]


def test_partial_metadata_extraction_cannot_publish_an_incomplete_site(
    tmp_path, monkeypatch, capsys
):
    servers = tmp_path / "mcp-servers"
    (servers / "pandas").mkdir(parents=True)
    (servers / "pandas/pyproject.toml").write_text('[project]\nname="pandas-mcp"\n')
    (tmp_path / "mcp-server-versions.toml").write_text(
        '[servers]\npandas="1.0.0"\n[documentation]\nupdated="2026-09-21"\n'
    )
    monkeypatch.setattr(
        GENERATOR.sys,
        "argv",
        ["generate_docs.py", str(servers), str(tmp_path / "site")],
    )
    monkeypatch.setattr(
        GENERATOR.MCPDataExtractor, "extract_mcp_data", lambda *args: {}
    )
    with pytest.raises(SystemExit) as error:
        GENERATOR.main()
    assert error.value.code == 1
    assert "extraction is incomplete" in capsys.readouterr().out
    assert not (tmp_path / "docs/mcps").exists()


def test_ci_runs_documentation_generator_in_its_locked_package_environment():
    workflow = (
        Path(__file__).resolve().parents[1] / ".github/workflows/docs-and-website.yml"
    ).read_text()
    assert "uv run --frozen python scripts/generate_docs.py" in workflow
