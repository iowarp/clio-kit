"""Unpublished components come from the launcher's own source, verified by the catalogue."""

import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import runpy
import subprocess
import tarfile
import threading
from urllib.error import HTTPError

import pytest

from clio_kit import component_source as source_module
from clio_kit import component_store as store
from clio_kit.component_source import Source, archive_url, source_from_direct_url

ROOT = Path(__file__).resolve().parents[1]
BUILD = runpy.run_path(str(ROOT / "scripts/package_components.py"))["build_components"]
COMMIT = "0123456789abcdef0123456789abcdef01234567"


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D401 - keep pytest output clean
        pass


@pytest.fixture
def release(tmp_path, monkeypatch):
    """A built catalogue served from loopback HTTP as its release assets."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text('[project]\nversion="1.0.0"\n')
    (source / "LICENSE").write_text("Repository licence\n")
    server = source / "mcp-servers/hdf5"
    (server / "src").mkdir(parents=True)
    (server / "clio-server.toml").write_text(
        'name="hdf5"\nruntime="python"\nentry="hdf5-mcp"\n'
    )
    (server / "uv.lock").write_text("version = 1\n")
    (server / "src/run.sh").write_text("#!/bin/sh\necho hdf5\n")
    (server / "src/run.sh").chmod(0o755)
    package = source / "plugins/lab"
    (package / ".claude-plugin").mkdir(parents=True)
    (package / ".claude-plugin/plugin.json").write_text(
        json.dumps({"name": "lab", "description": "Lab workflow", "version": "1.0.0"})
    )
    skill = package / "skills/reading-lab-data"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        '---\nname: reading-lab-data\ndescription: Use when reading lab data. Triggers on "lab".\n---\nRead it.\n'
    )
    assets = tmp_path / "assets"
    index = BUILD(source, assets)
    handler = functools.partial(_Quiet, directory=str(assets))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    index["base_url"] = f"http://127.0.0.1:{httpd.server_address[1]}"
    index_file = tmp_path / "index.json"
    index_file.write_text(json.dumps(index))
    monkeypatch.setattr(store, "INDEX_FILE", index_file)
    monkeypatch.delenv("CLIO_KIT_COMPONENT_BASE_URL", raising=False)
    monkeypatch.delenv("CLIO_KIT_OFFLINE", raising=False)
    monkeypatch.setenv("CLIO_KIT_CACHE_DIR", str(tmp_path / "cache"))
    yield index, assets, source
    httpd.shutdown()


def _unpublish(index, assets, key):
    (assets / index["artifacts"][key]["file"]).unlink()


def _installed(monkeypatch, found):
    monkeypatch.setattr(source_module, "installed_source", lambda: found)


def _commit_archive(source: Path, destination: Path) -> Path:
    """A GitHub-style archive: every path under one top folder."""
    with tarfile.open(destination, "w:gz") as tar:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                tar.add(path, f"repo-{COMMIT}/{path.relative_to(source).as_posix()}")
    return destination


def test_catalogue_names_each_artifacts_source_folder(release):
    index, _, _ = release
    sources = {key: record["source"] for key, record in index["artifacts"].items()}
    assert sources == {
        "server/hdf5": "mcp-servers/hdf5",
        "package/lab": "plugins/lab",
        "skill/reading-lab-data": "plugins/lab/skills/reading-lab-data",
    }


def test_a_published_component_downloads_without_reading_the_source(
    release, monkeypatch, capsys
):
    _installed(monkeypatch, Source("directory", "/nonexistent"))
    target = store.fetch("server/hdf5")
    assert (target / "src/run.sh").read_text() == "#!/bin/sh\necho hdf5\n"
    assert "not published" not in capsys.readouterr().err


def test_an_unpublished_component_comes_from_the_source_folder(
    release, monkeypatch, capsys
):
    index, assets, source = release
    _unpublish(index, assets, "server/hdf5")
    _installed(monkeypatch, Source("directory", str(source)))
    target = store.fetch("server/hdf5")
    assert (target / "LICENSE").read_text() == "Repository licence\n"
    assert (target / "src/run.sh").stat().st_mode & 0o777 == 0o755
    assert (
        "server/hdf5 is not published for CLIO Kit 1.0.0; taking it from its source folder"
        in (capsys.readouterr().err)
    )
    # The cached component is reused like a downloaded one, offline included.
    monkeypatch.setenv("CLIO_KIT_OFFLINE", "1")
    assert store.fetch("server/hdf5") == target


def test_an_unpublished_component_comes_from_the_source_commit_once(
    release, monkeypatch, tmp_path
):
    index, assets, source = release
    for key in ("server/hdf5", "package/lab"):
        _unpublish(index, assets, key)
    archive = _commit_archive(source, tmp_path / "commit.tar.gz")
    requested = []

    def url(found):
        requested.append(found.commit)
        return archive.as_uri()

    monkeypatch.setattr(source_module, "archive_url", url)
    _installed(monkeypatch, Source("git", "https://github.com/iowarp/clio-kit", COMMIT))
    server = store.fetch("server/hdf5")
    assert (server / "clio-server.toml").is_file()
    assert (server / "LICENSE").read_text() == "Repository licence\n"
    archive.unlink()  # The second component reads the cached commit archive.
    package = store.fetch("package/lab")
    assert (package / ".claude-plugin/plugin.json").is_file()
    assert not (package / "skills").exists()  # Skills are separate artifacts.
    assert requested == [COMMIT]
    cached = sorted((tmp_path / "cache/sources").glob("*.tar.gz"))
    assert [path.name for path in cached] == [f"{COMMIT}.tar.gz"]


def test_a_source_that_differs_from_the_catalogue_is_refused(release, monkeypatch):
    index, assets, source = release
    _unpublish(index, assets, "server/hdf5")
    (source / "mcp-servers/hdf5/uv.lock").write_text("version = 2\n")
    _installed(monkeypatch, Source("directory", str(source)))
    with pytest.raises(ValueError, match="differs from the catalogue"):
        store.fetch("server/hdf5")
    assert not store.artifact_path("server/hdf5").exists()


def test_a_source_missing_a_file_is_refused(release, monkeypatch):
    index, assets, source = release
    _unpublish(index, assets, "server/hdf5")
    (source / "mcp-servers/hdf5/src/run.sh").unlink()
    _installed(monkeypatch, Source("directory", str(source)))
    with pytest.raises(ValueError, match="lacks mcp-servers/hdf5/src/run.sh"):
        store.fetch("server/hdf5")
    assert not store.artifact_path("server/hdf5").exists()


def test_a_mirror_or_a_registry_install_never_falls_back(release, monkeypatch):
    index, assets, source = release
    _unpublish(index, assets, "server/hdf5")
    _installed(monkeypatch, None)
    with pytest.raises(ValueError, match="Cannot download server/hdf5 .*404"):
        store.fetch("server/hdf5")
    _installed(monkeypatch, Source("directory", str(source)))
    monkeypatch.setenv("CLIO_KIT_COMPONENT_BASE_URL", assets.as_uri())
    with pytest.raises(ValueError, match="Cannot download server/hdf5"):
        store.fetch("server/hdf5")


def test_only_a_missing_asset_falls_back(monkeypatch):
    _installed(monkeypatch, Source("directory", "/checkout"))
    monkeypatch.delenv("CLIO_KIT_COMPONENT_BASE_URL", raising=False)
    missing = HTTPError("https://example.org/a", 404, "Not Found", {}, io.BytesIO())
    failing = HTTPError("https://example.org/a", 503, "Unavailable", {}, io.BytesIO())
    assert store._unpublished_source(missing) == Source("directory", "/checkout")
    assert store._unpublished_source(failing) is None
    assert store._unpublished_source(ConnectionResetError()) is None


def test_a_catalogue_without_source_folders_cannot_fall_back(release, monkeypatch):
    index, assets, source = release
    _unpublish(index, assets, "server/hdf5")
    del index["artifacts"]["server/hdf5"]["source"]
    store.INDEX_FILE.write_text(json.dumps(index))
    _installed(monkeypatch, Source("directory", str(source)))
    with pytest.raises(ValueError, match="does not name the component's source folder"):
        store.fetch("server/hdf5")


@pytest.mark.parametrize(
    "record,expected",
    [
        (
            {
                "url": "https://github.com/iowarp/clio-kit",
                "vcs_info": {"vcs": "git", "commit_id": COMMIT},
            },
            Source("git", "https://github.com/iowarp/clio-kit", COMMIT),
        ),
        (
            {
                "url": "https://github.com/iowarp/clio-kit",
                "vcs_info": {"vcs": "git", "commit_id": "main"},
            },
            None,
        ),
        (
            {
                "url": "https://hg.example.org/kit",
                "vcs_info": {"vcs": "hg", "commit_id": COMMIT},
            },
            None,
        ),
        (
            {"url": "file:///work/clio-kit", "dir_info": {}},
            Source("directory", "/work/clio-kit"),
        ),
        ({"url": "file:///work/clio-kit", "dir_info": {"editable": True}}, None),
        ({"url": "https://files.example.org/clio_kit.whl", "archive_info": {}}, None),
    ],
)
def test_direct_url_records(record, expected):
    assert source_from_direct_url(json.dumps(record)) == expected


def test_direct_url_absent_or_malformed():
    assert source_from_direct_url(None) is None
    assert source_from_direct_url("{not json") is None


def test_commit_archives_come_from_github_or_git():
    found = Source("git", "https://github.com/iowarp/clio-kit.git", COMMIT)
    assert archive_url(found) == (
        f"https://codeload.github.com/iowarp/clio-kit/tar.gz/{COMMIT}"
    )
    other = Source("git", "https://gitlab.example.org/iowarp/clio-kit", COMMIT)
    assert archive_url(other) is None


def test_a_commit_on_another_git_host_is_fetched_with_git(
    release, monkeypatch, tmp_path
):
    index, assets, source = release
    _unpublish(index, assets, "server/hdf5")

    def git(*args):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
            cwd=source,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "source")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (source / "mcp-servers/hdf5/uv.lock").write_text("version = 2\n")  # Not committed.
    _installed(monkeypatch, Source("git", source.as_uri(), commit))
    target = store.fetch("server/hdf5")
    assert (target / "uv.lock").read_text() == "version = 1\n"
    assert (tmp_path / f"cache/sources/{commit}.tar.gz").is_file()
