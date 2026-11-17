"""The RPC socket's folder keeps the socket path within the Unix limit."""

from __future__ import annotations

from pathlib import Path

from avogadro_mcp import session


def test_a_short_folder_is_kept(tmp_path: Path) -> None:
    d = session.socket_dir(tmp_path / "tmp")
    assert d == tmp_path / "tmp" and d.is_dir()


def test_a_deep_folder_moves_the_socket_under_tmp(tmp_path: Path) -> None:
    deep = tmp_path / ("x" * 120) / "tmp"
    d = session.socket_dir(deep)
    try:
        assert len(str(d / session.RPC_NAME)) <= session.SOCKET_PATH_MAX
        assert str(d).startswith("/tmp/clio-avo-") and not deep.exists()
    finally:
        d.rmdir()


def test_the_vendored_novnc_modules_are_all_present() -> None:
    """Every module noVNC imports is shipped (a missing one leaves the view blank)."""
    import re

    root = Path(session.__file__).parent / "novnc"
    missing = []
    for js in root.rglob("*.js"):
        for target in re.findall(r'from\s+"(\.[^"]+)"', js.read_text()):
            if not (js.parent / target).resolve().is_file():
                missing.append(f"{js.relative_to(root)} -> {target}")
    assert not missing, missing


def test_a_release_the_install_skill_unpacked_is_found(tmp_path: Path, monkeypatch) -> None:
    """Installed in one conversation, found in every other (no per-conversation path)."""
    from avogadro_mcp import install

    for v in ("2.0.0", "2.1.0"):
        run = tmp_path / ".local" / "opt" / f"avogadro-{v}" / "squashfs-root" / "AppRun"
        run.parent.mkdir(parents=True)
        run.write_text("#!/bin/sh\n")
        run.chmod(0o755)
    monkeypatch.delenv(install.ENV, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(install.Path, "home", staticmethod(lambda: tmp_path))
    got = install.find()
    assert got["found"] and got["source"] == "install-avogadro"
    assert "avogadro-2.1.0" in got["command"]
