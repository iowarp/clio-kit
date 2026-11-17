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
