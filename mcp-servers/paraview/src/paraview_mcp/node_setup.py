"""What the front sends a node half's machine: its code, and the shell steps
that probe the machine and build the node half's packages there."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shlex
import tarfile
from pathlib import Path
from typing import Any

from . import config

NODE_SOURCE = Path(__file__).resolve().parent / "node"
RUN_NODE = """import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paraview_node.main import main
main()
"""


def code_bundle() -> tuple[bytes, str]:
    """The node half's code as a gzip tarball, and its content hash."""
    files = sorted(p for p in NODE_SOURCE.glob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update(RUN_NODE.encode())
    code_hash = digest.hexdigest()[:16]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:

        def add(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(data))

        for path in files:
            add(f"paraview_node/{path.name}", path.read_bytes())
        add("run_node.py", RUN_NODE.encode())
        add("HASH", code_hash.encode())
    return buffer.getvalue(), code_hash


def parse_markers(text: str) -> dict[str, str]:
    """Read ``CLIO_<KEY>=value`` lines a probe script printed."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("CLIO_") and "=" in line:
            key, _, value = line.partition("=")
            out[key[5:]] = value.strip()
    return out


def local_folder(version: str) -> str:
    """The node half's folder on this machine: clio-kit's own data folder.

    On another machine it is ``~/.clio/paraview/<version>/``. Here it sits under
    ``CLIO_KIT_CACHE_DIR`` when the host sets it, as the ParaView client does,
    else in clio-kit's data folder. A host that confines the servers it launches
    grants and keeps that cache folder; inside a Windows AppContainer the data
    folder resolves to a per-launch profile the fence does not let them write.
    """
    configured = os.environ.get("CLIO_KIT_CACHE_DIR", "").strip()
    if configured:
        return str(Path(configured).expanduser() / "paraview-node" / version)
    import platformdirs

    base = platformdirs.user_data_dir("clio-kit", appauthor=False)
    return str(Path(base) / "paraview" / version)


def node_base(pvserver: str | None) -> str:
    """The record name of the node half for ``pvserver``.

    A node half joins one pvserver for its life, so each pvserver has its own
    record, lock and log: a view of one never stops the node half of another.
    """
    if not pvserver:
        return "node"
    return "node-pv" + hashlib.sha1(pvserver.encode()).hexdigest()[:8]


def probe_script(version: str, folder: str | None = None, base: str = "node") -> str:
    where = shlex.quote(folder) if folder else f'"$HOME/.clio/paraview/{version}"'
    return f"""
D={where}
mkdir -p "$D" && chmod 700 "$D"
# A new server version prepares a new folder; its scenes come from the newest older one.
if [ ! -d "$D/scenes" ]; then
  for old in $(ls -d "$D"/../*/scenes 2>/dev/null | sort -V -r); do
    cp -R "$old" "$D/scenes" && break
  done
fi
echo "CLIO_DIR=$D"
echo "CLIO_PVPYTHON=$(command -v pvpython 2>/dev/null || true)"
[ -f "$D/env.json" ] && echo "CLIO_ENV=$(cat "$D/env.json")"
[ -f "$D/code/HASH" ] && echo "CLIO_CODE=$(cat "$D/code/HASH")"
[ -f "$D/{base}.json" ] && echo "CLIO_NODE=$(cat "$D/{base}.json")"
# A node half answers only inside its own network namespace, as sandboxed servers have.
NS=$(readlink /proc/self/ns/net 2>/dev/null | tr -cd 0-9)
echo "CLIO_NS=$NS"
if [ -n "$NS" ]; then
  [ -f "$D/{base}-$NS.json" ] && echo "CLIO_OWNNODE=$(cat "$D/{base}-$NS.json")"
  if [ -f "$D/{base}.lock" ] && command -v flock >/dev/null 2>&1; then
    if flock -n "$D/{base}.lock" true; then echo "CLIO_LOCKED=0"; else echo "CLIO_LOCKED=1"; fi
  fi
fi
exit 0
"""


def node_slot(
    marks: dict[str, str], base: str = "node"
) -> tuple[str, dict[str, Any] | None]:
    """The record name this front uses, and the node half it names, if any.

    A node half answers only inside its own network namespace, and its pid
    means nothing outside its pid namespace. A host that confines each server
    it launches in fresh namespaces can leave a node half another server
    started holding the folder. That one is never linked to or stopped: this
    front runs its own beside it, under ``<base>-<namespace>``.
    """
    node = json.loads(marks["NODE"]) if marks.get("NODE") else None
    ns = marks.get("NS", "")
    if not ns:
        return base, node
    if marks.get("OWNNODE"):
        return f"{base}-{ns}", json.loads(marks["OWNNODE"])
    held = marks.get("LOCKED") == "1"
    theirs = node is not None and str(node.get("net") or ns) != ns
    if theirs and not held:
        return base, None
    if theirs or (node is None and held):
        return f"{base}-{ns}", None
    return base, node


PYINFO = (
    "import json,platform,sys,sysconfig;"
    "print('CLIO_PYINFO='+json.dumps({'python':'%d.%d'%sys.version_info[:2],"
    "'platform':sysconfig.get_platform(),'machine':platform.machine(),"
    "'libc':list(platform.libc_ver())}))"
)

GL_PROBE = (
    "from paraview.simple import *\n"
    "v=CreateRenderView();Show(Sphere(),v);Render(v)\n"
    "print('CLIO_GL='+('ok' if v.GetRenderWindow().SupportsOpenGL() else 'no'))"
)


def gl_probe_script(pvpython: str) -> str:
    """Shell lines that report whether offscreen rendering works natively.

    Login and compute nodes often have no X server, EGL or system OSMesa. The
    Kitware binaries ship their own OSMesa behind the launcher's ``--mesa``
    switch, so the probe tries that when native rendering fails.
    """
    pv = shlex.quote(pvpython)
    probe = shlex.quote(GL_PROBE)
    return (
        "echo 'Trying offscreen rendering' >&2\n"
        f"if {pv} --force-offscreen-rendering -c {probe} "
        "| grep -q '^CLIO_GL=ok'; then echo CLIO_GLMODE=native\n"
        f"elif echo 'Trying the Mesa that ParaView ships' >&2; {pv} --mesa "
        f"--force-offscreen-rendering -c {probe} "
        "| grep -q '^CLIO_GL=ok'; then echo CLIO_GLMODE=mesa\n"
        "else echo CLIO_GLMODE=none; fi"
    )


def platform_tags(info: dict[str, Any]) -> tuple[list[str], str]:
    """pip ``--platform`` values and the uv ``--python-platform`` for pvpython."""
    machine = info.get("machine") or "x86_64"
    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine.lower(), machine)
    plat = str(info.get("platform", ""))
    if plat.startswith("win"):
        return (["win_amd64"], "x86_64-pc-windows-msvc")
    if plat.startswith("macosx"):
        tag_machine = "arm64" if machine == "aarch64" else machine
        return ([f"macosx_11_0_{tag_machine}"], f"{machine}-apple-darwin")
    libc = info.get("libc") or ["glibc", "2.17"]
    try:
        major, minor = (int(p) for p in str(libc[1]).split(".")[:2])
    except (ValueError, IndexError):
        major, minor = 2, 17
    pip = [
        f"manylinux_{major}_{m}_{machine}"
        for m in sorted({minor, 28, 17})
        if m <= minor
    ]
    pip.append(f"manylinux2014_{machine}")
    uv_minor = 28 if (major, minor) >= (2, 28) else 17
    return (sorted(set(pip)), f"{machine}-manylinux_2_{uv_minor}")


def build_script(folder: str, pyinfo: dict[str, Any], pvpython: str | None) -> str:
    """Install the trame packages into ``folder/site`` for pvpython's Python."""
    pip_plats, uv_plat = platform_tags(pyinfo)
    packages = " ".join(config.NODE_PACKAGES)
    plats = " ".join(f"--platform {p}" for p in pip_plats)
    pyver = pyinfo["python"]
    # Not --quiet: each line an install prints is the sign it has not stalled.
    pip_args = (
        f"install --disable-pip-version-check --no-input --progress-bar off "
        f'--timeout 20 --retries 1 --target "$T" --python-version {pyver} {plats} '
        f"--only-binary=:all: {packages}"
    )
    pv = shlex.quote(pvpython) if pvpython else '""'
    return f"""
D={shlex.quote(folder)}
T="$D/site.building.$$"
rm -rf "$T"; mkdir -p "$T"
ok=1
if command -v uv >/dev/null 2>&1; then
  echo "Installing with uv" >&2
  UV_HTTP_TIMEOUT=20 uv pip install --target "$T" --python-version {pyver} \\
    --python-platform {uv_plat} {packages} && ok=0
fi
if [ $ok -ne 0 ] && python3 -m pip --version >/dev/null 2>&1; then
  echo "Installing with pip" >&2
  python3 -m pip {pip_args} && ok=0
fi
if [ $ok -ne 0 ] && [ -n {pv} ] && {pv} -m pip --version >/dev/null 2>&1; then
  echo "Installing with pvpython's pip" >&2
  {pv} -m pip {pip_args} && ok=0
fi
if [ $ok -ne 0 ]; then rm -rf "$T"; echo "CLIO_BUILD=failed"; exit 3; fi
rm -rf "$D/site"; mv "$T" "$D/site"
echo "CLIO_BUILD=ok"
"""
