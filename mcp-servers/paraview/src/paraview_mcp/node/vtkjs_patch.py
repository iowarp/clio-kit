"""Guard a null matrix in the vtk.js that trame-vtk serves to the browser.

vtk.js (bundled in trame-vtk 2.11.17, the newest release) draws an actor placed in
display coordinates, such as the colour legend ParaView sends for local rendering, by
multiplying with the mapper's inverse shift-and-scale matrix. That matrix is ``null``
whenever coordinate shift and scale is off, and which way it goes can change when the
buffers are rebuilt for a new size: after the view grew (the canvas maximized) every
frame threw ``Cannot read properties of null (reading '0')`` and the view stayed black.
With shift and scale off the inverse is the identity, so skipping the multiply is the
correct result. The bundle is minified, so the branch is matched by its shape, not by
names, and a bundle without that shape is reported, never patched blindly.
"""

from __future__ import annotations

import re
from pathlib import Path

BUNDLE = Path("trame_vtk/modules/common/serve/trame-vtk.js")

# The display-coordinate branch: ...tmpMat4[13]=-1,MUL(X.tmpMat4,X.tmpMat4,INV),X.set...
_BRANCH = re.compile(
    r"(tmpMat4\[13\]=-1,)(\w+\((\w+)\.tmpMat4,\3\.tmpMat4,(\w+)\))(,\w+\.setUniformMatrix\(\"MCPCMatrix\")"
)
_GUARDED = re.compile(r"tmpMat4\[13\]=-1,\w+&&\w+\((\w+)\.tmpMat4,\1\.tmpMat4,\w+\)")


def patch_text(text: str) -> tuple[str, str]:
    """``(text, outcome)``: outcome is ``patched``, ``already`` or ``branch_not_found``."""
    if _GUARDED.search(text):
        return text, "already"
    patched, count = _BRANCH.subn(r"\1\4&&\2\5", text)
    if count != 1:
        return text, "branch_not_found"
    return patched, "patched"


def apply(site: Path) -> str:
    """Patch the bundle under ``site``; the outcome names what happened, for the node log."""
    bundle = site / BUNDLE
    if not bundle.is_file():
        return "bundle_missing"
    text, outcome = patch_text(bundle.read_text(encoding="utf-8"))
    if outcome == "patched":
        bundle.write_text(text, encoding="utf-8")
    return outcome
