"""Two shape-matched fixes to the vtk.js and vue-vtk-js that trame-vtk serves to the browser.

Both bundles ship minified inside trame-vtk 2.11.17 (the newest release), so each
branch is matched by its shape, not by names, and a bundle without that shape is
reported, never patched blindly.

Display-actor guard. vtk.js draws an actor placed in display coordinates, such as
the colour legend ParaView sends for local rendering, by multiplying with the
mapper's inverse shift-and-scale matrix. That matrix is ``null`` whenever coordinate
shift and scale is off, and which way it goes can change when the buffers are
rebuilt for a new size: after the view grew (the canvas maximized) every frame threw
``Cannot read properties of null (reading '0')`` and the view stayed black. With
shift and scale off the inverse is the identity, so skipping the multiply is the
correct result.

Remote attach. A remote view shows its pictures only after a first one comes back
at the size the page measured when it mounted. ParaView renders one view for every
page that watches it, at the size the newest page asked for, so a page that mounts
while another page (a second App view, a reloaded frame whose old link is still
open) has asked for a different size gets every picture at that other size, keeps
asking for a new one, and stays blank for good while pictures keep arriving. The
patched page accepts the first well-formed picture, shows it, and when it came at
another size asks for its own size, so the next pictures fit it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

BUNDLE = Path("trame_vtk/modules/common/serve/trame-vtk.js")


@dataclass(frozen=True)
class Patch:
    name: str
    branch: re.Pattern[str]
    guarded: re.Pattern[str]
    replacement: str


# The display-coordinate branch: ...tmpMat4[13]=-1,MUL(X.tmpMat4,X.tmpMat4,INV),X.set...
DISPLAY_GUARD = Patch(
    "display-actor guard",
    re.compile(
        r"(tmpMat4\[13\]=-1,)(\w+\((\w+)\.tmpMat4,\3\.tmpMat4,(\w+)\))(,\w+\.setUniformMatrix\(\"MCPCMatrix\")"
    ),
    re.compile(r"tmpMat4\[13\]=-1,\w+&&\w+\((\w+)\.tmpMat4,\1\.tmpMat4,\w+\)"),
    r"\1\4&&\2\5",
)

# The mount handshake: W===X&&H===Y?(SUB.unsubscribe(),VIEW.getCanvasView().setBackgroundImage(IMG),DONE()):STREAM.render()
# The picture is already known to be well formed (its size matches its metadata).
REMOTE_ATTACH = Patch(
    "remote attach",
    re.compile(
        r"(\w+)===(\w+)&&(\w+)===(\w+)\?\((\w+)\.unsubscribe\(\),(\w+)\.getCanvasView\(\)\.setBackgroundImage\((\w+)\),"
        r"(\w+)\(\)\):\w+\.render\(\)"
    ),
    re.compile(
        r"\.setBackgroundImage\(\w+\),\w+\(\),\(\w+!==\w+\|\|\w+!==\w+\)&&setTimeout\("
    ),
    r"(\5.unsubscribe(),\6.getCanvasView().setBackgroundImage(\7),\8(),(\1!==\2||\3!==\4)&&setTimeout(()=>{"
    r"const clioCanvas=\6.getCanvasView(),[clioW,clioH]=clioCanvas.getSize();"
    r"clioCanvas.setSize(clioW+2,clioH+2),\6.resize()}))",
)

PATCHES = (DISPLAY_GUARD, REMOTE_ATTACH)


def patch_text(text: str, patch: Patch = DISPLAY_GUARD) -> tuple[str, str]:
    """``(text, outcome)``: outcome is ``patched``, ``already`` or ``branch_not_found``."""
    if patch.guarded.search(text):
        return text, "already"
    patched, count = patch.branch.subn(patch.replacement, text)
    if count != 1:
        return text, "branch_not_found"
    return patched, "patched"


def apply(site: Path) -> str:
    """Patch the bundle under ``site``; the outcome names what happened to each fix, for the node log."""
    bundle = site / BUNDLE
    if not bundle.is_file():
        return "bundle_missing"
    text = bundle.read_text(encoding="utf-8")
    outcomes = []
    changed = False
    for patch in PATCHES:
        text, outcome = patch_text(text, patch)
        changed = changed or outcome == "patched"
        outcomes.append(f"{patch.name}: {outcome}")
    if changed:
        bundle.write_text(text, encoding="utf-8")
    return ", ".join(outcomes)
