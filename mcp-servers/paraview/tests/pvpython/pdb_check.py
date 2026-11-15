"""A molecule from a PDB file takes a representation and a colour (JR-P13).

    pvpython pdb_check.py OUT_DIR

ParaView's PDB reader shows a molecule with no colour association; clearing a colour
that was never set used to fail with ``invalid association string 'NONE'``. Prints
``PDB=<json>`` with each call's outcome.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from paraview_mcp.node.engine import Engine

# Three atoms of a made-up peptide fragment, in fixed PDB columns.
ATOMS = """\
ATOM      1  N   THR A   1      17.047  14.099   3.625  1.00 13.79           N
ATOM      2  CA  THR A   1      16.967  12.784   4.338  1.00 10.80           C
ATOM      3  C   THR A   1      15.685  12.755   5.133  1.00  9.19           C
END
"""


def main(out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    pdb = out / "fragment.pdb"
    pdb.write_text(ATOMS)
    engine = Engine(out / "folder")
    engine.run("open_view", {"path": str(pdb), "new": True})
    seen: dict[str, Any] = {}
    for name, args in {
        "points": {"object": "o1", "representation": "Points"},
        "surface_opacity": {
            "object": "o1",
            "representation": "Surface",
            "opacity": 0.5,
        },
    }.items():
        try:
            result = engine.run("set_representation", args)
            seen[name] = result["object"]["display"]["representation"]
        except Exception as exc:  # noqa: BLE001 - the report names the failure
            seen[name] = f"error: {exc}"
    return seen


if __name__ == "__main__":
    print("PDB=" + json.dumps(main(Path(sys.argv[1]))))
