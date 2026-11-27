"""Lean words for the agent."""

from __future__ import annotations

from typing import Any

from . import install


def status_text(d: dict[str, Any]) -> str:
    av, disp = d["avogadro"], d["display"]
    lines = []
    if not d["platform"]["supported"]:
        return d["platform"]["why"]
    if av["found"]:
        lines.append(f"Avogadro: {av['command']} (via {av['source']}).")
    else:
        lines.append(install.missing_text(av))
    if disp["missing"]:
        lines.append(
            f"Missing for the streamed view: {', '.join(disp['missing'])}. {disp['install']}"
        )
    else:
        lines.append("The virtual display and VNC server are available.")
    lines.append("Running now." if d["running"] else "Not running.")
    return " ".join(lines)


def open_text(d: dict[str, Any]) -> str:
    what = f"opened {d['opened']}" if d.get("opened") else "is open with an empty molecule"
    return (
        f"Avogadro {what} in the view. The person builds, edits and selects atoms there; "
        "when they say they are done or ask about it, read_avogadro reads the molecule open "
        "in Avogadro, with their selection."
    )


def read_text(d: dict[str, Any]) -> str:
    sel = d["selected"]
    sel_text = (
        "selected atoms (0-based): "
        + ", ".join(f"{i} {e}" for i, e in zip(sel, d["selected_elements"], strict=True))
        if sel
        else "no atoms selected"
    )
    return (
        (
            f"Avogadro has {d['formula']} ({d['n_atoms']} atoms) open; {sel_text}. "
            if d.get("source", "live") == "live"
            else f"The person sent {d['formula']} ({d['n_atoms']} atoms) from Avogadro; "
            f"{sel_text}. "
        )
        + f"Structure: {d['structure']['path']} (XYZ); Avogadro's CJSON: {d['cjson']}."
        + (" It has a unit cell (kept in the CJSON)." if d.get("unit_cell") else "")
    )
