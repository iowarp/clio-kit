"""ParaView calls that changed between versions, made to work as scripts write them.

Scene scripts are written from what an agent knows of ParaView, often an older
release. Two calls that ParaView 6 broke are mended in this process, once, so a
script that uses them works the first time:

- ``lut.ApplyPreset("Viridis (matplotlib)")``: ParaView 6 dropped the
  " (matplotlib)" qualifier from several presets and raises on the old name.
  The name is resolved as the view descriptor's colour maps are.
- ``ColorBy(display, None)`` on a display that colours nothing (a slice of an
  STL, a PDB's molecule): its association is ``NONE``, which ParaView cannot
  parse, so clearing colours that were never set raises. There is nothing to
  clear, so it does nothing.
"""

from __future__ import annotations

from typing import Any

_installed = False


def install() -> None:
    global _installed
    if _installed:
        return
    from paraview import servermanager

    from .sources import preset_name

    apply_preset = servermanager.TransferFunctionProxy.ApplyPreset

    def ApplyPreset(self: Any, presetName: str, rescale: bool = True) -> Any:
        name = preset_name(str(presetName)) if isinstance(presetName, str) else None
        return apply_preset(self, name or presetName, rescale)

    color_by = servermanager.RepresentationProxy.ColorBy

    def ColorBy(self: Any, value: Any = None, separate: bool = False) -> Any:
        if value is None and self.ColorArrayName.GetAssociation() in (None, "NONE"):
            return None
        return color_by(self, value, separate)

    ApplyPreset.__doc__ = apply_preset.__doc__
    ColorBy.__doc__ = color_by.__doc__
    servermanager.TransferFunctionProxy.ApplyPreset = ApplyPreset
    servermanager.RepresentationProxy.ColorBy = ColorBy
    _installed = True
