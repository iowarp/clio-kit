"""What trame-vtk sends to the browser when the browser draws the scene.

ParaView 6 representations hand their mappers a vtkPartitionedDataSetCollection,
which trame-vtk 2.11 cannot serialize, so local rendering would show an empty
scene: trame-vtk's own merge-to-polydata serializer handles any composite
dataset. ParaView draws a colour legend through a widget representation whose
actor is a vtkScalarBarActor, which trame-vtk can send.

trame-vtk lays every legend out in one place, and the browser's size is not
known here, so the browser shows one legend: the selected object's when its
legend is on, else the newest one shown. Images ParaView renders (remote mode,
``capture``, ``export_artifact``) show every legend where ParaView puts it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import array_cache


def _shown_legends(renderer: Any) -> list[Any]:
    props = renderer.GetViewProps()
    found = []
    for i in range(props.GetNumberOfItems()):
        prop = props.GetItemAsObject(i)
        if (
            prop.GetClassName() == "vtkPVScalarBarRepresentation"
            and prop.GetVisibility()
        ):
            found.append(prop)
    return found


def chosen_legend(legends: list[Any], preferred: str | None) -> Any:
    """The legend the browser shows: the one titled ``preferred``, else the newest."""
    if not legends:
        return None
    if preferred:
        for legend in legends:
            if str(legend.GetScalarBarActor().GetTitle()).startswith(preferred):
                return legend
    return legends[-1]


def _new_map() -> Any:
    from vtkmodules.vtkRenderingCore import vtkColorTransferFunction

    return vtkColorTransferFunction()


class _LegendActor:
    """The legend's actor, with a colour map of its own for the browser.

    Sent as is, the legend shares the field's colour map object with the mappers;
    trame-vtk then sends that one object in two forms, and after a colour map change
    the browser lost both the legend and the scene. The legend gets a copy.
    """

    def __init__(self, actor: Any, copy: Any) -> None:
        self._actor = actor
        self._copy = copy

    def GetLookupTable(self) -> Any:
        lut = self._actor.GetLookupTable()
        if lut is None:
            return None
        self._copy.DeepCopy(lut)
        return self._copy

    def __getattr__(self, name: str) -> Any:
        return getattr(self._actor, name)


#: Per view: given a renderer, whether it is the view's own and, if so, the title
#: of the legend the view prefers. trame-vtk's registry is process-wide, and a node
#: half holds a render view per scene.
_CHOOSERS: list[Callable[[Any], tuple[bool, str | None]]] = []


def _preferred(renderer: Any) -> str | None:
    for chooser in _CHOOSERS:
        mine, title = chooser(renderer)
        if mine:
            return title
    return None


def register_serializers(chooser: Callable[[Any], tuple[bool, str | None]]) -> None:
    """Register the composite-dataset and legend serializers with trame-vtk, once,
    and ``chooser`` for one view's legend."""
    _CHOOSERS.append(chooser)
    if len(_CHOOSERS) > 1:
        return
    from trame_vtk.modules.vtk.serializers import initialize_serializers
    from trame_vtk.modules.vtk.serializers.actors import scalar_bar_actor_serializer
    from trame_vtk.modules.vtk.serializers.data import merge_to_polydata_serializer
    from trame_vtk.modules.vtk.serializers.registry import (
        register_instance_serializer,
    )

    initialize_serializers()
    array_cache.install()
    for name in ("vtkPartitionedDataSetCollection", "vtkPartitionedDataSet"):
        register_instance_serializer(name, merge_to_polydata_serializer)

    copies: dict[int, Any] = {}

    def legend(parent: Any, rep: Any, rep_id: Any, context: Any, depth: int) -> Any:
        if chosen_legend(_shown_legends(parent), _preferred(parent)) is not rep:
            return None
        # A fresh copy each time: trame-vtk sends it whole, as a new object.
        copies[id(rep)] = _new_map()
        actor = _LegendActor(rep.GetScalarBarActor(), copies[id(rep)])
        return scalar_bar_actor_serializer(parent, actor, rep_id, context, depth)

    register_instance_serializer("vtkPVScalarBarRepresentation", legend)
