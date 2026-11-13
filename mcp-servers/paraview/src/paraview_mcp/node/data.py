"""Exact numbers from ParaView data: array summaries, field statistics, probing.

Everything here reads data through ParaView; nothing estimates from pixels or
from a colour bar. Imports of ParaView happen inside functions so the module
can be imported by the front's tests.
"""

from __future__ import annotations

import math
from typing import Any

ASSOCIATIONS = ("POINTS", "CELLS")


class DataError(Exception):
    """A request that names data the scene does not have."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _array_info(arrays: Any, association: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in arrays.keys():
        info = arrays[name]
        components = int(info.GetNumberOfComponents())
        low, high = info.GetRange(-1 if components > 1 else 0)
        out.append(
            {
                "name": name,
                "association": association,
                "components": components,
                "range": [float(low), float(high)],
            }
        )
    return out


def describe(proxy: Any) -> dict[str, Any]:
    """Summarise one pipeline object's current output."""
    info = proxy.GetDataInformation()
    bounds = list(info.GetBounds())
    arrays = _array_info(proxy.PointData, "POINTS") + _array_info(
        proxy.CellData, "CELLS"
    )
    return {
        "data_type": info.GetDataSetTypeAsString(),
        "points": int(info.GetNumberOfPoints()),
        "cells": int(info.GetNumberOfCells()),
        "bounds": [float(b) for b in bounds] if bounds[0] <= bounds[1] else None,
        "memory_kb": int(info.GetMemorySize()),
        "arrays": arrays,
    }


def find_association(proxy: Any, field: str) -> str:
    """Return POINTS or CELLS for ``field`` on ``proxy``, or raise DataError."""
    if field in proxy.PointData.keys():
        return "POINTS"
    if field in proxy.CellData.keys():
        return "CELLS"
    names = sorted(set(proxy.PointData.keys()) | set(proxy.CellData.keys()))
    raise DataError(
        "unknown_field",
        f"Field {field!r} is not on this object. Its fields are: "
        + (", ".join(names) or "none"),
    )


def color_component(
    proxy: Any, color_by: list[str] | None, component: Any
) -> int | None:
    """Check a colour component: None for the magnitude, else an index of the field."""
    if not color_by:
        raise DataError(
            "no_color_field",
            "Colour the object by a field before choosing a component.",
        )
    if str(component).lower() == "magnitude":
        return None
    arrays = proxy.PointData if color_by[0] == "POINTS" else proxy.CellData
    count = int(arrays[color_by[1]].GetNumberOfComponents())
    try:
        index = int(component)
    except (TypeError, ValueError):
        index = {"x": 0, "y": 1, "z": 2}.get(str(component).lower(), -1)
    if not 0 <= index < count:
        raise DataError(
            "bad_component",
            f"{color_by[1]} has {count} component(s); give Magnitude or 0 to {count - 1}.",
        )
    return index


def _datasets(proxy: Any) -> list[Any]:
    """Fetch ``proxy``'s output and return its leaf datasets."""
    from paraview import servermanager

    data = servermanager.Fetch(proxy)
    if data is None:
        return []
    if data.IsA("vtkCompositeDataSet"):
        leaves = []
        iterator = data.NewIterator()
        iterator.InitTraversal()
        while not iterator.IsDoneWithTraversal():
            leaves.append(iterator.GetCurrentDataObject())
            iterator.GoToNextItem()
        return leaves
    return [data]


def _values(proxy: Any, field: str, association: str, component: int | None) -> Any:
    import numpy
    from vtkmodules.util.numpy_support import vtk_to_numpy

    chunks = []
    for dataset in _datasets(proxy):
        attributes = (
            dataset.GetPointData() if association == "POINTS" else dataset.GetCellData()
        )
        array = attributes.GetArray(field) if attributes is not None else None
        if array is None:
            continue
        values = vtk_to_numpy(array).astype("float64")
        if values.ndim == 2:
            if component is None:
                values = numpy.linalg.norm(values, axis=1)
            else:
                if component >= values.shape[1]:
                    raise DataError(
                        "bad_component",
                        f"{field} has {values.shape[1]} components; "
                        f"component {component} does not exist.",
                    )
                values = values[:, component]
        chunks.append(values)
    if not chunks:
        return numpy.zeros(0)
    return numpy.concatenate(chunks)


def measure(
    proxy: Any, field: str, bins: int = 16, component: int | None = None
) -> dict[str, Any]:
    """Exact statistics and a histogram of ``field`` on ``proxy``."""
    import numpy

    association = find_association(proxy, field)
    values = _values(proxy, field, association, component)
    finite = values[numpy.isfinite(values)]
    if finite.size == 0:
        return {
            "field": field,
            "association": association,
            "component": "magnitude" if component is None else component,
            "count": int(values.size),
            "finite": 0,
            "message": "The field has no finite values on this object.",
        }
    counts, edges = numpy.histogram(finite, bins=max(1, min(int(bins), 256)))
    return {
        "field": field,
        "association": association,
        "component": "magnitude" if component is None else component,
        "count": int(values.size),
        "finite": int(finite.size),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "median": float(numpy.median(finite)),
        "histogram": {
            "edges": [float(e) for e in edges],
            "counts": [int(c) for c in counts],
        },
    }


def _round(value: float) -> float:
    if not math.isfinite(value):
        return value
    return float(f"{value:.6g}")


def probe(proxy: Any, position: list[float]) -> dict[str, Any] | None:
    """The point of ``proxy`` nearest ``position`` and every field value there."""
    from vtkmodules.vtkCommonDataModel import vtkPointLocator

    best: dict[str, Any] | None = None
    for block, dataset in enumerate(_datasets(proxy)):
        if dataset is None or dataset.GetNumberOfPoints() == 0:
            continue
        locator = vtkPointLocator()
        locator.SetDataSet(dataset)
        locator.BuildLocator()
        point_id = int(locator.FindClosestPoint(position))
        point = dataset.GetPoint(point_id)
        distance = math.dist(point, position)
        if best is not None and distance >= best["distance"]:
            continue
        values: dict[str, Any] = {}
        attributes = dataset.GetPointData()
        for index in range(attributes.GetNumberOfArrays()):
            array = attributes.GetArray(index)
            if array is None or array.GetName() is None:
                continue
            tuple_ = array.GetTuple(point_id)
            values[array.GetName()] = (
                _round(tuple_[0]) if len(tuple_) == 1 else [_round(v) for v in tuple_]
            )
        best = {
            "point_id": point_id,
            "block": block,
            "point": [_round(c) for c in point],
            "distance": distance,
            "point_values": values,
        }
    return best
