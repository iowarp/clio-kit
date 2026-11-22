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


def _remote() -> bool:
    from paraview import servermanager

    connection = servermanager.ActiveConnection
    return connection is not None and bool(connection.IsRemote())


def _view_time() -> float | None:
    """The active view's time, which ``simple.UpdatePipeline`` would use too."""
    from paraview import simple

    view = simple.GetActiveView()
    value = getattr(view, "ViewTime", None) if view is not None else None
    return None if value is None else float(value)


def _leaves(data: Any) -> list[Any]:
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


def _datasets(proxy: Any, time: float | None = None) -> list[Any]:
    """``proxy``'s output at ``time`` (the view's, by default) as its leaf datasets, in
    a built-in session.

    The pipeline's own output is read directly: ``Fetch`` passes data through a reduction
    that, for a rectilinear grid, returns the right number of points with half their
    values zero (ParaView 6.2). A remote session never brings whole datasets here; it
    reduces them on the pvserver (``_gather``).
    """
    from paraview import simple

    simple.UpdatePipeline(time=time, proxy=proxy)
    return _leaves(proxy.GetClientSideObject().GetOutputDataObject(0))


def _fetch(proxy: Any, time: float | None, helper: str) -> Any:
    """``servermanager.Fetch`` at an explicit time: gather every rank's output of
    ``proxy`` with ``helper`` and move the result here."""
    from paraview import servermanager

    # Like Fetch, the helper proxies are never registered and go with these names.
    filters = servermanager.filters
    reducer = filters.ReductionFilter(Input=servermanager.OutputPort(proxy, 0))
    reducer.PostGatherHelperName = helper
    if time is None:
        reducer.UpdatePipeline()
    else:
        reducer.UpdatePipeline(time)
    info = reducer.GetDataInformation(0)
    kind = info.GetDataSetType()
    if info.GetCompositeDataSetType() > 0:
        kind = info.GetCompositeDataSetType()
    fetcher = filters.ClientServerMoveData(Input=reducer)
    fetcher.OutputDataType = kind
    fetcher.WholeExtent = info.GetExtent()[:]
    if time is None:
        fetcher.UpdatePipeline()
    else:
        fetcher.UpdatePipeline(time)
    output = fetcher.GetClientSideObject().GetOutputDataObject(0)
    if output is None:
        return None
    copy = output.NewInstance()
    copy.ShallowCopy(output)
    copy.UnRegister(None)
    return copy


# Runs on every pvserver rank inside a ProgrammableFilter, so a remote measure or probe
# moves a point or two per rank over the link instead of every array of every cell.
_RANK_SCRIPT = r"""
import numpy
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray

source = self.GetInputDataObject(0, 0)
leaves = []
if source is not None and source.IsA("vtkCompositeDataSet"):
    walk = source.NewIterator()
    walk.InitTraversal()
    while not walk.IsDoneWithTraversal():
        leaves.append(walk.GetCurrentDataObject())
        walk.GoToNextItem()
elif source is not None:
    leaves.append(source)
leaves = [d for d in leaves if d is not None and d.IsA("vtkDataSet")]
out = self.GetOutputDataObject(0)
points = vtkPoints()
columns = {}


def column(name, width=1):
    if name not in columns:
        array = vtkDoubleArray()
        array.SetName(name)
        array.SetNumberOfComponents(width)
        columns[name] = array
    return columns[name]


if MODE == "stats":
    from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline

    # A single structured dataset split over the ranks shares each boundary plane of
    # points between two pieces: the piece below owns it, unless it is the last.
    whole = None
    key = vtkStreamingDemandDrivenPipeline.WHOLE_EXTENT()
    info = self.GetInputInformation(0, 0)
    if (ASSOC == "POINTS" and source is not None and not source.IsA("vtkCompositeDataSet")
            and info is not None and info.Has(key)):
        whole = [int(v) for v in info.Get(key)]

    def owned(d, size):
        if whole is None or not hasattr(d, "GetExtent"):
            return None
        extent = list(d.GetExtent())
        axes = []
        for axis in range(3):
            low, high = extent[2 * axis], extent[2 * axis + 1]
            keep = numpy.ones(high - low + 1, dtype=bool)
            if high < whole[2 * axis + 1] and keep.size > 1:
                keep[-1] = False
            axes.append(keep)
        mask = (axes[2][:, None, None] & axes[1][None, :, None]
                & axes[0][None, None, :]).reshape(-1)
        return mask if mask.size == size else None

    chunks = []
    for d in leaves:
        attributes = d.GetPointData() if ASSOC == "POINTS" else d.GetCellData()
        array = attributes.GetArray(FIELD)
        if array is None:
            continue
        values = vtk_to_numpy(array).astype("float64")
        if values.ndim == 2:
            values = (numpy.linalg.norm(values, axis=1) if COMPONENT is None
                      else values[:, COMPONENT])
        keep = owned(d, values.shape[0])
        ghosts = attributes.GetArray("vtkGhostType")
        if ghosts is not None:
            unghosted = (vtk_to_numpy(ghosts) & 1) == 0
            keep = unghosted if keep is None else keep & unghosted
        if keep is not None:
            values = values[keep]
        chunks.append(values)
    values = numpy.concatenate(chunks) if chunks else numpy.zeros(0)
    finite = values[numpy.isfinite(values)]
    nan = float("nan")
    mean = float(finite.mean()) if finite.size else nan
    row = {
        "count": float(values.size),
        "finite": float(finite.size),
        "mean": mean,
        "m2": float(((finite - mean) ** 2).sum()) if finite.size else 0.0,
        "min": float(finite.min()) if finite.size else nan,
        "max": float(finite.max()) if finite.size else nan,
        "median": float(numpy.median(finite)) if finite.size else nan,
    }
    for name, value in row.items():
        column(name).InsertNextValue(value)
    counts = (numpy.histogram(finite, bins=BINS,
                              range=None if LOW is None else (LOW, HIGH))[0]
              if finite.size else numpy.zeros(BINS))
    column("histogram", BINS).InsertNextTuple([float(c) for c in counts])
    points.InsertNextPoint(0.0, 0.0, 0.0)
else:
    from vtkmodules.vtkCommonDataModel import vtkPointLocator

    best = None
    for block, d in enumerate(leaves):
        if d.GetNumberOfPoints() == 0:
            continue
        locator = vtkPointLocator()
        locator.SetDataSet(d)
        locator.BuildLocator()
        point_id = int(locator.FindClosestPoint(POSITION))
        point = d.GetPoint(point_id)
        distance = sum((a - b) ** 2 for a, b in zip(point, POSITION)) ** 0.5
        if best is None or distance < best[0]:
            best = (distance, block, point_id, point, d)
    if best is not None:
        distance, block, point_id, point, d = best
        points.InsertNextPoint(point)
        column("clio_distance").InsertNextValue(distance)
        column("clio_block").InsertNextValue(float(block))
        column("clio_point_id").InsertNextValue(float(point_id))
        attributes = d.GetPointData()
        for index in range(attributes.GetNumberOfArrays()):
            array = attributes.GetArray(index)
            if array is None or not array.GetName() or array.GetName() in columns:
                continue
            column(array.GetName(), array.GetNumberOfComponents()).InsertNextTuple(
                array.GetTuple(point_id))

out.SetPoints(points)
vertices = vtkCellArray()
for index in range(points.GetNumberOfPoints()):
    vertices.InsertNextCell(1)
    vertices.InsertCellPoint(index)
out.SetVerts(vertices)
for array in columns.values():
    out.GetPointData().AddArray(array)
"""


def _gather(
    proxy: Any, mode: str, time: float | None = None, **params: Any
) -> list[dict[str, Any]]:
    """Run ``_RANK_SCRIPT`` on every pvserver rank at ``time`` (the view's, by default)
    and bring back one row per rank: a small table, whatever the dataset's size."""
    from paraview import servermanager

    if time is None:
        time = _view_time()
    header = f"MODE = {mode!r}\n" + "".join(
        f"{name} = {value!r}\n" for name, value in params.items()
    )
    rank = servermanager.filters.ProgrammableFilter(
        Input=servermanager.OutputPort(proxy, 0)
    )
    rank.OutputDataSetType = "vtkPolyData"
    rank.Script = header + _RANK_SCRIPT
    if time is None:
        rank.UpdatePipeline()
    else:
        rank.UpdatePipeline(time)
    table = _fetch(rank, time, "vtkAppendPolyData")
    if table is None or not table.IsA("vtkPolyData"):
        raise DataError(
            "server_reduce",
            "The ParaView server did not answer a measurement. Its programmable "
            "filters need ParaView's Python on the server.",
        )
    attributes = table.GetPointData()
    rows: list[dict[str, Any]] = []
    for index in range(table.GetNumberOfPoints()):
        row: dict[str, Any] = {"point": list(table.GetPoint(index))}
        for column in range(attributes.GetNumberOfArrays()):
            array = attributes.GetArray(column)
            values = array.GetTuple(index)
            row[array.GetName()] = values[0] if len(values) == 1 else list(values)
        rows.append(row)
    if mode == "stats" and not rows:
        raise DataError(
            "server_reduce",
            "The ParaView server measured nothing. Its programmable filters need "
            "ParaView's Python on the server.",
        )
    return rows


def _values(
    proxy: Any,
    field: str,
    association: str,
    component: int | None,
    time: float | None = None,
) -> Any:
    import numpy
    from vtkmodules.util.numpy_support import vtk_to_numpy

    chunks = []
    for dataset in _datasets(proxy, time):
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


def _empty(
    field: str, association: str, component: int | None, count: int
) -> dict[str, Any]:
    return {
        "field": field,
        "association": association,
        "component": "magnitude" if component is None else component,
        "count": count,
        "finite": 0,
        "message": "The field has no finite values on this object.",
    }


def combine(
    rows: list[dict[str, Any]], bins: int, histograms: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Join the pvserver ranks' partial statistics into the whole field's.

    The mean and spread join exactly (Chan's pairwise update). The median is exact when
    one rank holds every finite value; otherwise the bin that holds it is given. The
    histogram's counts come from ``histograms`` when the ranks binned on shared edges,
    else from the one rank that had values.
    """
    count = int(sum(r["count"] for r in rows))
    held = [r for r in rows if r["finite"] > 0]
    finite = int(sum(r["finite"] for r in held))
    out: dict[str, Any] = {"count": count, "finite": finite}
    if not finite:
        return out
    mean = sum(r["finite"] * r["mean"] for r in held) / finite
    m2 = sum(r["m2"] + r["finite"] * (r["mean"] - mean) ** 2 for r in held)
    low = min(r["min"] for r in held)
    high = max(r["max"] for r in held)
    out.update(
        min=float(low),
        max=float(high),
        mean=float(mean),
        std=float(math.sqrt(max(m2, 0.0) / finite)),
    )
    if len(held) > 1 and histograms is None:
        # The ranks binned on their own ranges: their counts do not add up.
        out["median"] = None
        out["median_note"] = (
            f"The pvserver's {len(held)} ranks each hold part of the field, so the "
            "median is not measured here; measure_field gives the bin that holds it."
        )
        return out
    counts = [0.0] * bins
    for row in histograms if histograms is not None else held:
        part = row["histogram"]
        for index, value in enumerate(part if isinstance(part, list) else [part]):
            counts[index] += float(value)
    # numpy.histogram's edges: an empty range widens by a half on each side.
    first, last = (low - 0.5, high + 0.5) if low == high else (low, high)
    edges = [first + (last - first) * i / bins for i in range(bins)] + [float(last)]
    if len(held) == 1:
        out["median"] = float(held[0]["median"])
    else:
        total, at = 0.0, bins - 1
        for index, value in enumerate(counts):
            total += value
            if total >= finite / 2.0:
                at = index
                break
        out["median"] = None
        out["median_between"] = [float(edges[at]), float(edges[at + 1])]
        out["median_note"] = (
            f"The pvserver's {len(held)} ranks each hold part of the field, so the "
            "median is given as the histogram bin that holds it."
        )
    out["histogram"] = {
        "edges": [float(e) for e in edges],
        "counts": [int(round(c)) for c in counts],
    }
    return out


def _measure_remote(
    proxy: Any,
    field: str,
    association: str,
    bins: int,
    component: int | None,
    time: float | None,
    histogram: bool = True,
) -> dict[str, Any]:
    arrays = proxy.PointData if association == "POINTS" else proxy.CellData
    width = int(arrays[field].GetNumberOfComponents())
    if component is not None and width > 1 and component >= width:
        raise DataError(
            "bad_component",
            f"{field} has {width} components; component {component} does not exist.",
        )
    params = {
        "FIELD": field,
        "ASSOC": association,
        "COMPONENT": component if width > 1 else None,
        "BINS": bins,
        "LOW": None,
        "HIGH": None,
    }
    rows = _gather(proxy, "stats", time, **params)
    held = [r for r in rows if r["finite"] > 0]
    histograms = None
    if len(held) > 1 and histogram:
        # Bin every rank on the same edges, from the whole field's range.
        low = min(r["min"] for r in held)
        high = max(r["max"] for r in held)
        histograms = _gather(
            proxy, "stats", time, **{**params, "LOW": low, "HIGH": high}
        )
    joined = combine(rows, bins, histograms)
    if not joined["finite"]:
        return _empty(field, association, component, joined["count"])
    return {
        "field": field,
        "association": association,
        "component": "magnitude" if component is None else component,
        **joined,
    }


def measure(
    proxy: Any,
    field: str,
    bins: int = 16,
    component: int | None = None,
    time: float | None = None,
    histogram: bool = True,
) -> dict[str, Any]:
    """Exact statistics and a histogram of ``field`` on ``proxy`` at ``time`` (the
    view's, by default). The view itself is not moved. Without ``histogram``, a
    pvserver whose ranks split the field skips the pass that bins them alike."""
    import numpy

    association = find_association(proxy, field)
    bins = max(1, min(int(bins), 256))
    if _remote():
        return _measure_remote(
            proxy, field, association, bins, component, time, histogram
        )
    values = _values(proxy, field, association, component, time)
    finite = values[numpy.isfinite(values)]
    if finite.size == 0:
        return _empty(field, association, component, int(values.size))
    counts, edges = numpy.histogram(finite, bins=bins)
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

    if _remote():
        rows = _gather(proxy, "probe", POSITION=[float(c) for c in position])
        if not rows:
            return None
        near = min(rows, key=lambda r: r["clio_distance"])
        return {
            "point_id": int(near["clio_point_id"]),
            "block": int(near["clio_block"]),
            "point": [_round(c) for c in near["point"]],
            "distance": float(near["clio_distance"]),
            "point_values": {
                name: [_round(v) for v in value]
                if isinstance(value, list)
                else _round(value)
                for name, value in near.items()
                if name != "point" and not name.startswith("clio_")
            },
        }
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
