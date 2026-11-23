"""The default colour range: one that represents the field rather than its extremes.

ParaView maps a field's colours over its whole data range, so a few extreme values
(a thin boundary layer, a spike, a body's surface) take the top of the colour map and
leave the rest of the field in a narrow band of colour. The default here is the range
from the 2nd to the 98th percentile of the values shown (the convention of xarray's
``robust=True``), measured on the objects coloured by the field at the view's time.
It grows as steps arrive, as ParaView's own maps grow, from its own record rather
than ParaView's full-range rescale. A range set by hand, or a colour map locked by a
script, stays where it is; ``rescale`` asks for the data's full range instead.
"""

from __future__ import annotations

import json
import math
from typing import Any

from . import colors, data, manifest
from .held import LEDGER

PERCENTILES = (2.0, 98.0)
#: Evenly spaced order statistics each pvserver rank sends back.
SAMPLES = 1001
#: Values a pvserver rank reads at most for the range, evenly strided beyond that.
SAMPLE = 2_000_000


def percentile(ordered: list[float], percent: float) -> float:
    """numpy's linear percentile of sorted values."""
    at = percent / 100.0 * (len(ordered) - 1)
    low = math.floor(at)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (at - low)


def _share(quantiles: list[float], value: float) -> float:
    """The fraction of a rank's values at or below ``value``, from its order statistics."""
    last = len(quantiles) - 1
    if value < quantiles[0]:
        return 0.0
    if value >= quantiles[-1] or last == 0:
        return 1.0
    low, high = 0, last
    while high - low > 1:
        middle = (low + high) // 2
        if quantiles[middle] <= value:
            low = middle
        else:
            high = middle
    step = quantiles[high] - quantiles[low]
    part = (value - quantiles[low]) / step if step > 0 else 1.0
    return (low + part) / last


def pooled(rows: list[dict[str, Any]], percent: float) -> float | None:
    """A percentile of the whole field from each rank's order statistics, weighted by
    the values each rank holds. One rank holding the field gives numpy's answer at
    the sample points (the 2nd and 98th percentiles are among them)."""
    held = [
        (float(r["finite"]) * float(r.get("stride") or 1), list(r["quantiles"]))
        for r in rows
        if r.get("finite") and isinstance(r.get("quantiles"), list)
    ]
    if not held:
        return None
    if len(held) == 1:
        return percentile(held[0][1], percent)
    total = sum(weight for weight, _ in held)
    target = percent / 100.0
    low = min(q[0] for _, q in held)
    high = max(q[-1] for _, q in held)
    for _ in range(200):
        middle = (low + high) / 2.0
        if middle in (low, high):
            break
        below = sum(weight * _share(q, middle) for weight, q in held) / total
        if below < target:
            low = middle
        else:
            high = middle
    return high


def _width(proxy: Any, association: str, field: str) -> int:
    arrays = proxy.PointData if association == "POINTS" else proxy.CellData
    return int(arrays[field].GetNumberOfComponents())


def _whole(proxy: Any, association: str, field: str, component: int | None) -> Any:
    arrays = proxy.PointData if association == "POINTS" else proxy.CellData
    width = int(arrays[field].GetNumberOfComponents())
    which = 0 if width == 1 else (-1 if component is None else component)
    return [float(v) for v in arrays[field].GetRange(which)]


def span(
    proxy: Any, field: str, component: int | None, time: float | None
) -> dict[str, list[float]] | None:
    """The 2nd to 98th percentile and the whole range of ``field`` on ``proxy`` at
    ``time``, or None when it has no finite values."""
    association = data.find_association(proxy, field)
    if component is not None and _width(proxy, association, field) == 1:
        component = None
    if data._remote():
        rows = data._gather(
            proxy,
            "stats",
            time,
            FIELD=field,
            ASSOC=association,
            COMPONENT=component,
            BINS=1,
            LOW=None,
            HIGH=None,
            QUANTILES=SAMPLES,
            SAMPLE=SAMPLE,
        )
        low, high = (pooled(rows, p) for p in PERCENTILES)
        if low is None or high is None:
            return None
        whole = _whole(proxy, association, field, component)
        return {"span": [low, high], "whole": whole}
    import numpy

    values = data._values(proxy, field, association, component, time)
    finite = values[numpy.isfinite(values)]
    if finite.size == 0:
        return None
    low, high = (float(v) for v in numpy.percentile(finite, PERCENTILES))
    return {"span": [low, high], "whole": [float(finite.min()), float(finite.max())]}


def _lut_span(lut: Any) -> tuple[float, float] | None:
    points = list(lut.RGBPoints)
    return (float(points[0]), float(points[-4])) if len(points) >= 8 else None


def span_all(
    proxy: Any, wanted: list[tuple[str, int | None]], time: float | None
) -> list[dict[str, list[float]] | None]:
    """``span`` of each (field, component) on ``proxy`` at ``time``; on a pvserver,
    in one pass over its ranks. A field not on ``proxy`` measures None."""
    if not data._remote():
        found: list[dict[str, list[float]] | None] = []
        for field, component in wanted:
            try:
                found.append(span(proxy, field, component, time))
            except data.DataError:
                found.append(None)
        return found
    asked: list[tuple[str, str, int | None]] = []
    where: list[int | None] = []
    for field, component in wanted:
        try:
            association = data.find_association(proxy, field)
        except data.DataError:
            where.append(None)
            continue
        if component is not None and _width(proxy, association, field) == 1:
            component = None
        where.append(len(asked))
        asked.append((field, association, component))
    rows = (
        data._gather(
            proxy,
            "stats",
            time,
            FIELDS=asked,
            BINS=1,
            LOW=None,
            HIGH=None,
            QUANTILES=SAMPLES,
            SAMPLE=SAMPLE,
        )
        if asked
        else []
    )
    measured: list[dict[str, list[float]] | None] = []
    for index, (field, association, component) in enumerate(asked):
        mine = [r for r in rows if int(r.get("which") or 0) == index]
        low, high = (pooled(mine, p) for p in PERCENTILES)
        if low is None or high is None:
            measured.append(None)
            continue
        whole = _whole(proxy, association, field, component)
        measured.append({"span": [low, high], "whole": whole})
    return [None if at is None else measured[at] for at in where]


def signature(engine: Any, item: dict[str, Any]) -> tuple[str, str, int] | None:
    """What ``item`` shows, for measuring it once for every view that shows the same:
    its shared reader, the filters from there to it, and the reader's generation.
    None for what a script made or anything not on a shared reader."""
    scene = engine.scene or {}
    readers = getattr(engine, "readers", None)
    if readers is None:
        return None
    chain: list[Any] = []
    current: dict[str, Any] | None = item
    while current is not None:
        kind = current.get("kind")
        if kind == "reader":
            proxy = engine.proxies.get(current["id"])
            source = readers.of(proxy) if proxy is not None else None
            if source is None:
                return None
            text = json.dumps(chain[::-1], sort_keys=True, default=str)
            return source.key, text, source.generation
        if kind != "filter":
            return None
        chain.append([current.get("type"), current.get("properties") or {}])
        current = manifest.find_object(scene, current.get("input") or "")
    return None


def _siblings(engine: Any, sign: tuple[str, str, int]) -> list[tuple[str, int | None]]:
    """The fields the views of one shared reader colour on what ``sign`` shows."""
    source = engine.readers.named(sign[0])
    users = list(source.users.values()) if source is not None else [engine]
    wanted: list[tuple[str, int | None]] = []
    for user in users:
        for item in (user.scene or {}).get("objects") or []:
            shown = item.get("display") or {}
            field = (shown.get("color_by") or [None, None])[1]
            if not field or shown.get("visible") is False or shown.get("color_range"):
                continue
            if signature(user, item) != sign:
                continue
            pair = (field, shown.get("color_component"))
            if pair not in wanted:
                wanted.append(pair)
    return wanted


def measure(
    engine: Any,
    item: dict[str, Any],
    proxy: Any,
    field: str,
    component: int | None,
    time: float | None,
) -> dict[str, list[float]] | None:
    """``span`` of ``field`` on ``item``, measured once for every view of a shared
    reader showing the same at that time: one pass measures each field they colour,
    and the others take theirs from it."""
    sign = signature(engine, item)
    if sign is None:
        return span(proxy, field, component, time)
    readers = engine.readers
    hit, value = readers.range_of((*sign, time, field, component))
    if hit:
        return value
    wanted = [(field, component)]
    for pair in _siblings(engine, sign):
        if pair not in wanted and not readers.range_of((*sign, time, *pair))[0]:
            wanted.append(pair)
    found = span_all(proxy, wanted, time)
    for pair, measured in zip(wanted, found, strict=True):
        readers.keep_range((*sign, time, *pair), measured)
    return found[0]


def locked(lut: Any) -> bool:
    return (
        "AutomaticRescaleRangeMode" in lut.ListProperties()
        and str(lut.AutomaticRescaleRangeMode) == "Never"
    )


def _set(simple: Any, field: str, low: float, high: float) -> None:
    simple.GetColorTransferFunction(field).RescaleTransferFunction(low, high)
    simple.GetOpacityTransferFunction(field).RescaleTransferFunction(low, high)


def _shown(
    engine: Any, field: str, extra: tuple[Any, Any] | None = None
) -> list[tuple[Any, Any]]:
    """The visible objects coloured by ``field``, with their pipeline objects."""
    found = []
    items = list((engine.scene or {}).get("objects") or [])
    if extra is not None and all(item is not extra[0] for item in items):
        items.append(extra[0])
    for item in items:
        shown = item.get("display") or {}
        proxy = engine.proxies.get(item["id"])
        if extra is not None and item is extra[0]:
            proxy = extra[1]
        if proxy is None or shown.get("visible") is False:
            continue
        if (shown.get("color_by") or [None, None])[1] == field:
            found.append((item, proxy))
    return found


def fit(engine: Any, item: dict[str, Any], proxy: Any, rep: Any) -> None:
    """Give ``item``'s field its default colour range, unless it has one by hand."""
    display = item.get("display") or {}
    color_by = display.get("color_by")
    field = (color_by or [None, None])[1]
    if not field:
        return
    simple = engine.simple
    lut = simple.GetColorTransferFunction(field)
    shown = _shown(engine, field, (item, proxy))
    displays = [i.get("display") or {} for i, _ in shown] or [display]
    if any(d.get("color_range") for d in displays) or locked(lut):
        return
    if any(d.get("range_mode") == "data" for d in displays):
        rep.RescaleTransferFunctionToDataRange(True, False)
        return
    time = getattr(engine.view, "ViewTime", None)
    found = []
    for other, other_proxy in shown:
        try:
            measured = measure(
                engine,
                other,
                other_proxy,
                field,
                (other.get("display") or {}).get("color_component"),
                None if time is None else float(time),
            )
        except Exception:  # noqa: BLE001 - no measurement means ParaView's own range
            measured = None
        if measured is not None:
            found.append(measured)
    if not found:
        rep.RescaleTransferFunctionToDataRange(True, False)
        return
    low = min(f["span"][0] for f in found)
    high = max(f["span"][1] for f in found)
    whole = [min(f["whole"][0] for f in found), max(f["whole"][1] for f in found)]
    if not low < high:
        low, high = whole
    before = next((d["auto_range"] for d in displays if d.get("auto_range")), None)
    if before:
        low, high = min(low, before[0]), max(high, before[1])
        seen = next((d["data_range"] for d in displays if d.get("data_range")), None)
        if seen:
            whole = [min(whole[0], seen[0]), max(whole[1], seen[1])]
    if not low < high:
        rep.RescaleTransferFunctionToDataRange(True, False)
        return
    _set(simple, field, low, high)
    for shown_display in displays:
        shown_display.update(auto_range=[low, high], data_range=whole)


def grow(engine: Any, scene: dict[str, Any]) -> None:
    """Each field's default range takes in a new step's values."""
    with LEDGER.phase("update"):
        engine.view.Update()
    done: set[str] = set()
    for item in scene["objects"]:
        shown = item.get("display") or {}
        field = (shown.get("color_by") or [None, None])[1]
        proxy = engine.proxies.get(item["id"])
        if proxy is None or not field or field in done or shown.get("color_range"):
            continue
        if shown.get("visible") is False:
            continue
        done.add(field)
        with LEDGER.phase("range"):
            rep = engine.simple.GetDisplayProperties(proxy, engine.view)
            fit(engine, item, proxy, rep)


def full(engine: Any, item: dict[str, Any], proxy: Any) -> None:
    """``rescale``: the field's whole data range, which then grows with the data."""
    scene = engine.require_scene()
    color_by = (item.get("display") or {}).get("color_by")
    if not color_by:
        return
    rep = engine.simple.GetDisplayProperties(proxy, engine.view)
    lut = engine.simple.GetColorTransferFunction(color_by[1])
    if locked(lut):
        lut.AutomaticRescaleRangeMode = "Grow and update on 'Apply'"
    rep.RescaleTransferFunctionToDataRange(False, True)
    colors.share(scene, color_by, "color_range", None)
    colors.share(scene, color_by, "auto_range", None)
    colors.share(scene, color_by, "range_mode", "data")


def spans(simple: Any, scene: dict[str, Any]) -> dict[str, Any]:
    """Each coloured field's colour range and lock, before a script runs."""
    return {
        name: (_lut_span(lut), locked(lut))
        for name in colors.fields(scene)
        for lut in [simple.GetColorTransferFunction(name)]
    }


def _close(a: tuple[float, float], b: list[float] | tuple[float, float]) -> bool:
    scale = max(abs(b[1] - b[0]), abs(b[0]), abs(b[1]), 1e-300)
    return abs(a[0] - b[0]) <= 1e-6 * scale and abs(a[1] - b[1]) <= 1e-6 * scale


def after_script(engine: Any, scene: dict[str, Any], before: dict[str, Any]) -> None:
    """A colour range a script set is kept as set by hand; one ParaView chose (the
    data's range, as ``ColorBy`` gives a new colour map) becomes the default range."""
    simple = engine.simple
    for name in colors.fields(scene):
        lut = simple.GetColorTransferFunction(name)
        now, lock = _lut_span(lut), locked(lut)
        if now is None or before.get(name) == (now, lock):
            continue
        shown = _shown(engine, name)
        if not shown:
            continue
        item, proxy = shown[0]
        color_by = (item.get("display") or {}).get("color_by")
        wholes = []
        for other, other_proxy in shown:
            component = (other.get("display") or {}).get("color_component")
            association = data.find_association(other_proxy, name)
            wholes.append(_whole(other_proxy, association, name, component))
        whole = (min(w[0] for w in wholes), max(w[1] for w in wholes))
        if lock or not _close(now, whole):
            colors.set_range(simple, scene, color_by, now)
            continue
        colors.share(scene, color_by, "auto_range", None)
        colors.share(scene, color_by, "range_mode", None)
        fit(engine, item, proxy, simple.GetDisplayProperties(proxy, engine.view))
