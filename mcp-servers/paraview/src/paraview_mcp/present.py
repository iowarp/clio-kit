"""Plain-text summaries of node half results, for the agent to read."""

from __future__ import annotations

from typing import Any


def _num(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _range(values: list[float] | None) -> str:
    if not values:
        return "?"
    return f"[{_num(values[0])}, {_num(values[1])}]"


def object_line(item: dict[str, Any]) -> str:
    display = item.get("display") or {}
    data = item.get("data") or {}
    parts = [f"{item['id']} {item.get('label') or item.get('type')}"]
    detail = [str(item.get("type") or item.get("kind"))]
    if item.get("input"):
        detail.append(f"from {item['input']}")
    if not display.get("visible", True):
        detail.append("hidden")
    else:
        detail.append(str(display.get("representation", "?")))
        color = display.get("color_by")
        if color:
            detail.append(f"coloured by {color[1]}")
    if data:
        detail.append(f"{data.get('points', 0)} points, {data.get('cells', 0)} cells")
    parts.append(f"({', '.join(detail)})")
    arrays = data.get("arrays") or []
    if arrays:
        parts.append(
            "fields: "
            + "; ".join(
                f"{a['name']} {a['association'].lower()}"
                + (f" {a['components']}-component" if a["components"] > 1 else "")
                + f" range {_range(a['range'])}"
                for a in arrays
            )
        )
    return " ".join(parts)


def time_line(time_state: dict[str, Any]) -> str:
    steps = int(time_state.get("steps") or 0)
    if steps <= 1:
        return "Time: one step." if steps == 1 else "Time: the data has no time steps."
    line = (
        f"Time: step {time_state['index'] + 1} of {steps} "
        f"(time {_num(time_state['time'])})"
    )
    if time_state.get("follow_latest"):
        return line + ", following the latest step."
    return (
        line
        + f", parked; {time_state.get('newer_steps', 0)} newer step(s), latest time "
        f"{_num(time_state.get('latest_time'))}."
    )


def view_line(view: dict[str, Any]) -> str:
    mode = view.get("mode", "local")
    size = view.get("geometry_mb")
    threshold = view.get("threshold_mb")
    how = (
        "drawn in the browser (local rendering)"
        if mode == "local"
        else "drawn by ParaView and sent as images (remote rendering)"
    )
    extra = ""
    if size is not None and threshold is not None:
        extra = f"; geometry {size:.2f} MB, remote render threshold {threshold:g} MB"
    if view.get("pinned"):
        extra += "; chosen by hand in the view"
    return f"View: {how}{extra}."


def scene_text(summary: dict[str, Any], where: str) -> str:
    """The agent-facing description of a scene."""
    lines = []
    restored = " (restored from its saved manifest)" if summary.get("restored") else ""
    dataset = summary.get("dataset") or {}
    source = dataset.get("path") or (
        f"the ParaView session at {dataset['session']}"
        if dataset.get("session")
        else ""
    )
    lines.append(f"Scene {summary['scene']} {where}{restored}: {source}.")
    if dataset.get("growing"):
        lines.append("The source is marked as still growing; new steps are picked up.")
    lines.append("Objects:")
    for item in summary.get("objects") or []:
        lines.append(f"- {object_line(item)}")
    lines.append(time_line(summary.get("time") or {}))
    lines.append(view_line(summary.get("view") or {}))
    selection = summary.get("selection")
    if selection:
        lines.append(selection_line(selection))
    lines.append(
        f"Pass scene={summary['scene']!r} to open_view to reopen exactly this scene."
    )
    return "\n".join(lines)


def selection_line(selection: dict[str, Any] | None) -> str:
    if not selection:
        return "Nothing is picked."
    values = ", ".join(
        f"{k}={_num(v) if not isinstance(v, list) else [_num(x) for x in v]}"
        for k, v in (selection.get("point_values") or {}).items()
    )
    position = ", ".join(_num(c) for c in selection.get("point") or [])
    return (
        f"Picked: point {selection.get('point_id')} of {selection.get('object')} "
        f"({selection.get('label')}) at ({position}); values: {values or 'none'}."
    )


def measure_text(result: dict[str, Any]) -> str:
    head = (
        f"{result['field']} ({result['association'].lower()}, "
        f"{result['component']}) on {result['object']} {result.get('label') or ''}"
    ).strip()
    if "min" not in result:
        return f"{head}: {result.get('message', 'no values')}"
    hist = result["histogram"]
    bins = ", ".join(
        f"[{_num(lo)}, {_num(hi)}): {count}"
        for lo, hi, count in zip(hist["edges"][:-1], hist["edges"][1:], hist["counts"])
    )
    return (
        f"{head}: {result['finite']} finite of {result['count']} values; "
        f"min {_num(result['min'])}, max {_num(result['max'])}, mean "
        f"{_num(result['mean'])}, median {_num(result['median'])}, std "
        f"{_num(result['std'])}.\nHistogram: {bins}."
    )


def camera_text(camera: dict[str, Any]) -> str:
    return (
        "Camera: position ("
        + ", ".join(_num(v) for v in camera["position"])
        + "), focal point ("
        + ", ".join(_num(v) for v in camera["focal_point"])
        + "), view up ("
        + ", ".join(_num(v) for v in camera["view_up"])
        + ")."
    )
