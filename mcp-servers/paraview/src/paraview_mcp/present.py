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
    if item.get("by"):
        detail.append(f"made by script {item['by']}")
    if not display.get("visible", True):
        detail.append("hidden")
    else:
        detail.append(str(display.get("representation", "?")))
        color = display.get("color_by")
        if color:
            component = display.get("color_component")
            which = "" if component is None else f" component {component}"
            colormap = f" ({display['colormap']})" if display.get("colormap") else ""
            detail.append(f"coloured by {color[1]}{which}{colormap}")
            if display.get("color_range"):
                low, high = display["color_range"]
                detail.append(f"colour range {_num(low)} to {_num(high)} set by hand")
            if display.get("scalar_bar"):
                detail.append("colour legend shown")
        elif display.get("solid_color"):
            detail.append("solid colour " + _hex(display["solid_color"]))
        if display.get("opacity", 1.0) < 1.0:
            detail.append(f"opacity {_num(float(display['opacity']))}")
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
    if mode == "remote" and view.get("reason") == "volume" and not view.get("pinned"):
        extra += "; a volume is shown, which only ParaView draws"
    if mode == "remote" and view.get("reason") == "slice" and not view.get("pinned"):
        extra += "; an image slice is shown, which only ParaView draws"
    if mode == "remote" and view.get("reason") == "growing" and not view.get("pinned"):
        extra += "; the source is growing, so each new step arrives as one image"
    if view.get("pinned"):
        extra += "; chosen by hand in the view"
    return f"View: {how}{extra}{frame_note(view)}."


#: A frame slower than this is reported, with what makes it slow when that is known.
SLOW_FRAME_SECONDS = 2.0


def frame_note(view: dict[str, Any]) -> str:
    """How long the last frame took, when slow; and why, when a volume's grid explains it."""

    seconds = view.get("frame_seconds")
    if seconds is None or seconds < SLOW_FRAME_SECONDS:
        return ""
    note = f"; its frames take up to {seconds:g} s each to draw"
    grids = [g for g in view.get("volume_grids") or [] if g != "vtkImageData"]
    if grids and view.get("mode") == "remote":
        note += (
            f", because ParaView draws a volume of a {grids[0]} far more slowly than one "
            "of an image grid: a ResampleToImage of it draws the same volume much faster"
        )
    return note


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
    if not source:
        lines[-1] = (
            f"Scene {summary['scene']} {where}{restored}: an empty scene; "
            "add sources with run_scene_script."
        )
    if dataset.get("growing"):
        lines.append("The source is marked as still growing; new steps are picked up.")
    if dataset.get("display_note"):
        lines.append(dataset["display_note"])
    lines.append("Objects:")
    for item in summary.get("objects") or []:
        lines.append(f"- {object_line(item)}")
    for step in summary.get("scripts") or []:
        made = ", ".join(step.get("made") or []) or "nothing kept"
        lines.append(
            f"Script {step['id']} {step.get('label')} ({step.get('lines')} lines) made {made}; "
            "it is replayed when the scene is restored."
        )
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


#: Lines of a series written out in text; the structured result has every step.
SERIES_LINES = 40


def series_text(result: dict[str, Any]) -> str:
    """A field's statistics at each time step."""
    series = result.get("series") or []
    head = (
        f"{result['field']} ({str(result.get('association') or '?').lower()}, "
        f"{result.get('component')}) on {result['object']} {result.get('label') or ''}"
    ).strip()
    if not series:
        return f"{head}: no step was measured."
    lines = [f"{head}, over {len(series)} of {result.get('steps', 0)} step(s):"]
    every = max(1, -(-len(series) // SERIES_LINES))
    shown = series[::every]
    for row in shown:
        where = f"step {row['index'] + 1}" + (
            f" (time {_num(row['time'])})" if row.get("time") is not None else ""
        )
        if "min" not in row:
            lines.append(f"- {where}: {row.get('message', 'no values')}")
            continue
        lines.append(
            f"- {where}: min {_num(row['min'])}, max {_num(row['max'])}, mean "
            f"{_num(row['mean'])}, std {_num(row['std'])}"
        )
    if every > 1:
        lines.append(
            f"Every {every}th step is written here; the structured result has all of them."
        )
    return "\n".join(lines)


def changes_lines(changes: list[dict[str, Any]]) -> list[str]:
    """The newest scene changes, and whether the person or the agent made them."""
    if not changes:
        return []
    return ["Recent changes, oldest first:"] + [
        f"- {c.get('at', '')} by the {c.get('by', 'agent')}: {c.get('text', '')}"
        for c in changes
    ]


def camera_text(camera: dict[str, Any]) -> str:
    projection = " Parallel projection." if camera.get("parallel_projection") else ""
    return (
        "Camera: position ("
        + ", ".join(_num(v) for v in camera["position"])
        + "), focal point ("
        + ", ".join(_num(v) for v in camera["focal_point"])
        + "), view up ("
        + ", ".join(_num(v) for v in camera["view_up"])
        + ")."
        + projection
    )


def capture_note(seconds: float | None, view: dict[str, Any]) -> str:
    """How long a capture took to draw, when slow; and why, as for the view."""

    if seconds is None or seconds < SLOW_FRAME_SECONDS:
        return ""
    cause = frame_note({**view, "frame_seconds": seconds}).partition(", because")[2]
    return f" It took {seconds:g} s to draw" + (f", because{cause}." if cause else ".")


def _hex(rgb: list[float]) -> str:
    return "#" + "".join(f"{round(float(v) * 255):02x}" for v in rgb)


def script_text(result: dict[str, Any], where: str) -> str:
    """What a scene script built, then the scene as it now stands."""
    step = result.get("step") or {}
    added = result.get("added") or []
    head = (
        f"Ran script {step.get('id')} {step.get('label')} ({step.get('lines')} lines) in the "
        f"scene: added {len(added)} object(s)."
    )
    lines = [head] + [f"+ {object_line(item)}" for item in added]
    if not added:
        lines.append("It added no objects (it may have changed existing ones).")
    lines.append(scene_text(result.get("scene") or {}, where))
    return "\n".join(lines)


def animation_text(result: dict[str, Any]) -> str:
    """What an exported animation holds, and where it is."""
    first, last = result.get("steps") or [0, 0]
    size = float(result.get("size_bytes") or 0) / 1e6
    return (
        f"Wrote {result.get('path')} on the ParaView machine: {result.get('format')} by "
        f"{result.get('encoder')}, {result.get('frames')} frames (time steps {first + 1} to "
        f"{last + 1}) at {_num(result.get('fps'))} per second, {_num(result.get('seconds'))} s, "
        f"{result.get('width')}x{result.get('height')}, {size:.1f} MB. Frames are rendered in "
        "the scene's own view, as the live view shows them."
    )
