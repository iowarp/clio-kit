"""What changed in the scene, and who changed it, in words the agent reads.

Every command that changes the scene goes through ``Engine.run``. When it
commits, the engine records one line here: who asked (the person in the view,
or the agent through a tool), what the command did and when. The view reports
the newest lines to its host, so the agent's next turn knows what the person
did; ``scene_state`` returns the same lines.

A person's run of the same kind of change on the same object (dragging an
opacity slider, stepping through time, turning the camera) is one line,
updated in place, not one line per step.
"""

from __future__ import annotations

import time
from typing import Any

KEEP = 12
ACTORS = ("person", "agent")


def _num(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _vector(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(_num(v) for v in value) + ")"
    return _num(value)


def _object(scene: dict[str, Any] | None, object_id: Any) -> str:
    if scene is not None:
        for item in scene.get("objects") or []:
            if object_id in (item["id"], item.get("label")):
                label = item.get("label") or item.get("type")
                return f"{item['id']} {label}"
    return str(object_id)


def _properties(properties: dict[str, Any] | None) -> str:
    if not properties:
        return ""
    return ", ".join(f"{k} {_vector(v)}" for k, v in properties.items())


def describe(
    name: str,
    args: dict[str, Any],
    result: dict[str, Any] | None,
    scene: dict[str, Any] | None,
) -> tuple[str, str] | None:
    """A (key, text) pair for a committed command, or None when it says nothing.

    The key groups repeated changes of one kind on one target.
    """
    result = result or {}
    if name == "open_view":
        dataset = (scene or {}).get("dataset") or {}
        what = dataset.get("path") or dataset.get("session") or args.get("scene")
        how = "restored" if result.get("restored") else "opened"
        return "open", f"{how} {what}"
    if name == "create_filter":
        made = result.get("object") or {}
        text = f"added {made.get('id')} {made.get('type') or args.get('filter')}"
        text += f" on {_object(scene, made.get('input') or args.get('input'))}"
        props = _properties(args.get("properties"))
        text += f" ({props})" if props else ""
        if args.get("hide_input"):
            text += ", its input hidden"
        return f"add:{made.get('id')}", text
    if name == "set_filter_property":
        made = result.get("object") or {}
        target = _object(scene, made.get("id") or args.get("object"))
        before = result.get("before") or {}
        parts = [
            f"{key} {_vector(before.get(key)) if before.get(key) is not None else 'default'}"
            f" -> {_vector(value)}"
            for key, value in (args.get("properties") or {}).items()
        ]
        keys = ",".join(sorted(args.get("properties") or {}))
        return f"set:{made.get('id')}:{keys}", f"{target}: " + ", ".join(parts)
    if name == "remove_scene_object":
        removed = ", ".join(result.get("removed") or [str(args.get("object"))])
        return f"remove:{removed}", f"removed {removed}"
    if name == "set_representation":
        return _representation(args, scene)
    if name == "set_timestep":
        state = result.get("time") or {}
        steps = state.get("steps") or 0
        text = f"time step {int(state.get('index', 0)) + 1} of {steps}"
        if state.get("time") is not None:
            text += f" (time {_num(state['time'])})"
        if state.get("follow_latest"):
            text += ", following the latest step"
        return "time", text
    if name in ("set_camera", "fit_camera"):
        # A preset pressed twice is one line; different presets are separate lines.
        text = _camera(name, args)
        return f"camera:{text}", text
    return None


def _representation(
    args: dict[str, Any], scene: dict[str, Any] | None
) -> tuple[str, str]:
    target = _object(scene, args.get("object"))
    parts = []
    if args.get("visible") is not None:
        parts.append("shown" if args["visible"] else "hidden")
    if args.get("representation"):
        parts.append(f"drawn as {args['representation']}")
    color = args.get("color_by")
    if color is not None:
        if color in ("", "none", "solid"):
            parts.append("solid colour")
        else:
            component = args.get("component")
            which = (
                "" if component in (None, "Magnitude") else f" component {component}"
            )
            parts.append(f"coloured by {color}{which}")
    elif args.get("component") is not None:
        parts.append(f"colour component {args['component']}")
    if args.get("colormap"):
        parts.append(f"colour map {args['colormap']}")
    if args.get("rescale"):
        parts.append("colour range rescaled to the data")
    if args.get("scalar_bar") is not None:
        parts.append("colour legend " + ("on" if args["scalar_bar"] else "off"))
    if args.get("opacity") is not None:
        parts.append(f"opacity {_num(float(args['opacity']))}")
    fields = ",".join(sorted(k for k, v in args.items() if v is not None))
    return f"display:{args.get('object')}:{fields}", f"{target}: " + (
        ", ".join(parts) or "display unchanged"
    )


def _camera(name: str, args: dict[str, Any]) -> str:
    if name == "fit_camera":
        target = args.get("object")
        return (
            f"camera fitted to {target}" if target else "camera reset to fit everything"
        )
    parts = []
    if args.get("direction"):
        parts.append(f"camera looks along {args['direction'].upper()}")
    if args.get("parallel_projection") is not None:
        parts.append(
            "parallel projection"
            if args["parallel_projection"]
            else "perspective projection"
        )
    if args.get("position") or args.get("focal_point"):
        parts.append("camera placed")
    turns = [
        f"{label} {_num(float(args[key]))}°"
        for key, label in (("azimuth", "azimuth"), ("elevation", "elevation"))
        if args.get(key)
    ]
    if turns:
        parts.append("camera turned " + ", ".join(turns))
    if args.get("zoom") not in (None, 1, 1.0):
        parts.append(f"zoom {_num(float(args['zoom']))}")
    return ", ".join(parts) or "camera moved"


class ChangeLog:
    """The newest scene changes, oldest first, at most ``KEEP``."""

    def __init__(self, keep: int = KEEP) -> None:
        self.keep = keep
        self.entries: list[dict[str, Any]] = []

    def record(self, by: str, key: str, text: str) -> dict[str, Any]:
        """Add a change, or update the newest one when it is the same run."""
        if by not in ACTORS:
            by = "agent"
        stamp = time.strftime("%H:%M:%S", time.localtime())
        last = self.entries[-1] if self.entries else None
        if last is not None and last["by"] == by == "person" and last["key"] == key:
            last.update(text=text, at=stamp, count=last["count"] + 1)
            return last
        entry = {"by": by, "key": key, "text": text, "at": stamp, "count": 1}
        self.entries.append(entry)
        del self.entries[: -self.keep]
        return entry

    def recent(self) -> list[dict[str, Any]]:
        """The entries as the view and ``scene_state`` report them."""
        return [{k: e[k] for k in ("by", "text", "at")} for e in self.entries]

    def clear(self) -> None:
        self.entries.clear()


def lines(entries: list[dict[str, Any]]) -> list[str]:
    """One readable line per change, for the agent."""
    return [
        f"{e.get('at', '')} by the {e.get('by', 'agent')}: {e.get('text', '')}".strip()
        for e in entries
    ]
