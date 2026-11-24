"""Scene scripts: the agent builds the scene in ParaView's own language, inside it.

``run_scene_script`` runs ``paraview.simple`` code in this process, the one that owns the
scene and serves its live view, so what the script builds is what the person sees. After
the script, every pipeline source it created is adopted as a scene object (``kind:
"made"``) with the display it was left with, and what it changed in how an existing
object is drawn is kept in that object's display. The step is kept in the manifest
(``kind: "script"``). Restoring a scene replays its scripts in order and pairs the sources
each one creates, in creation order, with the objects it recorded; the recorded displays
(and the person's later edits to them) are applied on top. See docs/scene-scripts.md.
"""

from __future__ import annotations

import traceback
from typing import Any

from . import manifest, ranges
from .camera import same_camera
from .errors import CommandError

#: How much of a failing script's traceback the agent is shown.
TRACE_CHARS = 1500


class SceneScripts:
    """Mixed into the engine: run, adopt and replay scene scripts."""

    simple: Any
    view: Any
    root: Any
    scene: dict[str, Any] | None
    proxies: dict[str, Any]

    # Provided by the engine.
    def require_scene(self) -> dict[str, Any]:
        raise NotImplementedError

    def _show(self, item: dict[str, Any], proxy: Any, apply: bool) -> None:
        raise NotImplementedError

    def _apply_time(self, scene: dict[str, Any]) -> None:
        raise NotImplementedError

    def update_view_mode(self) -> str:
        raise NotImplementedError

    def summary(self, restored: bool = False) -> dict[str, Any]:
        raise NotImplementedError

    def object_summary(self, item: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def read_camera(self) -> dict[str, Any]:
        raise NotImplementedError

    def hold_camera(self, held: bool) -> None:
        raise NotImplementedError

    # -- running ------------------------------------------------------------
    def _sources(self) -> dict[int, Any]:
        return {
            int(proxy.GetGlobalID()): proxy
            for proxy in self.simple.GetSources().values()
        }

    def _views(self) -> dict[int, Any]:
        return {int(view.GetGlobalID()): view for view in self.simple.GetViews()}

    def _namespace(self, scene: dict[str, Any]) -> dict[str, Any]:
        objects: dict[str, Any] = {}
        for item in scene["objects"]:
            proxy = self.proxies.get(item["id"])
            if proxy is not None:
                objects[item["id"]] = proxy
                if item.get("label"):
                    objects.setdefault(str(item["label"]), proxy)
        folder = manifest.scene_dir(self.root, scene["scene"])
        folder.mkdir(parents=True, exist_ok=True)
        namespace: dict[str, Any] = {
            "simple": self.simple,
            "view": self.view,
            "objects": objects,
            "folder": str(folder),
            "__name__": "__scene_script__",
        }
        namespace.update(
            {
                name: getattr(self.simple, name)
                for name in dir(self.simple)
                if name[:1].isupper()
            }
        )
        return namespace

    def _execute(self, scene: dict[str, Any], code: str, label: str) -> list[Any]:
        """Run ``code``; the sources it created, in creation order. A script that fails,
        or makes a render view of its own, leaves nothing behind."""
        sources, views = self._sources(), self._views()
        self.simple.SetActiveView(self.view)
        error: CommandError | None = None
        try:
            exec(  # noqa: S102 - running the agent's scene script is this tool's purpose
                compile(code, f"<scene script {label}>", "exec"), self._namespace(scene)
            )
        except Exception as exc:  # noqa: BLE001 - any script error is reported, typed
            tail = traceback.format_exc()[-TRACE_CHARS:]
            error = CommandError("script_error", f"The script failed: {exc}\n{tail}")
        made = [p for key, p in sorted(self._sources().items()) if key not in sources]
        extra_views = [v for key, v in self._views().items() if key not in views]
        if error is None and extra_views:
            error = CommandError(
                "script_made_a_view",
                "The script made a render view of its own. Build into `view`, the "
                "scene's view; use export_animation for videos.",
            )
        if error is not None:
            for proxy in reversed(made):
                self.simple.Delete(proxy)
            for view in extra_views:
                self.simple.Delete(view)
            self.simple.SetActiveView(self.view)
            raise error
        self.simple.SetActiveView(self.view)
        return made

    def _adopt(self, proxy: Any) -> dict[str, Any]:
        """The display a script left on ``proxy``, as the manifest keeps displays."""
        from paraview import servermanager

        rep = servermanager.GetRepresentation(proxy, self.view)
        if rep is None:
            return {"visible": False}
        named = [str(v) for v in rep.ColorArrayName]
        color_by = named if len(named) == 2 and named[1] else None
        display: dict[str, Any] = {
            "visible": bool(rep.Visibility),
            "representation": str(
                rep.SMProxy.GetProperty("Representation").GetElement(0)
            ),
            "color_by": color_by,
            "opacity": float(rep.Opacity),
            "colormap": None,
            "scalar_bar": bool(color_by) and bool(rep.IsScalarBarVisible(self.view)),
        }
        if color_by is None:
            display["solid_color"] = [round(float(v), 4) for v in rep.DiffuseColor]
        return display

    def _shown_before(self, scene: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """How each object already in the scene is drawn, before a script runs."""
        return {
            item["id"]: self._adopt(self.proxies[item["id"]])
            for item in scene["objects"]
            if item.get("kind") != "script" and item["id"] in self.proxies
        }

    def _keep_changes(
        self, scene: dict[str, Any], before: dict[str, dict[str, Any]]
    ) -> None:
        """What a script changed in how an existing object is drawn is kept in its
        display, as set_representation keeps it; otherwise the next redraw, and a
        restore after the script's replay, would put the old display back."""
        for item in scene["objects"]:
            was = before.get(item["id"])
            proxy = self.proxies.get(item["id"])
            if was is None or proxy is None:
                continue
            now = self._adopt(proxy)
            if now == was:
                continue
            display = item.setdefault("display", {})
            if "color_by" in now and now["color_by"] != was.get("color_by"):
                # A new field's colour map is the one the script left; the old
                # field's range and component do not carry over.
                display.update(
                    color_by=now.get("color_by"),
                    color_component=None,
                    colormap=None,
                    color_range=None,
                    auto_range=None,
                    range_mode=None,
                )
                if now.get("color_by"):
                    display["solid_color"] = None
            for key in ("visible", "representation", "opacity", "scalar_bar"):
                if key in now and now[key] != was.get(key):
                    display[key] = now[key]
            if (
                "color_by" in now
                and not now["color_by"]
                and now.get("solid_color") != was.get("solid_color")
            ):
                display["solid_color"] = now.get("solid_color")

    def cmd_run_scene_script(
        self, code: str, label: str | None = None
    ) -> dict[str, Any]:
        scene = self.require_scene()
        step_id = f"s{scene['next_id']}"
        scene["next_id"] += 1
        label = (label or "").strip() or f"script {step_id}"
        before = ranges.spans(self.simple, scene)
        drawn = self._shown_before(scene)
        camera = self.read_camera()
        made = self._execute(scene, code, label)
        self._keep_changes(scene, drawn)
        if not same_camera(camera, self.read_camera()):
            # The script placed the camera: it stays there as the data moves.
            self.hold_camera(True)
        step: dict[str, Any] = {
            "id": step_id,
            "kind": "script",
            "label": label,
            "code": code,
            "made": [],
            "removed": [],
        }
        scene["objects"].append(step)
        added = []
        for proxy in made:
            object_id = f"o{scene['next_id']}"
            scene["next_id"] += 1
            names = [
                name for (name, _), p in self.simple.GetSources().items() if p is proxy
            ]
            item = {
                "id": object_id,
                "kind": "made",
                "by": step_id,
                "label": names[0] if names else object_id,
                "type": proxy.GetXMLName(),
                "display": self._adopt(proxy),
            }
            step["made"].append(object_id)
            scene["objects"].append(item)
            self.proxies[object_id] = proxy
            added.append(item)
        self._apply_time(scene)
        ranges.after_script(self, scene, before)
        self.update_view_mode()
        return {
            "step": {"id": step_id, "label": label, "lines": len(code.splitlines())},
            "added": [self.object_summary(item) for item in added],
            "scene": self.summary(),
        }

    # -- replaying ------------------------------------------------------------
    def replay_script(self, step: dict[str, Any]) -> dict[str, Any]:
        """Run a recorded step again; its sources by recorded object id."""
        scene = self.require_scene()
        made = self._execute(scene, step["code"], step.get("label") or step["id"])
        if len(made) != len(step.get("made") or []):
            for proxy in reversed(made):
                self.simple.Delete(proxy)
            raise CommandError(
                "script_replay_differs",
                f"Replaying {step.get('label')} made {len(made)} objects, not the "
                f"{len(step.get('made') or [])} it made before (its data may have changed).",
            )
        pairs = dict(zip(step.get("made") or [], made, strict=True))
        for object_id in step.get("removed") or []:
            proxy = pairs.pop(object_id, None)
            if proxy is not None:
                self.simple.Delete(proxy)
        return pairs


def forget_made(scene: dict[str, Any], gone: list[str]) -> None:
    """Objects a script made that were removed: its replay removes them again."""
    for step in scene["objects"]:
        if step.get("kind") == "script":
            removed = [
                object_id for object_id in step.get("made") or [] if object_id in gone
            ]
            step["removed"] = sorted(set(step.get("removed") or []) | set(removed))
