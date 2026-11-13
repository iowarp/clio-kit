"""The scene manifest: the whole scene as data, written after every commit.

A manifest is enough to rebuild a scene in a fresh ParaView: the datasets,
the filters with their properties, how each object is shown, the timestep,
the camera and the view mode. It is plain JSON in the node half's folder.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

SCHEMA = 1


def scene_id_for(path: str) -> str:
    """A stable scene id for a dataset path, so reopening finds its scene."""
    stem = os.path.basename(path.rstrip("/")) or "data"
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", stem)
    slug = re.sub(r"[^A-Za-z0-9]+", "-", stem).strip("-").lower()[:24] or "data"
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}"


def new_scene(scene_id: str, dataset: dict[str, Any] | None) -> dict[str, Any]:
    """An empty scene with its default time, camera and view state."""
    return {
        "schema": SCHEMA,
        "scene": scene_id,
        "dataset": dataset,
        "objects": [],
        "next_id": 1,
        "time": {"follow_latest": True, "index": 0},
        "camera": None,
        "view": {"mode": "local", "pinned": False},
        "selection": None,
        "updated": None,
    }


def scene_dir(root: Path, scene_id: str) -> Path:
    """The folder that holds one scene's manifest and exports."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,80}", scene_id):
        raise ValueError(f"not a scene id: {scene_id!r}")
    return root / "scenes" / scene_id


def write(root: Path, scene: dict[str, Any]) -> Path:
    """Write ``scene`` atomically to its manifest file and return the path."""
    folder = scene_dir(root, scene["scene"])
    folder.mkdir(parents=True, exist_ok=True)
    record = copy.deepcopy(scene)
    record["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    target = folder / "manifest.json"
    temporary = folder / "manifest.json.tmp"
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True), "utf-8")
    os.replace(temporary, target)
    return target


def read(root: Path, scene_id: str) -> dict[str, Any] | None:
    """Read a scene's manifest, or None when there is none."""
    path = scene_dir(root, scene_id) / "manifest.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text("utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError(f"scene {scene_id} has manifest schema {data.get('schema')}")
    return data


def known_scenes(root: Path) -> list[str]:
    """Every scene id with a manifest in this folder."""
    base = root / "scenes"
    if not base.is_dir():
        return []
    return sorted(p.parent.name for p in base.glob("*/manifest.json"))


def find_object(scene: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    """The object with id or label ``object_id``."""
    for item in scene["objects"]:
        if object_id in (item["id"], item.get("label")):
            return item
    return None


def descendants(scene: dict[str, Any], object_id: str) -> list[str]:
    """Ids of ``object_id`` and every object built on it, inputs first."""
    out = [object_id]
    for item in scene["objects"]:
        if item.get("input") in out:
            out.append(item["id"])
    return out
