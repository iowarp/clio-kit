"""The view descriptor: how ParaView opens one output, and whether it is still growing.

The schema is CLIO's view descriptor v1, vendored unchanged in
``schemas/view-descriptor.v1.json`` from iowarp/clio-jarvis
(``schemas/view-descriptor.v1.json`` at a6aba08, sha256 8773b9ed...c1b076). A
simulation package reports one per output; any MCP host may pass one to
``open_view``. The model below takes every field of v1 and is lenient where the
schema is strict: only ``path`` is required and unknown fields are ignored, so a
host that knows only the path, or a later schema version, still opens the data.
It also takes ``geometry``, surfaces drawn with the data, which v1 does not have
yet; a descriptor without it opens as before.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_ID = "clio.view-descriptor.v1"
SCHEMA_FILE = "view-descriptor.v1.json"


def schema() -> dict[str, Any]:
    """The vendored JSON Schema."""
    text = (
        resources.files("paraview_mcp")
        .joinpath("schemas", SCHEMA_FILE)
        .read_text(encoding="utf-8")
    )
    loaded: dict[str, Any] = json.loads(text)
    return loaded


class _Part(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Follow(_Part):
    signal: Annotated[
        Literal["reader-time-steps", "file-count", "mtime"] | None,
        Field(
            description="How a new step shows: the reader's time list, more files, "
            "or a newer modification time."
        ),
    ] = None
    poll_seconds: Annotated[
        float | None,
        Field(gt=0, le=3600, description="Suggested seconds between checks."),
    ] = None
    expected_steps: Annotated[
        int | None,
        Field(ge=1, description="Steps the producer will write in total, when known."),
    ] = None


class TimeInfo(_Part):
    unit: Annotated[
        str | None, Field(description="Unit of the time values, such as step or s.")
    ] = None
    step_interval: Annotated[
        float | None,
        Field(
            gt=0, description="Simulation steps, or time, between two written steps."
        ),
    ] = None


class Display(_Part):
    array: Annotated[str | None, Field(description="Array to colour by.")] = None
    association: Literal["POINTS", "CELLS"] | None = None
    representation: (
        Literal["Outline", "Surface", "Slice", "Volume", "Contour"] | None
    ) = None
    slice_normal: Annotated[list[float] | None, Field(min_length=3, max_length=3)] = (
        None
    )
    contour_values: Annotated[list[float] | None, Field(min_length=1)] = None
    color_map: Annotated[
        str | None, Field(description="ParaView colour map preset name.")
    ] = None
    range: Annotated[
        list[float] | None,
        Field(
            min_length=2,
            max_length=2,
            description="Fixed colour range, so colours mean the same at every step.",
        ),
    ] = None


class Geometry(_Part):
    path: Annotated[
        str,
        Field(
            description="A surface file ParaView reads, such as an STL, on the same "
            "machine as the data and in the data's coordinates."
        ),
    ]
    label: Annotated[
        str | None, Field(description="Its name in the scene, such as body.")
    ] = None
    visible: Annotated[
        bool, Field(description="Show it when the view opens; else it opens hidden.")
    ] = True


class Producer(_Part):
    package: str | None = None
    version: str | None = None
    sha256: str | None = None


class ViewDescriptor(_Part):
    """How to open one output: what a simulation package or the agent writes for it."""

    schema_: Annotated[
        str | None,
        Field(alias="schema", description=f"The schema and version, {SCHEMA_ID}."),
    ] = None
    name: Annotated[
        str | None, Field(description="The output's name in its package.")
    ] = None
    path: Annotated[
        str,
        Field(
            description="Path or path pattern on the machine where ParaView runs: a "
            "file, a folder (an ADIOS2 .bp output), or a glob such as "
            "/data/run/step_*.vti for a file series."
        ),
    ]
    path_kind: Annotated[
        Literal["file", "directory", "glob"] | None,
        Field(description="What path names: a file, a folder, or a glob."),
    ] = None
    format: Annotated[
        str | None,
        Field(description="The data format, such as adios2-bp5 or vtk-xml-image."),
    ] = None
    reader: Annotated[
        str | None,
        Field(
            description="The ParaView reader, by its paraview.simple name, such as "
            "ADIOS2VTXReader or XMLImageDataReader. Omit to let ParaView choose."
        ),
    ] = None
    reader_properties: Annotated[
        dict[str, Any] | None,
        Field(description="That reader's properties, by ParaView property name."),
    ] = None
    growing: Annotated[
        bool,
        Field(
            description="The source is still being written: re-read its time list as "
            "it grows, and follow the newest step."
        ),
    ] = False
    follow: Follow | None = None
    time: TimeInfo | None = None
    display: Annotated[
        Display | None,
        Field(description="A first view: the array, colours and representation."),
    ] = None
    geometry: Annotated[
        list[Geometry] | None,
        Field(
            description="Surfaces to draw with the data: the body a solver meshes or "
            "voxelises, as it was made, say. Each opens as a scene object of its own."
        ),
    ] = None
    producer: Producer | None = None

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    def extras(self) -> dict[str, Any] | None:
        """What the node keeps with the dataset beyond the reader: the output's name,
        how to follow it, its time unit, the first view, its geometry and its
        producer."""
        kept = {
            "name": self.name,
            "path_kind": self.path_kind,
            "follow": _dump(self.follow),
            "time": _dump(self.time),
            "display": _dump(self.display),
            "geometry": [g.model_dump(exclude_none=True) for g in self.geometry or []],
            "producer": _dump(self.producer),
        }
        kept = {k: v for k, v in kept.items() if v}
        return kept or None


def _dump(part: BaseModel | None) -> dict[str, Any] | None:
    if part is None:
        return None
    return part.model_dump(exclude_none=True) or None
