"""DECIMER MCP server: read a chemical structure drawing into SMILES.

DECIMER (Rajan et al.; Steinbeck group) is a TensorFlow image-to-SMILES model. Of the
four optical structure recognition models ChemGraph benchmarks (DECIMER, MolNexTR,
MolScribe, OCSRGlyph; 722 images), it has the highest exact match (0.90). TensorFlow
lives only in this server's environment; the agent's own kernel never imports it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, TypeVar

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from mcp.types import TextContent
from pydantic import Field

from . import images, present, recognize
from .errors import DecimerError

PATHS_META = "ai.iowarp.clio/paths"

mcp = FastMCP(
    "decimer",
    version="1.0.0",  # keep equal to mcp-server-versions.toml
    instructions=(
        "Read a chemical structure drawing (a PNG or JPEG of one molecule) into SMILES "
        "with the DECIMER model, and check the SMILES with RDKit. image_to_smiles returns "
        "the SMILES, whether RDKit accepts it, its canonical form, formula and mass; pass "
        "hand_drawn=true for a sketch. A figure with several molecules or text should be "
        "cut down to one structure first with crop_image. The SMILES is a model's reading: "
        "have the person compare it with the image (for example in Ketcher) before using "
        "it. The first call downloads about 300 MB of weights and is slow."
    ),
)

T = TypeVar("T")


def _run(fn: Callable[[], T]) -> T:
    try:
        return fn()
    except DecimerError as exc:
        raise ToolError(exc.text()) from exc


def _result(text: str, data: dict[str, Any]) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], structured_content=data)


@mcp.tool(
    name="image_to_smiles",
    title="Image to SMILES",
    description=(
        "Read the chemical structure drawn in an image (PNG, JPEG, GIF, TIFF, BMP, WEBP; one "
        "molecule) into a SMILES string with DECIMER, then check it with RDKit: valid or "
        "not, canonical SMILES, formula, molecular weight (g/mol), stereo centres, "
        "InChIKey, and the model's per-token confidence. An invalid SMILES is still "
        "returned as the model wrote it. hand_drawn=true uses DECIMER's model for sketches. "
        "The result is a model's reading and must be checked against the image."
    ),
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    tags={"chemistry", "ocsr"},
    meta={PATHS_META: {"image_path": {"machine": "local", "access": "read"}}},
)
async def image_to_smiles(
    image_path: Annotated[str, Field(description="Path of the image of one structure.")],
    hand_drawn: Annotated[
        bool, Field(description="True for a hand-drawn sketch, false for a printed drawing.")
    ] = False,
) -> ToolResult:
    def work() -> dict[str, Any]:
        return recognize.image_to_smiles(images.check_image(image_path), hand_drawn)

    data = _run(work)
    return _result(present.smiles_text(data), data)


@mcp.tool(
    name="crop_image",
    title="Crop Image",
    description=(
        "Cut a rectangle out of an image (for example one structure from a paper's "
        "figure) and save it as PNG under output_dir/chemistry/decimer/. box is "
        "[left, top, right, bottom] in pixels from the top-left corner, or as fractions "
        "of the width and height with unit='fraction'. Returns the new image's path."
    ),
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    tags={"chemistry", "ocsr"},
    meta={
        PATHS_META: {
            "image_path": {"machine": "local", "access": "read"},
            "output_dir": {"machine": "local", "access": "write"},
        }
    },
)
async def crop_image(
    image_path: Annotated[str, Field(description="Path of the image to crop.")],
    box: Annotated[
        list[float], Field(description="[left, top, right, bottom].", min_length=4, max_length=4)
    ],
    unit: Annotated[str, Field(description="'pixel' (default) or 'fraction'.")] = "pixel",
    output_dir: Annotated[
        str | None,
        Field(description="Folder for the crop; default $CLIO_KIT_ARTIFACTS or the cwd."),
    ] = None,
) -> ToolResult:
    data = _run(lambda: images.crop_image(image_path, box, unit, output_dir))
    return _result(present.crop_text(data), data)


@mcp.resource("decimer://capabilities")
def capabilities() -> dict[str, Any]:
    """Describe what the decimer MCP server can do."""
    return {
        "tools": ["image_to_smiles", "crop_image"],
        "model": "DECIMER 2.8.1 (TensorFlow 2.20 CPU), printed and hand-drawn models",
        "benchmark": "exact match 0.899 on ChemGraph's 722-image OCSR benchmark",
        "weights": f"about 300 MB per model from Zenodo, cached in {recognize.cache_dir()}",
        "accepts": "PNG, JPEG, GIF, TIFF, BMP, WEBP, HEIC images of one structure",
    }


def main() -> None:
    """Entry point for the decimer MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
