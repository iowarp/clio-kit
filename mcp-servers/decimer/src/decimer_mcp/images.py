"""Image checks and cropping: no model here, so these import and test without TensorFlow.

Adapted from ChemGraph (argonne-lcf/ChemGraph, Apache-2.0), tools/ocsr_core.py: the
image is recognised by its leading bytes, never its extension, and checked to be a
regular file of bounded size before it is read.
"""

from __future__ import annotations

import os
import re
import stat
import time
from pathlib import Path
from typing import Any

from .errors import DecimerError

MAX_IMAGE_BYTES = 20_000_000

_MAGIC = {
    b"\x89PNG\r\n\x1a\n": "png",
    b"\xff\xd8\xff": "jpeg",
    b"GIF87a": "gif",
    b"GIF89a": "gif",
    b"II*\x00": "tiff",
    b"MM\x00*": "tiff",
    b"BM": "bmp",
}


def sniff_format(head: bytes) -> str | None:
    """The image format implied by a file's leading bytes, or None."""
    for magic, fmt in _MAGIC.items():
        if head.startswith(magic):
            return fmt
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[4:8] == b"ftyp" and head[8:12] in {b"heic", b"heix", b"mif1", b"msf1"}:
        return "heic"
    return None


def check_image(image_path: str) -> dict[str, Any]:
    """Validate an image file and return ``{path, format, width, height, bytes}``.

    Raises ``DecimerError`` with ``image_not_found``, ``unreadable_image`` or
    ``image_too_large``.
    """
    path = Path(os.path.expanduser(image_path)).resolve()
    try:
        info = path.stat()
    except FileNotFoundError:
        raise DecimerError(
            "image_not_found",
            f"No file at {path}. Pass the path of a PNG or JPEG of the structure drawing.",
            path=str(path),
        ) from None
    if not stat.S_ISREG(info.st_mode):
        raise DecimerError("unreadable_image", f"{path} is not a regular file.", path=str(path))
    if info.st_size > MAX_IMAGE_BYTES:
        raise DecimerError(
            "image_too_large",
            f"{path} is {info.st_size / 1e6:.1f} MB; the limit is "
            f"{MAX_IMAGE_BYTES / 1e6:.0f} MB. Crop the structure out first (crop_image).",
            path=str(path),
            bytes=info.st_size,
        )
    with path.open("rb") as handle:
        head = handle.read(16)
    fmt = sniff_format(head)
    if fmt is None:
        raise DecimerError(
            "unreadable_image",
            f"{path} is not a PNG, JPEG, GIF, TIFF, BMP, WEBP or HEIC image (by its "
            "content, whatever its extension). Export the figure as PNG and try again.",
            path=str(path),
        )
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise DecimerError(
            "unreadable_image",
            f"{path} looks like {fmt.upper()} but could not be decoded ({exc}). "
            "Re-export it as PNG and try again.",
            path=str(path),
        ) from None
    return {
        "path": str(path),
        "format": fmt,
        "width": width,
        "height": height,
        "bytes": info.st_size,
    }


def output_dir(base: str | None, run_name: str) -> Path:
    """``<base or $CLIO_KIT_ARTIFACTS or cwd>/chemistry/decimer/<run_name>/``, created."""
    root = Path(base or os.environ.get("CLIO_KIT_ARTIFACTS") or os.getcwd()).expanduser()
    folder = root / "chemistry" / "decimer" / run_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder.resolve()


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-") or "image"


def crop_image(
    image_path: str, box: list[float], unit: str = "pixel", out_dir: str | None = None
) -> dict[str, Any]:
    """Cut ``box`` = [left, top, right, bottom] out of an image and save it as PNG.

    ``unit`` is ``pixel`` (default) or ``fraction`` (0 to 1 of width and height).
    """
    source = check_image(image_path)
    if len(box) != 4 or unit not in {"pixel", "fraction"}:
        raise DecimerError(
            "bad_box",
            "box must have four numbers [left, top, right, bottom] and unit must be "
            "'pixel' or 'fraction'.",
            box=box,
            unit=unit,
        )
    width, height = source["width"], source["height"]
    if unit == "fraction":
        left, top, right, bottom = (
            box[0] * width,
            box[1] * height,
            box[2] * width,
            box[3] * height,
        )
    else:
        left, top, right, bottom = (float(v) for v in box)
    left, top = max(0, round(left)), max(0, round(top))
    right, bottom = min(width, round(right)), min(height, round(bottom))
    if right - left < 8 or bottom - top < 8:
        raise DecimerError(
            "bad_box",
            f"The box {box} leaves {max(0, right - left)} x {max(0, bottom - top)} pixels of "
            f"a {width} x {height} image. Give [left, top, right, bottom] in pixels, or as "
            "fractions between 0 and 1.",
            box=box,
            image_size=[width, height],
        )
    from PIL import Image

    stem = _slug(Path(source["path"]).stem)
    folder = output_dir(out_dir, f"{stem}-crop-{time.strftime('%Y%m%d-%H%M%S')}")
    target = folder / f"{stem}_crop_{left}_{top}_{right}_{bottom}.png"
    with Image.open(source["path"]) as image:
        image.crop((left, top, right, bottom)).save(target, format="PNG")
    return {
        "source": source["path"],
        "box_pixels": [left, top, right, bottom],
        "box_unit": unit,
        "image": {
            "path": str(target),
            "format": "png",
            "width": right - left,
            "height": bottom - top,
            "bytes": target.stat().st_size,
        },
    }
