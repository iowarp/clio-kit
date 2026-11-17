"""Image checks and cropping, no model needed."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from decimer_mcp import images
from decimer_mcp.errors import DecimerError


def test_check_image_reads_png(caffeine_png: Path) -> None:
    info = images.check_image(str(caffeine_png))
    assert info["format"] == "png"
    assert (info["width"], info["height"]) == (400, 400)


def test_missing_file_is_typed(tmp_path: Path) -> None:
    with pytest.raises(DecimerError) as err:
        images.check_image(str(tmp_path / "nope.png"))
    assert err.value.reason == "image_not_found"
    assert err.value.text().endswith("[image_not_found]")


def test_non_image_renamed_png_is_refused(tmp_path: Path) -> None:
    fake = tmp_path / "structure.png"
    fake.write_text("not an image at all")
    with pytest.raises(DecimerError) as err:
        images.check_image(str(fake))
    assert err.value.reason == "unreadable_image"


def test_truncated_png_is_refused(caffeine_png: Path, tmp_path: Path) -> None:
    broken = tmp_path / "broken.png"
    broken.write_bytes(caffeine_png.read_bytes()[:200])
    with pytest.raises(DecimerError) as err:
        images.check_image(str(broken))
    assert err.value.reason == "unreadable_image"


def test_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(DecimerError) as err:
        images.check_image(str(tmp_path))
    assert err.value.reason == "unreadable_image"


def test_crop_in_pixels(caffeine_png: Path, artifacts: Path) -> None:
    data = images.crop_image(str(caffeine_png), [10, 20, 210, 220])
    out = Path(data["image"]["path"])
    assert out.is_file() and out.is_relative_to(artifacts / "chemistry" / "decimer")
    assert data["box_pixels"] == [10, 20, 210, 220]
    with Image.open(out) as image:
        assert image.size == (200, 200)


def test_crop_in_fractions_with_output_dir(caffeine_png: Path, tmp_path: Path) -> None:
    data = images.crop_image(str(caffeine_png), [0, 0, 0.5, 0.25], "fraction", str(tmp_path))
    assert data["box_pixels"] == [0, 0, 200, 100]
    assert Path(data["image"]["path"]).is_relative_to(tmp_path / "chemistry" / "decimer")


@pytest.mark.parametrize(
    ("box", "unit"), [([0, 0, 2, 2], "pixel"), ([1, 2, 3], "pixel"), ([0, 0, 1, 1], "inch")]
)
def test_bad_box(caffeine_png: Path, artifacts: Path, box: list[float], unit: str) -> None:
    with pytest.raises(DecimerError) as err:
        images.crop_image(str(caffeine_png), box, unit)
    assert err.value.reason == "bad_box"
