"""Malformed image containers must be rejected before Docusaurus decodes them."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "check_website_images",
    Path(__file__).resolve().parents[1] / "scripts/check_website_images.py",
)
assert spec and spec.loader
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


@pytest.mark.parametrize(
    "payload",
    [
        b"icns\x00\x00\x00\x10ic07\x00\x00\x00\x00",
        b"\xff\x0a" + bytes(14),
        b"\x00\x00\x00\x0cJXL \r\n\x87\n" + bytes(16),
        b"\x00\x00\x00\x00ftypheic" + bytes(16),
        b"\x00\x00\x00\x00ftypavif" + bytes(16),
    ],
)
def test_rejects_affected_bytes_even_with_png_extension(tmp_path, payload):
    (tmp_path / "misleading.png").write_bytes(payload)
    failures = guard.check_images(tmp_path)
    assert len(failures) == 1
    assert "misleading.png" in failures[0]


def test_supported_png_and_text_pass(tmp_path):
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(24))
    (tmp_path / "readme.md").write_text("Use PNG images.\n")
    assert guard.check_images(tmp_path) == []
