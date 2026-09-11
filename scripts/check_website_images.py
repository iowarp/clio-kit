"""Reject image containers affected by unpatched image-size parser advisories.

Docusaurus reads local Markdown images during builds. Check their byte signatures
before that parser runs, including files with misleading extensions. Remove this
guard after upgrading to an upstream fix for GHSA-w3rx-r6r6-pgpr and
GHSA-5p2g-fcmc-qvqq. This does not make npm audit report the dependency as patched.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

EXCLUDED = {"node_modules", "build", ".docusaurus", ".git", ".cache-loader"}


def unsupported_format(header: bytes) -> str | None:
    """Detect affected containers without invoking any image decoder."""
    if header.startswith(b"icns"):
        return "ICNS"
    if header.startswith((b"\xff\x0a", b"\x00\x00\x00\x0cJXL \r\n\x87\n")):
        return "JPEG XL"
    if header[4:8] == b"ftyp":
        return "ISO BMFF (including HEIF/AVIF)"
    return None


def check_images(root: Path) -> list[str]:
    failures = []
    for directory, children, files in os.walk(root):
        children[:] = sorted(child for child in children if child not in EXCLUDED)
        for name in sorted(files):
            path = Path(directory) / name
            with path.open("rb") as stream:
                kind = unsupported_format(stream.read(32))
            if kind:
                failures.append(f"{path.relative_to(root)}: {kind}")
    return failures


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "clio-kit-website"
    failures = check_images(root)
    if failures:
        print("Unsupported website image containers:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        print("Use PNG, JPEG, GIF, WebP or SVG assets instead.", file=sys.stderr)
        return 1
    print("PASS: website assets contain no blocked image containers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
