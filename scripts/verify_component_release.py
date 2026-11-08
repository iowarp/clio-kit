"""Verify that a release contains every component pinned by its launcher."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile


def verify(dist: Path) -> dict:
    wheels = list(dist.glob("*.whl"))
    sdists = list(dist.glob("clio_kit-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Expected exactly one launcher wheel and sdist")
    with zipfile.ZipFile(wheels[0]) as wheel:
        index = json.loads(wheel.read("clio_kit/_components.json"))
        if any(".data/data/" in name for name in wheel.namelist()):
            raise ValueError("Launcher wheel contains component payloads")
    with tarfile.open(sdists[0]) as sdist:
        member = next(
            m for m in sdist if m.name.endswith("/src/clio_kit/_components.json")
        )
        source = sdist.extractfile(member)
        assert source is not None
        if json.load(source) != index:
            raise ValueError("Wheel and sdist component catalogues differ")
    assets = dist / "components"
    expected = {record["file"] for record in index["artifacts"].values()}
    actual = {path.name for path in assets.glob("*.tar.gz")}
    if actual != expected:
        raise ValueError(
            f"Component assets differ: missing={expected - actual}, unexpected={actual - expected}"
        )
    for record in index["artifacts"].values():
        content = (assets / record["file"]).read_bytes()
        if (
            len(content) != record["size"]
            or hashlib.sha256(content).hexdigest() != record["sha256"]
        ):
            raise ValueError(f"Component identity mismatch: {record['file']}")
    return {
        "version": index["version"],
        "assets": len(expected),
        "wheel_bytes": wheels[0].stat().st_size,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.dist)))
