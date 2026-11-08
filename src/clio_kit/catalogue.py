"""The client-independent package catalogue and native marketplace exports."""

import json
from pathlib import Path

CATALOGUE_PATH = ".clio-kit/catalogue.json"


def read_catalogue(root: Path) -> dict:
    path = root / CATALOGUE_PATH
    if path.is_file():
        data = json.loads(path.read_text())
        if data.get("schema") != 1:
            raise ValueError("Unsupported CLIO Kit catalogue schema")
    else:
        # Existing third-party Claude catalogues remain accepted as input.
        native = json.loads((root / ".claude-plugin/marketplace.json").read_text())
        data = {
            "schema": 1,
            "name": native.get("name", "clio-kit"),
            "packages": native["plugins"],
        }
    entries = data.get("packages")
    if not isinstance(entries, list) or any(
        not isinstance(e, dict)
        or not isinstance(e.get("name"), str)
        or "source" not in e
        for e in entries
    ):
        raise ValueError("Catalogue needs named package sources")
    if len({e["name"] for e in entries}) != len(entries):
        raise ValueError("Duplicate catalogue package names")
    return data


def write_catalogue(root: Path, marketplace: dict) -> None:
    path = root / CATALOGUE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "name": marketplace["name"],
                "packages": marketplace["plugins"],
            },
            indent=2,
        )
        + "\n"
    )
