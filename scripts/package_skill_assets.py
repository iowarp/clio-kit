"""Build a metadata-only launcher and separate immutable component artifacts."""

import json
from pathlib import Path
import runpy

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        # Editable development keeps direct checkout access and builds no artifacts.
        if version == "editable":
            return
        root = Path(self.root)
        index = root / "src/clio_kit/_components.json"
        if (root / "mcp-servers").exists():
            output = Path(self.directory) / "components"
            builder = runpy.run_path(str(root / "scripts/package_components.py"))
            catalogue = builder["build_components"](root, output)
            # Generated metadata lives under the build directory, never in source.
            index = output / "index.json"
            assert catalogue["schema"] == 1
        elif not index.is_file():
            raise ValueError("Source distribution is missing its component catalogue")
        data = json.loads(index.read_text())
        if data["version"] != self.metadata.version:
            raise ValueError("Component catalogue version differs from the launcher")
        destination = (
            "clio_kit/_components.json"
            if self.target_name == "wheel"
            else "src/clio_kit/_components.json"
        )
        build_data.setdefault("force_include", {})[str(index)] = destination
