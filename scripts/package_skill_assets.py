"""Include skill resources from every local component package in built wheels."""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if self.target_name != "wheel":
            return
        root = Path(self.root)
        destinations = {
            f"clio-kit-skills/{folder.name}/skills"
            for folder in (root / "skills").iterdir()
            if folder.is_dir()
        }
        for kind in ("plugins", "agents", "hooks"):
            for folder in sorted((root / kind).glob("*/skills")):
                if not any(folder.glob("*/SKILL.md")):
                    continue
                target = f"clio-kit-skills/{folder.parent.name}/skills"
                if target in destinations:
                    raise ValueError(f"Duplicate skill package destination: {target}")
                if folder.is_symlink() or any(
                    path.is_symlink() for path in folder.rglob("*")
                ):
                    raise ValueError(
                        f"Linked skill content cannot be packaged: {folder}"
                    )
                destinations.add(target)
                build_data.setdefault("shared_data", {})[str(folder)] = target
