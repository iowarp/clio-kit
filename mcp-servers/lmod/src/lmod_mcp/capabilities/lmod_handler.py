"""
Lmod handler for executing module commands and parsing their output.
"""

import asyncio
import os
import re
from typing import List, Dict, Optional, Any

from .module_runtime import lmod_command, run_lmod

_COLLECTION_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
_EMPTY_SAVE_HINT = (
    "This server has no module loaded: it has no load tool and only sees the "
    "modules its own process started with. Start it from a shell where the "
    "modules are already loaded, or set LMOD_SYSTEM_DEFAULT_MODULES=<mod1:mod2> "
    "for the server and call module_restore with collection_name='system', then "
    "save again."
)


async def _run_module_command(
    args: List[str], capture_stderr: bool = False
) -> tuple[str, str, int]:
    """
    Run a module command and return stdout, stderr, and return code.

    Args:
        args: Command arguments (e.g., ['list'])
        capture_stderr: Whether to capture stderr separately

    Returns:
        tuple: (stdout, stderr, return_code)
    """
    backend = lmod_command()
    if backend:
        try:
            return await run_lmod(backend, args, capture_stderr)
        except (OSError, ValueError) as exc:
            return "", f"Unable to invoke Lmod: {exc}", 1

    # Compatibility for sites that provide an executable module wrapper.
    cmd = ["module"] + args

    # Preserve warning exit codes: quiet mode can report a refused save as success.
    env = os.environ.copy()
    env.pop("LMOD_QUIET", None)

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
            if capture_stderr
            else asyncio.subprocess.STDOUT,
            env=env,
        )
        stdout, stderr = await process.communicate()

        stdout_str = stdout.decode("utf-8") if stdout else ""
        stderr_str = stderr.decode("utf-8") if stderr else ""

        return stdout_str, stderr_str, process.returncode or 0
    except FileNotFoundError:
        return (
            "",
            "Module command not found. Set LMOD_CMD to Lmod's libexec/lmod "
            "executable, or put `lmod` on PATH.",
            1,
        )
    except Exception as e:
        return "", f"Error running module command: {str(e)}", 1


async def list_loaded_modules() -> Dict[str, Any]:
    """List all currently loaded modules."""
    stdout, stderr, returncode = await _run_module_command(
        ["list", "-t"], capture_stderr=True
    )

    if returncode != 0:
        return {
            "success": False,
            "error": stderr or "Failed to list modules",
            "modules": [],
        }

    # Parse module list (skip header lines)
    modules = []
    lines = (stdout or stderr).strip().split("\n")

    for line in lines:
        line = line.strip()
        # Skip empty lines and headers
        if (
            line
            and not line.startswith("Currently Loaded")
            and not line.startswith("No modules")
        ):
            modules.append(line)

    return {"success": True, "modules": modules, "count": len(modules)}


async def search_available_modules(pattern: Optional[str] = None) -> Dict[str, Any]:
    """Search for available modules."""
    args = ["avail", "-t"]
    if pattern:
        args.append(pattern)

    # module avail writes to stderr by default
    stdout, stderr, returncode = await _run_module_command(args, capture_stderr=True)

    # For module avail, output is in stderr
    output = stderr if stderr else stdout

    if returncode != 0:
        return {
            "success": False,
            "error": stderr or "Failed to search modules",
            "modules": [],
        }

    # Parse available modules
    modules = []
    lines = output.strip().split("\n")

    for line in lines:
        line = line.strip()
        # Skip headers, empty lines, and "name/" directory entries (not modules)
        if line and not line.endswith((":", "/")) and not line.startswith("/"):
            modules.append(line)

    return {
        "success": True,
        "modules": sorted(modules),
        "count": len(modules),
        "pattern": pattern,
    }


async def show_module_details(module_name: str) -> Dict[str, Any]:
    """Show detailed information about a module."""
    stdout, stderr, returncode = await _run_module_command(
        ["show", module_name], capture_stderr=True
    )

    if returncode != 0:
        return {
            "success": False,
            "error": stderr or f"Module {module_name} not found",
            "module": module_name,
        }

    # Parse module information
    info: Dict[str, Any] = {
        "module": module_name,
        "path": None,
        "help": [],
        "whatis": [],
        "prerequisites": [],
        "conflicts": [],
        "environment": [],
    }

    # Ensure list fields are properly typed
    help_list: List[str] = info["help"]
    whatis_list: List[str] = info["whatis"]
    prereq_list: List[str] = info["prerequisites"]
    conflicts_list: List[str] = info["conflicts"]
    env_list: List[str] = info["environment"]

    lines = (stdout or stderr).strip().split("\n")

    for line in lines:
        # Lmod prints the modulefile path as a header ending with a colon.
        if ".lua" in line or ".tcl" in line:
            info["path"] = line.strip().removesuffix(":")
            continue
        # Detect section headers
        if line.strip().endswith(":"):
            line.strip().rstrip(":").lower()
            continue

        # Skip separator lines
        if line.strip().startswith("---") or not line.strip():
            continue

        # Parse content based on current section
        if "help" in line.lower():
            help_match = re.search(r"help\s*\(\s*\[\[(.+?)\]\]\s*\)", line)
            if help_match:
                help_list.append(help_match.group(1))
        elif "whatis" in line.lower():
            whatis_match = re.search(r'whatis\s*\(\s*"(.+?)"\s*\)', line)
            if whatis_match:
                whatis_list.append(whatis_match.group(1))
        elif "prereq" in line.lower():
            prereq_match = re.search(r'prereq\s*\(\s*"(.+?)"\s*\)', line)
            if prereq_match:
                prereq_list.append(prereq_match.group(1))
        elif "conflict" in line.lower():
            conflict_match = re.search(r'conflict\s*\(\s*"(.+?)"\s*\)', line)
            if conflict_match:
                conflicts_list.append(conflict_match.group(1))
        elif (
            "setenv" in line.lower()
            or "prepend_path" in line.lower()
            or "append_path" in line.lower()
        ):
            env_list.append(line.strip())

    # Update info dict with modified lists
    info.update(
        {
            "help": help_list,
            "whatis": whatis_list,
            "prerequisites": prereq_list,
            "conflicts": conflicts_list,
            "environment": env_list,
        }
    )

    return {"success": True, **info}


async def spider_search(pattern: Optional[str] = None) -> Dict[str, Any]:
    """Search entire module tree using spider."""
    args = ["spider", "-t"]
    if pattern:
        args.append(pattern)

    stdout, stderr, returncode = await _run_module_command(args, capture_stderr=True)

    # Spider output is typically in stderr
    output = stderr if stderr else stdout

    if returncode != 0:
        return {
            "success": False,
            "error": stderr or "Failed to run spider search",
            "modules": [],
        }

    # Parse spider output
    modules = {}
    lines = output.strip().split("\n")

    for line in lines:
        line = line.strip()
        if (
            line
            and not line.endswith("/")  # terse "name/" directory entry
            and not line.startswith("The following")
            and not line.startswith("To find")
        ):
            # Extract module name and versions
            if ":" in line:
                name, versions = line.split(":", 1)
                modules[name.strip()] = [v.strip() for v in versions.strip().split(",")]
            else:
                modules[line] = []

    return {"success": True, "modules": modules, "pattern": pattern}


def _invalid_collection(collection_name: str) -> Optional[Dict[str, Any]]:
    """Reject names Lmod would treat as a path outside ~/.lmod.d or as an option."""
    if _COLLECTION_NAME.fullmatch(collection_name):
        return None
    return {
        "success": False,
        "error": (
            f"Invalid collection name {collection_name!r}: use letters, digits, "
            "'_', '.' or '-' (no path separators, not starting with '.' or '-')"
        ),
        "collection": collection_name,
    }


async def save_module_collection(collection_name: str) -> Dict[str, Any]:
    """Save current module configuration."""
    if invalid := _invalid_collection(collection_name):
        return invalid
    stdout, stderr, returncode = await _run_module_command(
        ["save", collection_name], capture_stderr=True
    )

    if returncode != 0 and "empty collection" in stderr:
        stderr += _EMPTY_SAVE_HINT
    if returncode != 0:
        return {
            "success": False,
            "error": stderr or f"Failed to save collection {collection_name}",
            "collection": collection_name,
        }

    return {
        "success": True,
        "message": f"Successfully saved module collection as {collection_name}",
        "collection": collection_name,
    }


async def restore_module_collection(collection_name: str) -> Dict[str, Any]:
    """Restore a saved module collection."""
    if invalid := _invalid_collection(collection_name):
        return invalid
    stdout, stderr, returncode = await _run_module_command(
        ["restore", collection_name], capture_stderr=True
    )

    if returncode != 0:
        return {
            "success": False,
            "error": stderr or f"Failed to restore collection {collection_name}",
            "collection": collection_name,
        }

    # Get the list of loaded modules after restore
    loaded_modules = await list_loaded_modules()

    return {
        "success": True,
        "message": f"Successfully restored module collection {collection_name}",
        "collection": collection_name,
        "loaded_modules": loaded_modules.get("modules", []),
    }


async def list_saved_collections() -> Dict[str, Any]:
    """List all saved module collections."""
    stdout, stderr, returncode = await _run_module_command(
        ["savelist"], capture_stderr=True
    )

    if returncode != 0:
        return {
            "success": False,
            "error": stderr or "Failed to list saved collections",
            "collections": [],
        }

    # Parse collection names
    collections = []
    lines = (stdout or stderr).strip().split("\n")

    for line in lines:
        line = line.strip()
        # Skip headers and empty lines
        if (
            line
            and not line.startswith("Named collection")
            and not line.startswith("No named")
        ):
            # Extract collection names (patterns like "1) name   2) name2   3) name3")
            matches = re.findall(r"\d+\)\s*([^\d\)]+?)(?=\s*\d+\)|$)", line)
            if matches:
                for match in matches:
                    collections.append(match.strip())
            elif line and not any(char in line for char in [":", "(", ")"]):
                collections.append(line)

    return {"success": True, "collections": collections, "count": len(collections)}
