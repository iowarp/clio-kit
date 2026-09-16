"""Utility functions for ChronoLog tests."""

import subprocess
import asyncio
from pathlib import Path


async def wait_for_archived_record(chronicle, story, expected, timeout=30):
    """Wait for the native keeper/grapher's asynchronous archive flush."""
    from chronomcp.capabilities.retrieve_handler import retrieve_interaction

    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        result = await retrieve_interaction(chronicle, story)
        if result != "No records found.":
            path = Path(result)
            try:
                assert path.read_text() == expected
            finally:
                path.unlink(missing_ok=True)
            return
        assert asyncio.get_running_loop().time() < deadline, "Record was not archived"
        await asyncio.sleep(1)


def are_chronolog_processes_running():
    """
    Check if all required ChronoLog processes are running.

    Returns:
        bool: True if all processes are running, False otherwise
    """
    required_processes = [
        "chronovisor_server",
        "chrono_grapher",
        "chrono_keeper",
        "chrono_player",
    ]

    try:
        # Use pgrep to find processes matching any of the required names
        result = subprocess.run(
            [
                "pgrep",
                "-laf",
                "chronovisor_server|chrono_grapher|chrono_keeper|chrono_player",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            return False

        found_processes = result.stdout.strip()

        # Check that we found at least one process for each required type
        for process in required_processes:
            if process not in found_processes:
                return False

        # If we get here, all processes are running
        return len(found_processes) > 0

    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
