"""Utility functions for ChronoLog tests."""

import subprocess


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
