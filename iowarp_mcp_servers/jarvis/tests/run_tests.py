#!/usr/bin/env python3
"""
Simple test runner script for Jarvis MCP tests using uv.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_file=None, verbose=False, coverage=False, markers=None):
    """Run the specified tests using uv."""

    # Get the test directory
    test_dir = Path(__file__).parent
    project_root = test_dir.parent

    # Build uv run pytest command
    cmd = ["uv", "run", "pytest"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(
            ["--cov=src", "--cov-report=term-missing", "--cov-report=html:htmlcov"]
        )

    # Add marker selection
    if markers:
        cmd.extend(["-m", markers])

    # Add specific test file or run all
    if test_file:
        test_path = test_dir / test_file
        if test_path.exists():
            cmd.append(str(test_path))
        else:
            print(f"Test file not found: {test_file}")
            return 1
    else:
        cmd.append(str(test_dir))

    print(f"Running: {' '.join(cmd)}")

    # Run the tests
    try:
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode
    except FileNotFoundError:
        print("Error: uv not found. Install it with:")
        print("  curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("Or run tests directly with: python -m pytest tests/")
        return 1


def run_simple_tests(test_file=None, verbose=False):
    """Run tests without coverage using simple pytest."""

    # Get the test directory
    test_dir = Path(__file__).parent
    project_root = test_dir.parent

    # Build simple pytest command
    cmd = ["python", "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    # Add specific test file or run all
    if test_file:
        test_path = test_dir / test_file
        if test_path.exists():
            cmd.append(str(test_path))
        else:
            print(f"Test file not found: {test_file}")
            return 1
    else:
        cmd.append(str(test_dir))

    print(f"Running: {' '.join(cmd)}")

    # Run the tests
    try:
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode
    except FileNotFoundError:
        print("Error: pytest not found. Try installing with uv:")
        print("  uv add --dev pytest pytest-asyncio")
        return 1


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="Run Jarvis MCP tests")

    parser.add_argument(
        "test_file", nargs="?", help="Specific test file to run (e.g., test_basic.py)"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Run tests in verbose mode"
    )

    parser.add_argument(
        "--coverage",
        "-c",
        action="store_true",
        help="Run tests with coverage reporting (requires uv)",
    )

    parser.add_argument(
        "--markers",
        "-m",
        help="Run only tests with specific markers (e.g., 'unit', 'not slow')",
    )

    parser.add_argument(
        "--simple", action="store_true", help="Run tests with simple pytest (no uv)"
    )

    parser.add_argument(
        "--list-tests", action="store_true", help="List available test files"
    )

    args = parser.parse_args()

    if args.list_tests:
        test_dir = Path(__file__).parent
        test_files = list(test_dir.glob("test_*.py"))
        print("Available test files:")
        for test_file in sorted(test_files):
            print(f"  {test_file.name}")
        return 0

    # Run the tests
    if args.simple or not args.coverage:
        return run_simple_tests(test_file=args.test_file, verbose=args.verbose)
    else:
        return run_tests(
            test_file=args.test_file,
            verbose=args.verbose,
            coverage=args.coverage,
            markers=args.markers,
        )


if __name__ == "__main__":
    sys.exit(main())
