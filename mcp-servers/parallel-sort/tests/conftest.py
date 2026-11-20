"""Shared fixtures for the parallel-sort server tests."""

import os

import pytest


@pytest.fixture(autouse=True)
def _allow_every_folder(monkeypatch):
    """Tool tests use temporary folders; test_path_policy.py covers the limits."""
    monkeypatch.setenv("PARALLEL_SORT_MCP_ALLOWED_ROOTS", os.sep)
