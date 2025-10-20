"""Tests for Chronolog stop session capabilities."""

import pytest
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chronomcp.capabilities.stop_handler import stop_chronolog
from .test_utils import are_chronolog_processes_running


class TestStopHandler:
    """Test ChronoLog session stop functionality"""

    @pytest.mark.asyncio
    async def test_stop_chronolog_basic(self):
        """Test basic stop functionality"""
        if not are_chronolog_processes_running():
            pytest.skip("ChronoLog processes are not running")

        result = await stop_chronolog()
        assert isinstance(result, str)
        assert "ChronoLog session stopped" in result
