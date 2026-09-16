"""Tests for Chronolog retrieve interaction capabilities."""

import pytest
import time
import random

try:
    from chronomcp.capabilities.retrieve_handler import retrieve_interaction
    from chronomcp.capabilities.record_handler import record_interaction
    from chronomcp.capabilities.start_handler import start_chronolog
    from chronomcp.capabilities.stop_handler import stop_chronolog

    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False

from .test_utils import are_chronolog_processes_running, wait_for_archived_record

pytestmark = pytest.mark.skipif(
    not HAS_DEPENDENCIES,
    reason="ChronoLog system dependencies not available",
)


class TestRetrieveHandler:
    """Test retrieve interaction functionality"""

    @pytest.mark.asyncio
    async def test_retrieve_empty_interaction(self):
        """Test basic retrieve functionality"""
        if not are_chronolog_processes_running():
            pytest.skip("ChronoLog processes are not running")

        chronicle_name = (
            f"test_chronicle_{int(time.time())}_{random.randint(1000, 9999)}"
        )
        story_name = f"test_story_{int(time.time())}_{random.randint(1000, 9999)}"
        result = await retrieve_interaction(chronicle_name, story_name)
        assert isinstance(result, str)
        assert result == "No records found."

    @pytest.mark.asyncio
    async def test_retrieve_after_record(self):
        """Test retrieving interaction after recording one"""
        if not are_chronolog_processes_running():
            pytest.skip("ChronoLog processes are not running")

        chronicle_name = (
            f"test_chronicle_{int(time.time())}_{random.randint(1000, 9999)}"
        )
        story_name = f"test_story_{int(time.time())}_{random.randint(1000, 9999)}"

        # Start a new session
        start_result = await start_chronolog(chronicle_name, story_name)
        assert isinstance(start_result, str)
        assert "ChronoLog session started" in start_result

        try:
            record_result = await record_interaction("Test question", "Test answer")
            assert "Interaction recorded to ChronoLog" in record_result
        finally:
            await stop_chronolog()
        await wait_for_archived_record(
            chronicle_name,
            story_name,
            "user: Test question, assistant: Test answer",
        )
