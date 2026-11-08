# capabilities/stop_chronolog.py

from fastmcp.exceptions import ToolError

from chronomcp.utils import config

# chronolog::ClientErrorCode (Client/cpp/include/client_errcode.h)
CL_ERR_NOT_ACQUIRED = -5


async def stop_chronolog() -> str:
    """Release the story and disconnect from ChronoLog."""
    if config._story_handle is None:
        raise ToolError("No active ChronoLog session to stop.")
    client = config.get_client()

    ret = client.ReleaseStory(config._active_chronicle, config._active_story)
    # Already released is still "stopped"; failing here would leave the session
    # state set with no way to clear it.
    if ret not in (0, CL_ERR_NOT_ACQUIRED):
        raise ToolError(f"Failed to release story '{config._active_story}': {ret}")

    ret = client.Disconnect()
    if ret != 0:
        raise ToolError(f"Failed to disconnect from ChronoLog: {ret}")

    config._active_chronicle = None
    config._active_story = None
    config._story_handle = None

    return "ChronoLog session stopped and disconnected"
