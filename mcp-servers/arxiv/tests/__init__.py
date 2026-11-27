"""Shared test helpers."""


async def settle(call):
    """Await a handler call; report a raised error as ``{"isError": True}``.

    Handlers raise on failure (the server turns that into a real MCP error).
    The tolerance tests only need "returned a result or failed cleanly".
    """
    try:
        return await call
    except Exception as exc:
        return {"isError": True, "error": str(exc)}
