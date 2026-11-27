"""Parse bare or bracketed log levels consistently across log operations."""


def parse_level_message(remainder: str) -> tuple[str, str]:
    # A timestamp may itself have been enclosed in brackets.
    parts = remainder.lstrip().removeprefix("]").lstrip().split(maxsplit=1)
    level = parts[0] if parts else ""
    if level.startswith("[") and level.endswith("]"):
        level = level[1:-1]
    return level.upper(), parts[1] if len(parts) > 1 else ""
