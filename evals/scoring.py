"""Decide whether a recorded tool result reports a failure.

Separate from the runner so it can be tested without the agent SDK, and
because getting it wrong is not a cosmetic problem: this number is what the
report presents as evidence about the servers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def call_failed(is_error: bool, text: str) -> bool:
    """Whether a tool result reports a failure, judged structurally.

    This replaced a scan for substrings like "error", "not found" and "could
    not" anywhere in the payload, which cannot tell a server reporting an error
    from a server *reporting on* errors: a log analyser handed a log full of
    ERROR lines returned a perfect result containing every word the scan looked
    for, and was recorded as a failure. A server's success is not decided by
    its output's vocabulary.
    """
    if is_error:
        return True
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            if payload.get("success") is False or payload.get("error"):
                return True
            # Some servers report failure inside an otherwise ordinary field
            # rather than by an error key, e.g. {"result": "Error: ..."}.
            return any(
                isinstance(value, str) and value.lstrip().lower().startswith("error")
                for value in payload.values()
            )
    return stripped.lower().startswith("error")


# Where the run happened. Everything below it is addressed relative to the
# repository so a recorded result is the same on any machine.
REPO_ROOT = str(Path(__file__).resolve().parents[1])

# An absolute path rooted in someone's home directory, taken whole.
_OUTSIDE_PATH = r"/(?:home|Users)/[^\s\"',)\]}]*"


def redact(text: str) -> str:
    """Replace this checkout's absolute path, and any home directory, with a marker.

    Recorded results are committed, so an absolute path would publish the
    username, institution and directory layout of whoever ran the evaluation,
    and would make the same run on two machines produce a spurious diff.
    """
    text = text.replace(REPO_ROOT, "<repo>")
    # A path inside the checkout keeps its useful tail; one outside it is
    # collapsed whole, because the directory names above a home directory carry
    # no analytical value and can name a person or an institution.
    text = re.sub(_OUTSIDE_PATH, "<home>", text)
    # Idempotent: also collapses a path left half-redacted by an earlier rule.
    return re.sub(r"<home>[^\s\"',)\]}]*", "<home>", text)


def anonymise(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Redact every recorded tool call's input and output in place."""

    def redact_value(value: Any) -> Any:
        if isinstance(value, str):
            return redact(value)
        if isinstance(value, list):
            return [redact_value(item) for item in value]
        if isinstance(value, dict):
            return {key: redact_value(item) for key, item in value.items()}
        return value

    for call in calls:
        call["input"] = redact_value(call["input"])
        call["output"] = redact(str(call.get("output", "")))
    return calls
