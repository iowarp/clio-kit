"""Review report paths explicitly named in a Codex edit; never scan the project."""

import json
from pathlib import Path
import re
import subprocess
import sys


def main():
    event = json.load(sys.stdin)
    arguments = event.get("tool_input", {})
    if isinstance(arguments, str):
        patch = arguments
        paths = []
    else:
        patch = arguments.get(
            "patch", arguments.get("input", arguments.get("command", ""))
        )
        paths = [arguments.get("file_path", "")]
    paths.extend(re.findall(r"^\*\*\* (?:Add|Update) File: (.+)$", patch, re.MULTILINE))
    verifier = (
        Path(__file__).parents[1] / "skills/dataset-report/scripts/verify_report.py"
    )
    messages = []
    for path in dict.fromkeys(paths):
        if not path:
            continue
        translated = {**event, "tool_input": {"file_path": path}}
        result = subprocess.run(
            [sys.executable, str(verifier), "hook"],
            input=json.dumps(translated),
            text=True,
            capture_output=True,
            check=True,
            timeout=25,
        )
        if result.stdout.strip():
            messages.append(
                json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
            )
    if messages:
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PostToolUse",
                        "additionalContext": "\n".join(messages),
                    }
                }
            )
        )


if __name__ == "__main__":
    main()
