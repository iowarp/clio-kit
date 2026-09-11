# capabilities/retrieve_interaction.py

import re
import json
import asyncio
from pathlib import Path
import tempfile
from chronomcp.utils import config, helpers


async def retrieve_interaction(
    chronicle_name: str | None = None,
    story_name: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> str:
    chronicle = chronicle_name or config.DEFAULT_CHRONICLE
    story = story_name or config.DEFAULT_STORY

    cmd = [
        "stdbuf",
        "-o0",
        config.READER_BINARY,
        "-c",
        config.CONFIG_FILE,
        "-C",
        chronicle,
        "-S",
        story,
    ]
    if start_time:
        st_ns = helpers.parse_time_arg(start_time, is_end=False)
        cmd += ["-st", st_ns]
    if end_time:
        et_ns = helpers.parse_time_arg(end_time, is_end=True)
        cmd += ["-et", et_ns]

    out, err = await asyncio.to_thread(helpers.run_reader, cmd)

    records = [
        json.loads(line.removeprefix("CLIO_RECORD_JSON "))
        for line in out.splitlines()
        if line.startswith("CLIO_RECORD_JSON ")
    ]
    if not records:
        # Support older site-provided reader binaries.
        records = re.findall(r'record="([^"]*)"', out)
    if not records:
        return "No records found."

    # Chronicle/story identifiers are data, never output path components.
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix="chronolog-records-",
        suffix=".txt",
        dir=Path.cwd(),
        delete=False,
    ) as f:
        filename = f.name
        f.write("\n".join(records))
    return filename
