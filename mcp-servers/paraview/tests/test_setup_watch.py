"""Setup steps (probes, the rendering check, installs) have no limit on their total
time: one runs while it prints, is stopped after printing nothing for the stall
window, and stops when the person cancels."""

from __future__ import annotations

import sys
import threading
import time

import pytest

from paraview_mcp.attach import Attacher
from paraview_mcp.config import Settings, Target
from paraview_mcp.errors import ParaViewError
from paraview_mcp.transport import LocalTransport, Watch, watched


def _python(code: str) -> list[str]:
    return [sys.executable, "-u", "-c", code]


def test_a_step_that_keeps_printing_runs_past_the_stall_window() -> None:
    told: list[str] = []
    began = time.monotonic()
    done = watched(
        _python(
            "import time\nfor i in range(6):\n    print('line', i, flush=True)\n"
            "    time.sleep(0.5)\nprint('CLIO_DONE=ok')"
        ),
        Watch(stall=1.5, progress=told.append),
        "A long install",
    )
    assert time.monotonic() - began > 2 * 1.5
    assert done.code == 0 and "CLIO_DONE=ok" in done.stdout
    assert told and told[0].startswith("A long install, ")
    assert any("line" in note for note in told)


def test_a_step_that_prints_nothing_for_the_stall_window_is_stopped() -> None:
    began = time.monotonic()
    with pytest.raises(ParaViewError) as caught:
        watched(
            _python("print('started', flush=True)\nimport time\ntime.sleep(60)"),
            Watch(stall=1.5),
            "A rendering check",
        )
    assert caught.value.code == "setup_stalled"
    assert "started" in caught.value.message
    assert time.monotonic() - began < 20


def test_a_cancel_stops_a_step_at_once() -> None:
    stop = threading.Event()
    threading.Timer(1.0, stop.set).start()
    began = time.monotonic()
    with pytest.raises(ParaViewError) as caught:
        watched(
            _python(
                "import time\nwhile True:\n    print('x', flush=True)\n"
                "    time.sleep(0.2)"
            ),
            Watch(stall=60, cancelled=stop.is_set),
            "A build",
        )
    assert caught.value.code == "cancelled"
    assert time.monotonic() - began < 10


def test_a_step_reads_its_stdin_and_reports_its_exit_code() -> None:
    done = watched(
        _python("import sys\ndata=sys.stdin.read()\nprint(len(data))\nsys.exit(3)"),
        Watch(stall=30),
        "A copy",
        stdin=b"x" * 300_000,
    )
    assert done.code == 3 and done.stdout.strip() == "300000"


def test_the_attacher_watches_setup_with_the_calls_cancel_and_progress() -> None:
    attacher = Attacher(Settings(target=Target(), setup_stall=42))
    stop = threading.Event()
    heard: list[str] = []
    attacher._call = (stop, heard.append)
    watch = attacher._watch()
    assert watch.stall == 42 and not watch.cancelled()
    assert watch.progress is not None
    watch.progress("Checking offscreen rendering, 3 s")
    assert heard == ["Checking offscreen rendering, 3 s"]
    stop.set()
    assert watch.cancelled()
    attacher._call = (None, None)
    attacher.cancel.set()
    assert watch.cancelled()


def test_a_local_step_runs_through_the_watch() -> None:
    transport = LocalTransport(Target())
    stop = threading.Event()
    stop.set()
    transport.watch = Watch(stall=60, cancelled=stop.is_set)
    with pytest.raises(ParaViewError) as caught:
        transport.run("sleep 30", label="Checking the ParaView folder")
    assert caught.value.code == "cancelled"
    assert "Checking the ParaView folder on this machine" in caught.value.message
