"""Composition root for the ``clio-kit`` console script.

``src/clio_kit/__init__.py`` defines the launcher's ``main`` click group but
is already a ratcheted, oversized module
(``scripts/check_file_size.py``'s ``RATCHET_BASELINE`` records its exact
line count; the diff guard in ``scripts/check_baseline_diff.py`` rejects any
same-PR increase to that number). A new command surface attaches to ``main``
from its own owner module and is wired in HERE instead, at import time, so
shipping it never has to grow ``__init__.py``.

Attaching at module scope (not inside :func:`cli`) means importing this
module is itself what registers the commands -- ``clio-kit``'s installed
console script does that naturally (``pyproject.toml``'s
``[project.scripts]`` points at ``clio_kit.cli_entry:cli``, not
``clio_kit:cli``), and a test can do the same with a plain
``import clio_kit.cli_entry`` before driving ``clio_kit.main`` with
``click.testing.CliRunner`` -- no subprocess required.
"""

from __future__ import annotations

from clio_kit import main
from clio_kit.skills import SKILL_COMMANDS

for _command in SKILL_COMMANDS:
    main.add_command(_command)


def cli() -> None:
    """Run the launcher CLI (every extension command above is already attached)."""
    main()


__all__ = ["cli"]
