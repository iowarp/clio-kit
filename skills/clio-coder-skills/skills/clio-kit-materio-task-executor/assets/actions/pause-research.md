# Action: pause-research

This guide is linked from an installed portable skill. Action requests use the
actual installed skill names; this standard package registers no slash commands.
The interactive assistant applies a role inline or assigns its skill to a generic
worker if the host supports delegation. No named worker registration is presumed.
If a native peer package supplies command or agent bindings, follow its binding
section below while preserving the same researcher gates and return checkpoints.

This guide is linked from an installed portable skill. Determine the package root
from that skill's directory. Before executing a shell example, bind PACKAGE_ROOT
to that actual root in the same command. Treat $ARGUMENTS as validated user input;
these are examples to fill with actual paths, not automatic host substitutions.



Use the host's file reading/editing and local command execution capabilities. Read
assets/references/research-policy.md, resolved from this package. If this task
needs researcher answers, the interactive assistant asks them through the host's
conversation facility before gated writes. Workers return questions with their
checkpoint kind. Use the named role's portable skill for the scientific work;
if the host does not support delegation, the interactive assistant performs that
role with the same input/output and decision boundaries. Do not invent worker,
search, fleet, or interactive capabilities. Inspect actual files before completion.
Use package scripts by resolving their links from the installed skill directory.
For an optional commit, set CHANGED_FILES to only the named files actually changed
and read back during this action. The research-state helper rejects directory staging.. Inspect WORKFLOW.md, STATE.md, and
actual partial task files. Record the active task, completed preparation, pending
physical work, decisions, missing inputs, and exact next action. Mark the task
paused without changing its identity or claiming unfinished results. Confirm any
missing continuation details with the host conversation facility. Read the updated documents back.
Run `python3 "${PACKAGE_ROOT}/assets/scripts/research_state.py" checkpoint save paused` and
verify its receipt. Offer resume-research and show the saved continuation. If
optional recording is enabled, pass only the changed named files to record.
