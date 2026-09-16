# Action: settings

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
and read back during this action. The research-state helper rejects directory staging.. Run
`python3 "${PACKAGE_ROOT}/assets/scripts/research_state.py" config` and display effective values:
web_search (default false, controls provided-URL fetching and Crossref queries),
auto_checkpoint (default true), and commit_research (default false).
Model and target selection belongs to the host's configured targets; no plugin setting
silently changes host models. Supplied data lives at `.research/data/`.

Ask with the host conversation facility which settings to change. Preserve unrelated existing config
fields. Write only actual boolean values in .research/config.json, then run the
config helper again and read the file back. Malformed configuration needs repair
before dependent work. An existing authorization remains valid; do not repeatedly
ask to reconfirm the same choice. Report the effective values and implications.
