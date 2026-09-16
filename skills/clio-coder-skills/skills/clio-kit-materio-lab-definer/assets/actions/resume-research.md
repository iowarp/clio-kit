# Action: resume-research

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
and read back during this action. The research-state helper rejects directory staging.


<objective>
Load research state and resume from the last paused or in-progress task. Equivalent to wtf-p action progress but triggers active resumption.
</objective>

<process>

## 1. Read State

```bash
cat .research/STATE.md 2>/dev/null
cat .research/WORKFLOW.md 2>/dev/null
```

## 2. Find Paused or Next Pending Task

Look for tasks with status `☐ paused` first, then `☐ pending`.

Change `☐ paused` back to `☑ in-progress` if found.

## 3. Show Resumption Context

Display:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Materio ► RESUMING RESEARCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Research:   [prompt]
Resuming:   Task [N]; [name]
Last saved: [timestamp from STATE.md]

Task context:
[full task block from WORKFLOW.md]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## 4. Update STATE.md

Set status = "active", record the resume timestamp. Only if `commit_research` is true in `.research/config.json`:
```bash
python3 "${PACKAGE_ROOT}/assets/scripts/research_state.py" record --message "research: resume at task [N]; [task name]" --files "${CHANGED_FILES[@]}"
```

## 5. Offer Next Action

Suggest: `Run execute-task using the materio-task-executor skill [N]` to continue, or `Run progress using the materio-research-explorer skill` for full overview.

</process>

<success_criteria>
- [ ] Paused task identified and status restored to in-progress
- [ ] Full task context shown to orient the user
- [ ] STATE.md updated
- [ ] Next command clearly offered
</success_criteria>
