# Action: status

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
Display a concise dashboard of the current materio research project. No subagents; reads files directly and formats output.
</objective>

<process>

## 1. Check Initialization

```bash
[ ! -f .research/RESEARCH.md ] && echo "No research project initialized. Run identify-research using the materio-research-explorer skill to start." && exit 0
```

## 2. Read All State Files

```bash
cat .research/RESEARCH.md
cat .research/LITERATURE.md 2>/dev/null | head -30
cat .research/WORKFLOW.md 2>/dev/null
cat .research/STATE.md 2>/dev/null
cat .research/DATA-INDEX.md 2>/dev/null | head -20
ls .research/tasks/ 2>/dev/null
ls .research/data/ 2>/dev/null
ls .planning/ 2>/dev/null | head -5
ls .research/handoff/ 2>/dev/null
ls -1t .research/checkpoints/ 2>/dev/null | head -3
```

## 3. Display Dashboard

Format and display:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Materio RESEARCH STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RESEARCH IDENTITY
  Prompt:   [research prompt from RESEARCH.md]
  Domain:   [domain + sub-field]
  Stage:    [career stage]

LITERATURE
  Status:   [LITERATURE.md exists? complete | pending]
  Gaps:     [N gaps identified]
  Keywords: [primary keywords]

VIRTUAL LAB
  Status:   [VIRTUAL-LAB.md exists? defined | not defined]
  Equipment:[N items] | Compute: [HPC systems] | Gaps: [N flagged]

WORKFLOW  ([N]/[total] tasks complete)
  Task 01:  [name]; [☑ complete | ☑ in-progress | ☐ pending | ☐ archived]
  Task 02:  [name]; [status]
  ...

DATA
  [N] files registered (.research/DATA-INDEX.md)
  [N] files in .research/data/

PAPER (wtf-p)
  [.planning/project.json → "wtf-p 0.6 project initialized" |
   other existing .planning files → "existing paper state; inspect compatibility" |
   .research/handoff/ → "Handoff written; paste into wtf-p" |
   "Not started; use Run wtfp using the materio-research-explorer skill"]

───────────────────────────────────────────
▶ SUGGESTED NEXT ACTION
  [smart routing based on state]
───────────────────────────────────────────
```

## 4. Smart Next Action

Based on state, suggest:
- No RESEARCH.md → `Run identify-research using the materio-research-explorer skill`
- No LITERATURE.md → `Run literature-review using the materio-literature-reviewer skill`
- No VIRTUAL-LAB.md → `Run define-virtual-lab using the materio-lab-definer skill`
- No WORKFLOW.md → `Run define-research-tasks using the materio-workflow-planner skill`
- Tasks pending → `Run execute-task using the materio-task-executor skill [next pending N]`
- All tasks complete, no paper → `Run wtfp using the materio-research-explorer skill`
- All tasks complete, paper in progress → `wtf-p action progress`

</process>
