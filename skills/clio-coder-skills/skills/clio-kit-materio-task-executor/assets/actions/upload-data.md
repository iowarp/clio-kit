# Action: upload-data

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
Register data files (experimental results, papers, datasets, scripts) into the .research/data/ index so that task executors can find and use them. Does not move files; records paths and metadata.

**Orchestrator role:** Accept file paths or scan for files, gather metadata via the host conversation facility, write/update .research/DATA-INDEX.md.
</objective>

<context>
Optional file path: $ARGUMENTS
</context>

<process>

## 1. Find Files to Register

If $ARGUMENTS provided:
```bash
[ -f "$ARGUMENTS" ] && echo "File exists: $ARGUMENTS"
[ -d "$ARGUMENTS" ] && ls "$ARGUMENTS"
```

If no argument: scan for common data file types:
```bash
find . -name "*.csv" -o -name "*.xlsx" -o -name "*.txt" -o -name "*.pdf" \
       -o -name "*.dat" -o -name "*.json" -o -name "*.mat" -o -name "*.bib" \
       2>/dev/null | grep -v ".research" | grep -v ".claude" | grep -v ".git" | head -20
```

## 2. For Each File, Gather Metadata

Use the host conversation facility (batch for multiple files):
- header: "Data Registration"
- question: "Found these files:\n[list]\n\nFor each, tell me:\n1. **Type**: experimental-data | paper | dataset | script | model-output | other\n2. **Description**: What does it contain? (1 sentence)\n3. **Relevant tasks**: Which workflow tasks use this data?\n4. **Format notes**: Units, column headers, any preprocessing needed?"
- options: "I'll describe them" | "Register all as-is with auto-detection"

## 3. Copy to .research/data/ (optional)

Use the host conversation facility:
- header: "Copy Files?"
- question: "Copy files into .research/data/ for centralized storage, or just index their current locations?"
- options: "Copy into .research/data/" | "Index in-place (keep original location)"

If copy:
```bash
python3 "${PACKAGE_ROOT}/assets/scripts/research_state.py" copy-data "$FILE"
```

## 4. Update DATA-INDEX.md

Write or append to `.research/DATA-INDEX.md`:

```markdown
## [filename]
- **Path**: [full path]
- **Type**: [type]
- **Description**: [description]
- **Relevant tasks**: Task [N], Task [M]
- **Format**: [format notes]
- **Registered**: [date]
```

## 5. Record

Only if `commit_research` is true in `.research/config.json`:
```bash
python3 "${PACKAGE_ROOT}/assets/scripts/research_state.py" record --message "data: register [N] files; [brief description]" --files "${CHANGED_FILES[@]}"
```

</process>

<offer_next>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Materio ► DATA REGISTERED ✓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Index: .research/DATA-INDEX.md
Files: [N] registered

───────────────────────────────────────────

Data is now available to the literature reviewer and task executors.
Papers and .bib files feed `Run literature-review using the materio-literature-reviewer skill`; datasets feed `Run execute-task using the materio-task-executor skill [N]`.

───────────────────────────────────────────

</offer_next>

<success_criteria>
- [ ] All provided files located and verified
- [ ] Type, description, and relevant tasks captured per file
- [ ] DATA-INDEX.md written/updated
- [ ] Files optionally copied to .research/data/
- [ ] Committed only if commit_research is true
</success_criteria>
