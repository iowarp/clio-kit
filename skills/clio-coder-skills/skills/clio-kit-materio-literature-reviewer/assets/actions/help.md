# Action: help

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



<objective>
Display the Materio action reference with the actual skill invocation vocabulary. No tools needed.
</objective>

<process>

Display the following:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Materio; materials research system
 Plan and automate your research with your coding agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CORE RESEARCH LOOP
─────────────────────────────────────────
 1. Run identify-research using the materio-research-explorer skill
       Guided interview → select domain, sub-field,
       scope, and confirm a research prompt

 2. Run literature-review using the materio-literature-reviewer skill
       Review literature, identify gaps, refine keywords
       (loops back to identify-research if needed)

 3. Run define-virtual-lab using the materio-lab-definer skill
       Map available resources: experimental equipment,
       HPC systems, software licenses, collaborations
       Questions are targeted to your research domain

 4. Run define-research-tasks using the materio-workflow-planner skill
       Choose from traditional research workflows or
       build from scratch; interview per task for assumptions
       (resource-aware: flags tasks requiring unavailable equipment)

 5. Run execute-task using the materio-task-executor skill [N | all]
       Execute a specific task or the full workflow
       Types: literature, experimental, computational,
              data-analysis, analytical, writing

 6. Run wtfp using the materio-research-explorer skill
       Bridge to wtf-p; translate research into a paper
       project with include/exclude control

TASK MANAGEMENT
─────────────────────────────────────────
 Run add-task using the materio-research-explorer skill          Add a task to the workflow
 Run remove-task using the materio-research-explorer skill [N]   Remove a task permanently
 Run archive-task using the materio-research-explorer skill [N]  Archive (preserve but deactivate)

DATA
─────────────────────────────────────────
 Run upload-data using the materio-research-explorer skill [file] Register data files, papers,
                           and datasets for task use

PROGRESS & CONTROL
─────────────────────────────────────────
 Run status using the materio-research-explorer skill             Full project dashboard
 Run progress using the materio-research-explorer skill           Statusline + smart routing
 Run pause-research using the materio-research-explorer skill     Pause + auto-checkpoint
 Run resume-research using the materio-research-explorer skill    Resume from paused state
 Run checkpoint using the materio-research-explorer skill save [label]   Save state snapshot
 Run checkpoint using the materio-research-explorer skill restore [name]  Restore snapshot
 Run checkpoint using the materio-research-explorer skill list           List snapshots

META
─────────────────────────────────────────
 Run help using the materio-research-explorer skill               This help
 Run settings using the materio-research-explorer skill           View/edit project config

PROJECT STATE (.research/)
─────────────────────────────────────────
 RESEARCH.md     Research identity, prompt, scope
 LITERATURE.md   Review results, gaps, keywords
 VIRTUAL-LAB.md  Equipment, HPC, software, gaps
 WORKFLOW.md     Tasks with status and assumptions
 STATE.md        Current position, phase, decisions
 DATA-INDEX.md   Registered data files
 config.json     Settings (web_search, commit_research…)
 data/           Uploaded datasets and papers
 tasks/          Task-by-task outputs (task-NN/)
 checkpoints/    State snapshot archives
 handoff/        Paste-ready blocks for wtf-p 0.6

TASK TYPES
─────────────────────────────────────────
 literature    → Read supplied papers and provided URLs, synthesize
 experimental  → Generate protocol + data templates
 computational → Generate simulation/analysis scripts
 data-analysis → Analyze uploaded data, produce figures
 analytical    → Mathematical modeling, derivations
 writing       → Bridge to Run wtfp using the materio-research-explorer skill

GUARDRAILS (after each task, advisory)
─────────────────────────────────────────
 check_physics     impossible values (T < 0 K, ρ ≤ 0…)
 verify_citations  Crossref check on quoted titles/DOIs
 check_scripts     syntax + hallucinated imports
 Findings are shown to you; nothing is auto-removed.

INTEGRATION WITH WTF-P
─────────────────────────────────────────
 Run wtfp using the materio-research-explorer skill prepares reviewed import briefs.
 0.6+: writes .research/handoff/ blocks you paste into
       wtf-p action new-paper → map-project → create-outline
 Existing paper state: inspect/reuse/repair first.
 The bridge never writes paper state. Reuse answers already
 answered during research planning.

TIPS
─────────────────────────────────────────
 • Each dispatch receives fresh worker context
 • Interviews happen in the command; agents only
   synthesize and write, and report back checkpoints
 • Run upload-data using the materio-research-explorer skill papers BEFORE literature-review:
   supplied papers first, provided URLs second
 • Materio never initializes git. Set commit_research=true in
   Run settings using the materio-research-explorer skill if you want optional named-file commits. Checkpoints use archives.
 • Run checkpoint using the materio-research-explorer skill save before long tasks
 • Run literature-review using the materio-literature-reviewer skill can loop back to
   identify-research if your prompt needs revision
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

</process>
