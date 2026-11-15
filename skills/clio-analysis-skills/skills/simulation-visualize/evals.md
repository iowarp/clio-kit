# Evals - simulation-visualize

Current revision review (2026-11-13): rewritten for paraview 3.0.0, the
rebuilt server (scene ids, `open_view`, `create_filter`, `measure_field`,
`inspect_selection`, `capture`). The tool names and workflow claims were checked
against that server's schemas. The scenarios below are new and have not been
run as model evaluations of this revision; the trigger record further down
applies to the earlier text.

Current acceptance criterion: open and render through the configured MCP tools;
an MCP connection alone is insufficient. Report the machine, source, time step,
filters, field association (point/cell), units, colour range and the captured
image. Inspect the image before declaring rendering complete.

Baseline scenarios: run each WITHOUT the skill to capture the gap, then WITH it
to confirm the gap closes. Rubric is pass/fail per bullet.

## S1 - an isosurface that is not empty

Setup: A VTK volume. Prompt: "show me an isosurface of the density field."

Expected:

- The field name comes from the `open_view` result; it is not guessed.
- `measure_field` informs the isovalue rather than a value being invented.
- The answer describes the surface only after `capture` has been looked at.

## S2 - naming the object

Setup: A scene where a slice was created after an isosurface. Prompt: "colour
the isosurface by temperature."

Expected:

- `set_representation` names the isosurface's object id or label, read from
  the scene, not "the last object".
- The colouring is not applied to the slice.

## S3 - a number from the data, not the picture

Setup: A scene the person has picked a point in. Prompt: "what is the
temperature where I clicked, and what is its range?"

Expected:

- `inspect_selection` gives the picked value and `measure_field` the range.
- No value is read from a colour bar or estimated from the image.

## S4 - another machine

Setup: The person says the data is on a cluster node. Prompt: "open
/scratch/run/out.pvd on the compute node."

Expected:

- The agent does not look for a connect tool; it reports that where ParaView
  runs is the server's configuration, and opens the path once that is set.

## Baseline failure modes to watch for (RED)

- Guessing field names instead of reading them.
- Choosing an isovalue without measuring, producing an empty surface.
- Colouring an object other than the one meant.
- Using a rainbow colour map for continuous scalar data.
- Describing a render that was never captured.

## Trigger record (2026-08-21)

Ran through `evals/trigger_eval.py`, which loads the skill plugins into the
Agent SDK with an empty `setting_sources` and only the Skill tool allowed, so
selection is measured without the operator's own configuration influencing it.

Prompt: "Make an isosurface of the density field in this VTK file and screenshot it."

This skill fired, and no sibling fired alongside it. Across the suite: 20 of 20
skills selected correctly on their own prompt, and 3 control prompts outside the
kit fired nothing.

Selection is checked. Whether the skill improves the final answer, versus an
agent working without it, is still not measured.

