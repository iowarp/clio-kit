# Evals - slicing

Recorded from live CLIO runs of this skill with the slicer server 1.0.0 and
PrusaSlicer 2.7.2 (the clio-agent apps ledger, "Slicer plugin (SL)", SL-Q2 to
SL-Q7), when the skill shipped as a marketplace plugin; the text moved here
unchanged apart from host-neutral wording. The scenarios below have not been
run as Codex evaluations. The Codex case `slicing` (`evals/codex_cases.py`) is
S3, which needs no slicer on the evaluation machine.

Acceptance: the numbers in the answer come from the tools (`slice`,
`layer_stats`), the presets used are named, and a layer is chosen from the
per-layer rows before the view is moved. A slice result alone does not show
that the right layer was found.

Baseline scenarios: run each WITHOUT the skill to capture the gap, then WITH it
to confirm the gap closes. Rubric is pass/fail per bullet.

## S1 - slice with plain words

Setup: A STEP part in the working folder. Prompt: "Slice the part for a Prusa MK4
in PLA at 0.2 mm and show me the layers."

Expected:

- One `slice` call with the printer, material and quality as said; no preset
  names guessed.
- The answer names the printer, print and material presets used and how the
  words matched, and quotes the time and filament as the slicer's estimates.
- The slicer's warnings are passed on in words.

Without the skill: the presets are not named, or a preset name is invented and
refused.

## S2 - find a layer by what it prints

Setup: A slice of a part with an internal channel. Prompt: "Show me the layer
where the channel roof starts."

Expected:

- `layer_stats` rows are read first; the layer is the first with bridge infill
  over the channel, matched by the rows' `extents`.
- An open view is moved with its `go_to_layer`; a second view is not opened.
- The answer names the layer and its Z, and says when it is unsure between spans.

Without the skill: a layer is guessed from the height, or a new view is opened
for every question.

## S3 - no slicer installed

Setup: No PrusaSlicer or OrcaSlicer on PATH, `SLICER_BIN` unset. Prompt: "Slice
the part and tell me how long it takes to print."

Expected:

- The `slicer_missing` answer is passed on: what was looked for, how to install a
  slicer or set `SLICER_BIN`.
- No other slicing route is tried and no time is invented.

Without the skill: the agent looks for another slicer or estimates a time.

## S4 - compare choices

Setup: A part with an overhang. Prompt: "Does it need supports? How much longer is it
with them?"

Expected:

- Two slices, with and without `support_material`, compared with
  `compare_slices`.
- The answer quotes both times and filaments and the slicer's support warning.

Without the skill: one slice, and the difference is estimated in words.
