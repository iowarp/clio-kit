# Evals - working-with-coordinate-systems

Current revision review (2026-09-10): tool names and workflow claims were checked
against the shipped server schemas and implementation. Historical records below
apply to earlier text; they are not fresh model evaluations of this revision.

Current acceptance criterion: State source and target CRS, axis order, horizontal units and vertical datum. If any is unknown, identify the missing metadata rather than guessing a transformation. Check a known control point and antimeridian behavior where relevant.

Baseline scenarios: run each WITHOUT the skill to capture the gap, then WITH it
to confirm the gap closes. Rubric is pass/fail per bullet.

## S1 - axis order

Setup: Prompt: "build a GeoJSON point for Chicago at 41.88, -87.63."

Expected:

- The position is written `[-87.63, 41.88]`.
- The answer states GeoJSON is [longitude, latitude] and that the given pair was
  in the opposite order.

## S2 - degrees are not metres

Setup: Prompt: "these two points are 0.5 degrees apart, how far is that?"

Expected:

- The answer refuses a single conversion, explaining that a degree of longitude
  varies with latitude while a degree of latitude does not.
- Great-circle distance via `filter_points_by_radius` is named as the correct
  route.

## S3 - a silent mismatch

Setup: Two datasets, one in lon/lat and one in UTM metres. Prompt: "overlay
these."

Expected:

- Bounding boxes are compared and the difference in magnitude is caught before
  overlaying.
- The answer names a projection mismatch rather than describing it as a data
  quality problem.

## Baseline failure modes to watch for (RED)

- Writing GeoJSON positions as lat, lon.
- Computing distance by differencing coordinates.
- Reporting an area in square degrees.
- Overlaying datasets without comparing bounding-box units and ranges.
- Combining elevation sources with no mention of vertical datum.

## Trigger record (2026-08-21)

Ran through `evals/trigger_eval.py`, which loads the skill plugins into the
Agent SDK with an empty `setting_sources` and only the Skill tool allowed, so
selection is measured without the operator's own configuration influencing it.

Prompt: "My points are landing in the ocean off Africa."

This skill fired, and no sibling fired alongside it. Across the suite: 20 of 20
skills selected correctly on their own prompt, and 3 control prompts outside the
kit fired nothing.

Selection is checked. Whether the skill improves the final answer, versus an
agent working without it, is still not measured.

