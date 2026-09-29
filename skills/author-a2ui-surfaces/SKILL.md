---
name: author-a2ui-surfaces
title: Author A2UI Surfaces
description: When a request calls for a dashboard, map, chart, table, or other interactive view of data, and exactly how to shape a valid first surface for it. Use when the current answer would benefit from an A2UI component instead of, or alongside, prose.
---

Companion to `present-interactive-analysis`: that skill covers *why* and *which*
component to reach for; this one covers the structural rules that make or break
the very first `create_a2ui_surface` call, so the surface renders on attempt one
instead of a refusal-fix-retry loop. Component property schemas are never
duplicated here — the catalog itself is the source of truth
(`load_skill("a2ui-catalog-<slug>")`, then `file="catalog.json#/components/<Name>"`
for one component).

## Recognize the request, don't wait to be asked

A request for a dashboard, an interactive view, a map, a chart, or a table over
data you already have or are about to fetch is a request for an A2UI surface —
produce it as part of that same turn's answer. Do not ship prose first and wait
for the user to separately ask to "see it as a2ui" or "show it visually"; by
then the work of shaping the surface is happening a turn late and under
pressure. Plain text stays right for a single fact, a short explanation, or
when the session's client cannot render A2UI at all (a `create_a2ui_surface`
refusal with reason `a2ui_client_capabilities_unknown` or
`a2ui_catalog_no_client_match` means exactly that — answer in prose and do not
retry).

## The shape that fails every time

These are the concrete rules a model trips on, each verified against the real
`clio-workspace` catalog validator (`jsonschema` compiled from `catalog.json`,
plus CLIO's safety walk):

- **One flat array, one `root`.** `components` is a flat list; exactly one
  entry has `"id": "root"` — that is the surface's mount point.
  `create_a2ui_surface` refuses outright (`a2ui_validation_failed`,
  `A2UI surface components must contain exactly one id="root" component`) with
  zero or with more than one.
- **Children are id references, never inline objects.** A container
  (`Row`/`Column`/`Grid`/`List`/`Frame`/`Tabs`/`Modal`) lists its children as
  an array of component id strings in its own `children` property; the
  children themselves are separate entries in the same flat array. Nesting a
  full component object inside `children` fails schema validation (confirmed:
  `component=Column id=root pointer=/children: [...] is not valid under any of
  the given schemas`). This is also what keeps every component independently
  addressable for a later `update_a2ui_components` call.
- **Every component schema is closed.** `unevaluatedProperties: false` on
  every component means an extra, misspelled, or borrowed-from-another
  -component property fails the whole component, not just that field —
  confirmed for a map's `tileUrl` and a data-table's `caption`
  (`Unevaluated properties are not allowed`). The renderer owns the map's
  basemap — never pass tile/style URLs. A table's title is a sibling `Text`,
  not a property on the table.
- **`accessibility` is always an object.** `{"label": "..."}`, never a bare
  string — a bare string fails with `'...' is not of type 'object'`.
- **A dynamic value is a literal or a `{"path": "/pointer"}` binding — the
  bound data lives separately.** Pass the actual values via
  `create_a2ui_surface`'s `data_model` argument (or `update_a2ui_data_model`
  on an existing surface), never as an invented `dataModel` property on a
  component. Prefer literals for anything static; reserve bindings for values
  you expect to refresh after creation.
- **`clio.time-series.v1` takes exactly one of `series` or `dataUri`, never
  both, never neither** (a `oneOf` — confirmed failing both ways with
  `is not valid under any of the given schemas`). Small inline data goes in
  `series`; anything larger goes through a registered artifact's `dataUri`
  (`^artifact://artifact_...$`).
- **`clio.metric.v1` is one number.** `label` and `value` are required; compose
  several metrics inside a `Row`/`Grid` instead of inventing a multi-value
  aggregate shape.
- **Size discipline.** Bring back what was asked for, not everything available
  — a request for "the 10 largest" is a 10-row table, not every row you
  fetched. Large series (thousands of points) belong in a registered artifact
  (`dataUri`) rather than inlined; components, string values, and nesting
  depth are all bounded server-side, and an oversized payload is a validation
  failure, not a silent truncation.
- **`catalog_id` stays empty unless you must pin one.** An empty `catalog_id`
  auto-selects this session's first declared catalog the client also
  supports; naming one that is not both declared and client-supported is a
  refusal (`a2ui_catalog_not_producible` / `a2ui_preferred_catalog_not_selectable`),
  never a silent substitution.
- **Actions are plain agent events unless the name is one of three.** Use
  `{"event": {"name": "...", "context": {...}}}` for an ordinary event the
  agent receives next turn — that now includes `agent.submit`/`form.submit`,
  there is no separate closed action vocabulary. Only `approval.respond`,
  `run.cancel`, and `run.retry` are routed elsewhere (the permission gate /
  run controller) before the agent ever sees them.

## The tool calls

- `create_a2ui_surface(surface_id, components, data_model=None, catalog_id="")`
  — creates a new surface, or revises one in place when `surface_id` matches a
  live id from a prior result's `session_surface_ids`. Always resend the full
  current component list; `data_model`, when given, becomes the surface's
  whole data model (bound `{"path": ...}` values resolve against it).
- `update_a2ui_components(surface_id, components)` — upserts one or more
  components on an already-created surface without resending the ones that
  did not change, and without the `root` requirement (that only applies to
  the first, surface-creating call).
- `update_a2ui_data_model(surface_id, path="/", value=None, delete=False)` —
  sets (or deletes) one JSON-Pointer path in an existing surface's data model,
  so a bound value refreshes without resending any component.
- `delete_a2ui_surface(surface_id)` — removes a surface that is no longer
  relevant to the conversation.

Call one of these at a time, in causal order; require `rendered: true` and
`state: "ready"` in the result before treating the surface as visible or
answering as if it were. A refusal is a typed dict (`ok: false`, `reason`,
`detail`, `hint`) — never printed as chat text — and its `hint` names the
exact next step (often `load_skill("a2ui-catalog-<slug>", file=...)` for the
one failing component); follow it and retry once, don't repeat the identical
call.

## Worked example: dashboard (map + metrics + time series + table)

A single `Column` root, one `Row` of metrics, one map, one time series, one
table — exactly the shape for "show this as an interactive dashboard: a map of
the points, a metric with the total, a time series over days, a table of the
largest N". Validated against the real `clio-workspace` catalog:

Arguments to `create_a2ui_surface` (verified against the real `clio-workspace`
catalog validator):

```json
{
  "surface_id": "quake-dashboard",
  "catalog_id": "",
  "components": [
    {"id": "root", "component": "Column",
     "children": ["title", "statsRow", "quakeMap", "quakeTrend", "quakeTable"]},
    {"id": "title", "component": "Text", "variant": "h3",
     "text": "M4.5+ Earthquakes — Last 30 Days"},
    {"id": "statsRow", "component": "Row", "children": ["totalMetric", "maxMetric"]},
    {"id": "totalMetric", "component": "clio.metric.v1",
     "label": "Total M4.5+ events", "value": 47, "unit": "events"},
    {"id": "maxMetric", "component": "clio.metric.v1",
     "label": "Largest magnitude", "value": 6.1, "unit": "Mw"},
    {"id": "quakeMap", "component": "clio.map.v1", "title": "Epicenters",
     "points": [
       {"id": "us7000q1", "label": "M6.1 - South Sandwich Islands",
        "latitude": -58.4, "longitude": -25.3},
       {"id": "us7000q2", "label": "M5.4 - Vanuatu",
        "latitude": -16.5, "longitude": 168.1}
     ]},
    {"id": "quakeTrend", "component": "clio.time-series.v1",
     "title": "Quakes per day", "xKey": "day", "yKeys": ["count"],
     "series": [
       {"day": "2026-08-30", "count": 1},
       {"day": "2026-08-31", "count": 2},
       {"day": "2026-09-01", "count": 0}
     ]},
    {"id": "quakeTable", "component": "clio.data-table.v1",
     "columns": [{"key": "place", "label": "Location"},
                 {"key": "magnitude", "label": "Magnitude"},
                 {"key": "time", "label": "Time (UTC)"}],
     "rows": [
       {"place": "South Sandwich Islands", "magnitude": 6.1, "time": "2026-09-14T02:11:00Z"},
       {"place": "Vanuatu", "magnitude": 5.4, "time": "2026-09-08T19:44:00Z"}
     ]}
  ]
}
```

Note what each piece demonstrates: `quakeMap`'s points carry no tile/style
URL; `quakeTrend` uses inline `series` (a few dozen days, well under artifact
territory) and never sets `dataUri` alongside it; `quakeTable` names its
title through the sibling `title` Text, not a table property; every container
(`root`, `statsRow`) references children by id.

## Worked example: bind a value, then refresh it without resending the tree

A long-running fetch shown as status + progress, where the progress value is
bound to the data model so it can be pushed forward with a small
`update_a2ui_data_model` call instead of resending the component list on every
tick:

Arguments to `create_a2ui_surface`:

```json
{
  "surface_id": "quake-fetch-progress",
  "data_model": {"progress": {"days": 12}},
  "components": [
    {"id": "root", "component": "Column",
     "children": ["fetchStatus", "fetchProgress"]},
    {"id": "fetchStatus", "component": "clio.status.v1",
     "label": "Fetching USGS feed", "state": "running"},
    {"id": "fetchProgress", "component": "clio.progress.v1",
     "label": "Days processed", "value": {"path": "/progress/days"}, "max": 30}
  ]
}
```

Once the fetch finishes, two small calls replace resending the whole surface —
`update_a2ui_data_model(surface_id="quake-fetch-progress", path="/progress/days",
value=30)`, then arguments to `update_a2ui_components`:

```json
{
  "surface_id": "quake-fetch-progress",
  "components": [
    {"id": "fetchStatus", "component": "clio.status.v1",
     "label": "Fetching USGS feed", "state": "complete"}
  ]
}
```
