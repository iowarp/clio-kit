---
name: recording-a-session-for-provenance
description: Use when explicitly recording or retrieving a research decision trail with ChronoLog. Triggers on "record this session", "ChronoLog", "retrieve our decisions". Not for job execution profiling; use diagnosing-a-slow-job.
metadata:
  bundle: clio-performance
  servers: clio-chronolog
  provenance: designed
  eval-status: scenarios-recorded
---

# Record and Retrieve Research Session Provenance

ChronoLog stores **the conversation** — user messages and the responses to them —
in a named chronicle and story. It is a provenance trail, not a profiler.

This is worth being precise about, because the server sits in a bundle with the
I/O profiler and the two get confused. ChronoLog has nothing to say about how
long a job ran or what it did to the filesystem. For that, see
`diagnosing-a-slow-job`.

## The handle is the whole thing

`clio-chronolog:start_chronolog` connects, creates the chronicle, and acquires a
story handle. Everything after depends on that handle existing:

1. `clio-chronolog:start_chronolog` — connect, name the chronicle and story
2. `clio-chronolog:record_interaction` — log a message and its response
3. `clio-chronolog:stop_chronolog` — release the handle and disconnect

`record_interaction` without a live handle fails. Recording is not automatic
either: nothing is captured unless the call is made for it, so a session with one
`start_chronolog` and no `record_interaction` produces an empty story.

**Always stop.** The handle is a held resource on the ChronoLog side. Leaving it
open after the work is done leaks it, and the story may not be readable until it
is released.

## Reading it back

`clio-chronolog:retrieve_interaction` takes the chronicle and story names, with
optional time filtering. This is a separate operation from recording — reading an
old story needs no `start_chronolog` for it.

Name chronicles and stories for how they will be searched later. A story called
`session` is unfindable a month afterwards; one named for the experiment and date
is not.

## When this is worth doing

Recording every session is pure overhead. It earns its place when the reasoning
matters as much as the result: a parameter sweep where the choices need
justifying later, a debugging session someone else will pick up, or work that
feeds a paper and has to be reconstructable.

## What not to do

- Do not use ChronoLog to investigate job performance.
- Do not call `record_interaction` before `start_chronolog`.
- Do not leave the handle open at the end of a session.
- Do not assume anything was captured — recording is explicit, per interaction.
- Do not name a chronicle something you could not search for later.

## Tool discovery across agents

Names such as `clio-hdf5:open_file` identify a server and its tool in this
guide. Your agent may expose a different prefix. Match the server and tool
against its live MCP inventory, then use the advertised name and input schema.
If a required server is unavailable, report it before attempting the workflow.

## Completion check

Record only the requested decision trail and relevant artifacts, excluding credentials and unrelated private content. Verify a known record can be retrieved, report chronicle/story and retrieval path, and stop the acquired session even if recording fails.
