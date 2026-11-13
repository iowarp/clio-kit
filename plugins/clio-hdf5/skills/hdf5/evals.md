# Evals - hdf5

Recorded 2026-10-13. These scenarios have not been run against a model. Each
pins a fact the guides got wrong before their October 2026 review against the
HDF5 2.0-2.2 and h5py 3.12-3.16 release notes, so a regression shows up as the
old answer coming back.

Current acceptance criterion: the answer names the guide's fact, checks the
installed HDF5 and h5py versions before version-specific advice, and reports
file facts only from an inspected file.

Baseline scenarios: run each WITHOUT the skill to capture the gap, then WITH it
to confirm the gap closes. Rubric is pass/fail per bullet.

## S1 - convention before walking

Setup: a NeXus file written by nexusformat (no root `default` attribute).
Prompt: "What's the main data in this file?"

Expected:

- The file is identified as NeXus from `NX_class` attributes.
- The answer finds the `NXdata` group even though the root `default` is
  missing, and names its `signal` dataset and `axes`.
- No dataset is read in full to answer.

## S2 - virtual datasets are writable

Prompt: "Can I write to a virtual dataset, or is it read-only?"

Expected:

- Writes are allowed and go through to the source datasets.
- The answer does not call virtual datasets read-only.

## S3 - ROS3 caching

Prompt: "I'm reading an HDF5 file from S3 with ROS3 and every read is slow.
Does ROS3 cache anything?"

Expected:

- The answer says ROS3 does cache: files written with paged aggregation can
  use the page buffer, and HDF5 2.2 adds a block cache.
- It suggests checking how the file was written (paged or not) before tuning.

## S4 - chunk cache default

Prompt: "What's the default number of chunk cache slots, and should I change it?"

Expected:

- 8191 slots since HDF5 2.0 (521 before); the answer checks the installed
  version.
- Advice to resize ties the cache to the chunk size and access pattern.

## S5 - building HDF5 2.x

Prompt: "How do I build HDF5 2.x with parallel support?"

Expected:

- CMake with `-DHDF5_ENABLE_PARALLEL=ON`.
- No `./configure` flags; Autotools was removed in HDF5 2.0.

## S6 - references in CLIO

Setup: a message containing `@water_224.h5:/entry_0000/4_azimuthal_integration/results/I`
and CLIO's description of that object.
Prompt: "What are the units here?"

Expected:

- The answer uses the attributes in the provided description instead of
  reopening the file.
- Objects named in the answer are written as references in the same form.

## S7 - boundary

Prompt: "Should I store these tables as HDF5 or Parquet?"

Expected:

- storage-format answers; this skill does not fire.
