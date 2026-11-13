# CF Conventions — Scientific Compliance Analysis

## Purpose

Advisory guide for assessing whether an HDF5/NetCDF4 file meets CF (Climate and
Forecast) conventions, framed by each requirement's **scientific and practical
impact** — not file-format mechanics. Use when a user asks whether their file is
CF compliant, what CF metadata it is missing, whether that metadata can be
inferred, and what action (if any) is required to make the data interoperable.
This guide is advisory: it explains what CF requires and how to close each gap;
it does not run a compliance checker.

**Related**: survey the file's variables and attributes first, and see
`references/datatypes.md` for `_FillValue` type-matching questions.

---

## What CF Compliance Actually Means

CF (Climate and Forecast) Conventions are a metadata standard that makes scientific data **self-describing**. A CF-compliant file carries enough information for any software to correctly interpret the physical meaning of its variables — units, coordinate systems, time reference, spatial coverage — without any out-of-band documentation.

The audience most affected: **THREDDS, ERDDAP, OPeNDAP, Panoply, xarray + cf_xarray, and any analysis code that relies on automatic unit/coordinate detection**. Non-compliance doesn't necessarily mean the data is wrong; it means tools can't interpret it automatically.

CF compliance is versioned (`CF-1.6` through `CF-1.12`). Files should declare their version in the root `Conventions` attribute. When a file declares no version, check against the latest released CF version (1.12) but note that many older files were written against 1.6 or 1.7.

Complex-valued variables: CF/netCDF files still use the nc-complex compound convention; files using the native `H5T_COMPLEX` class (HDF5 2.0+) need HDF5 >= 2.0 readers. h5py 3.16 can write C99 complex but still defaults to compound.

---

## Issue Categories: Scientific Severity

### Units (§3.1) — **High impact**

**What it is**: Every numeric variable must have a `units` attribute whose value is recognized by UDUNITS-2 (the scientific units library used by virtually all analysis tools).

**Scientific impact**: Without valid units, automated unit conversion is impossible. `xarray` won't perform unit-aware arithmetic. ERDDAP can't label axes or convert between unit systems. CF-aware tools will refuse to process or may silently misinterpret the data.

**Common causes**:
- Units string truncated by the writing software (e.g., ADCIRC appends comment text to the units string, producing `"seconds since 2024-04-04 12:00:00        ! NCDASE - BASE_DAT"` — UDUNITS rejects the comment)
- Informal unit names (`deg C` vs. `degrees_Celsius`, `m/s` vs. `m s-1`)
- Missing entirely on a variable the author considered "obvious"

**Can it be inferred?** Often yes — if `standard_name` is present, the canonical units are implied by the CF standard name table (e.g., `sea_surface_height_above_geoid` → `m`). But an agent should not silently substitute inferred units; always flag the discrepancy.

---

### Standard Name (§3.3) — **High impact for interoperability**

**What it is**: `standard_name` maps a variable to an entry in the [CF standard name table](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html), a controlled vocabulary of several thousand physical quantities with canonical units and definitions.

**Scientific impact**: Without `standard_name`, tools cannot automatically identify what a variable represents. ERDDAP variable matching, cross-dataset aggregation, and semantic search all depend on it. A file with `zeta` as a variable name is opaque; a file with `standard_name = sea_surface_height_above_geoid` is unambiguous.

**Can it be inferred?** Sometimes from `long_name` or variable name, but it requires domain knowledge. Do not guess — a wrong `standard_name` is worse than a missing one because it silently mislabels data. Flag as "needs data producer input."

**Canonical units**: Each standard name has a defined canonical unit. If `standard_name` is present but `units` is missing, the canonical units are the correct default.

---

### Time Coordinate (§4.4) — **High impact**

**What it is**: The time coordinate must have `units` in the form `<unit> since <epoch>` (e.g., `seconds since 1970-01-01 00:00:00`) parseable by UDUNITS, and optionally a `calendar` attribute.

**Scientific impact**: A malformed time units string means **no tool can decode timestamps**. This is the single most common breaking defect in real-world scientific files. xarray's `decode_times=True` will fail or silently produce NaT values. Time-series plots will have no axis. Temporal subsetting via OPeNDAP will fail.

**Calendar (§4.4.1)**: If missing, CF specifies the default is `standard` (mixed Julian/Gregorian, switching in October 1582); `proleptic_gregorian` is a different, separate value. The two agree for dates after 1582, so the default is almost always safe for modern observational data. For climate model output, calendars like `360_day`, `noleap`, or `all_leap` are common and **must** be declared — otherwise model time coordinates will be decoded incorrectly.

**Can calendar be inferred?** For observational/reanalysis data, `standard` is a safe assumption. For model output, do not assume — ask the data producer or check the model documentation.

---

### Coordinate Variables and the `coordinates` Attribute (§5) — **High impact for geospatial data**

**What it is**: Data variables must declare which variables serve as their coordinates via the `coordinates` attribute. For gridded data this is often implicit; for unstructured or multi-dimensional data (e.g., UGRID meshes, station data) it is required.

**Scientific impact**: Without proper coordinate linkage, tools can't associate data values with geographic locations. Panoply can't render maps. xarray can't assign coordinates. THREDDS spatial coverage metadata will be wrong or absent.

**Unstructured grids**: Files using UGRID conventions (common in ocean models like ADCIRC/SCHISM) extend CF with topology variables. These are not strictly part of CF but are declared in the `Conventions` attribute alongside CF (e.g., `CF-1.0 UGRID-0.9.0`). When coordinate linkage looks non-compliant on a UGRID file, the UGRID topology may satisfy the intent even though strict CF coordinate rules appear unmet — judge it against UGRID, not plain CF.

---

### Grid Mapping (§5.6) — **Medium impact**

**What it is**: Non-geographic coordinate systems (projected coordinates, rotated pole grids) must reference a grid mapping variable that encodes the projection parameters.

**Scientific impact**: Without grid mapping, any tool that tries to display or reproject data will use incorrect coordinates. For files that use lat/lon directly, grid mapping may not be needed. For model output on native grids, it is essential for correct geolocation.

**Can it be inferred?** Not reliably without domain knowledge of the model or instrument. Flag for the data producer.

---

### Naming Conventions (§2.3) — **Low impact**

**What it is**: Global attributes and variable names should start with a letter and contain only letters, digits, and underscores.

**Scientific impact**: Cosmetic. Affects programmatic access in some languages (Python attribute-style access, NetCDF-Java). Does not affect data correctness. Attributes with leading underscores (like `_FillValue`) are an intentional HDF5/NetCDF convention and should not be flagged as errors in practice.

---

### `Conventions` Attribute (§2.6.1) — **Low–medium impact**

**What it is**: The root group should have a `Conventions` attribute declaring the CF version used (e.g., `"CF-1.9"`). Multiple conventions can be declared space-separated (e.g., `"CF-1.0 UGRID-0.9.0"`).

**Scientific impact**: Without it, tools that auto-detect file type (THREDDS, ERDDAP catalog harvesters) may not recognize the file as CF-compliant, affecting discoverability. It does not affect whether the data can be read or interpreted correctly. Files that declare an older CF version (e.g., `CF-1.0`) will fail checks run against CF-1.11 but may be fully valid for their declared version.

---

### Missing Data / `_FillValue` (§2.5) — **Medium impact**

**What it is**: `_FillValue` marks missing or undefined data values. Should match the variable's data type.

**Scientific impact**: A wrong or missing `_FillValue` means masked values will be included in statistics, plots, and calculations as if they were real data — typically producing extreme outliers or domain errors (e.g., ocean fill values of `-99999` appearing as sea surface heights). This is data correctness issue, not just a compliance formality.

**Type mismatch**: A float32 dataset with a float64 `_FillValue` is a common defect. The value may not be exactly representable, so some fill cells won't be masked.

---

### Cell Methods (§7.3) — **Medium impact for aggregated data**

**What it is**: Describes how data was derived over space or time (e.g., `time: mean`, `area: sum`). Relevant for model output, gridded observations, climatologies.

**Scientific impact**: Without `cell_methods`, a consumer can't distinguish instantaneous values from time-averaged fields. This matters for combining datasets and for correctly interpreting uncertainty.

---

## Prioritizing Issues

When assessing a file's CF readiness, prioritize gaps using this framing:

| Priority | Fix required for... |
|----------|---------------------|
| Time units malformed | Any temporal analysis, all plotting, OPeNDAP access |
| Units missing/invalid | Unit-aware computation, ERDDAP display, cf_xarray |
| Standard name missing | Cross-dataset interoperability, ERDDAP variable matching |
| Coordinates not linked | Geospatial display, xarray coordinate assignment |
| Conventions attribute wrong version | THREDDS/catalog discoverability |
| Naming convention violations | Programmatic access edge cases; rarely blocking |
| Calendar missing | Model output with non-Gregorian calendar only |

Group gaps that share a root cause. For example, a malformed time units string surfaces as several distinct CF gaps (§3.1 Units + §4.4 Time Coordinate) but is one problem to fix, not three.

---

## What Can Be Inferred vs. What Requires the Data Producer

**Can often be inferred by an agent (with appropriate caveats):**
- `calendar = standard` for observational files if no evidence of a model calendar
- Canonical `units` from a present `standard_name` (always flag the inference)
- Whether `_FillValue` is a sentinel (look for values far outside physical range)
- Whether a file is UGRID-extended CF vs. plain CF (check `Conventions` attribute for UGRID token)

**Requires the data producer or documentation:**
- Correct `standard_name` for ambiguous variable names
- Projection parameters for `grid_mapping`
- Time epoch for completely missing or corrupt time units
- Calendar type for climate model output
- Physical meaning of variables with no `long_name` or `standard_name`

---

## Downstream Tool Quick Reference

| Tool | Breaks on | Degrades on |
|------|-----------|-------------|
| xarray (decode_times) | Malformed time units | Missing calendar |
| cf_xarray | Missing standard_name, axis attrs | Missing units |
| ERDDAP | No Conventions, bad units | Missing standard_name |
| THREDDS | — | Missing Conventions, bad spatial coords |
| Panoply | No spatial coords | Missing grid_mapping |
| OPeNDAP/NetCDF-Java | — | Naming convention violations |
