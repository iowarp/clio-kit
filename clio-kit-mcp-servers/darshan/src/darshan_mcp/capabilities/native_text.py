"""Read Darshan's documented tab-separated --base output."""

from datetime import datetime, timezone
import re


def module_counters(text: str) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 6 or line.startswith("#"):
            continue
        module, _, _, counter, value = parts[:5]
        try:
            number = int(value) if re.fullmatch(r"[-+]?\d+", value) else float(value)
        except ValueError:
            continue
        if number < 0:
            continue  # Darshan uses -1 for unavailable counters.
        counters = result.setdefault(module, {})
        counters[counter] = counters.get(counter, 0) + number
    return result


def parse_native_text(text: str) -> dict | None:
    if "# darshan log version:" not in text:
        return None
    result: dict = {"success": True, "job": {}, "modules": [], "files": {}}
    job = result["job"]
    aliases = {
        "uid": "user_id",
        "jobid": "job_id",
        "nprocs": "nprocs",
        "run time": "runtime",
    }
    for line in text.splitlines():
        if not line.startswith("# ") or ":" not in line:
            continue
        key, value = line[2:].split(":", 1)
        value = value.strip()
        if key in aliases:
            job[aliases[key]] = float(value) if key == "run time" else int(value)
        elif key in {"start_time", "end_time"}:
            job[key] = datetime.fromtimestamp(float(value), timezone.utc).isoformat()
        elif re.fullmatch(r"[A-Z0-9_-]+ module", key):
            result["modules"].append(key.removesuffix(" module"))
    grouped: dict[str, list[str]] = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 6 and not line.startswith("#"):
            grouped.setdefault(parts[5], []).append(line)
    for path, lines in grouped.items():
        modules = module_counters("\n".join(lines))
        # POSIX and MPI-IO can describe the same traffic. Prefer one layer.
        layer = next(
            (name for name in ("POSIX", "MPI-IO", "MPIIO", "STDIO") if name in modules),
            None,
        )
        if layer is None:
            continue
        values = modules[layer]
        prefix = "MPIIO" if layer in {"MPI-IO", "MPIIO"} else layer
        fields = {
            "bytes_read": "BYTES_READ",
            "bytes_written": "BYTES_WRITTEN",
            "read_ops": "READS",
            "write_ops": "WRITES",
            "read_time": "F_READ_TIME",
            "write_time": "F_WRITE_TIME",
            "sequential_reads": "SEQ_READS",
            "sequential_writes": "SEQ_WRITES",
        }
        entry = {
            field: values.get(f"{prefix}_{counter}", 0)
            for field, counter in fields.items()
        }
        if prefix == "MPIIO":
            for field, suffix in (("read_ops", "READS"), ("write_ops", "WRITES")):
                entry[field] = values.get(f"MPIIO_INDEP_{suffix}", 0) + values.get(
                    f"MPIIO_COLL_{suffix}", 0
                )
        entry["counter_layer"] = layer
        result["files"][path] = entry
    return result


def parse_legacy_text(stdout: str) -> dict:
    # Parse text output to extract key information
    parsed_data: dict = {
        "job": {},
        "modules": [],
        "files": {},
        "success": True,
    }

    lines = stdout.split("\n")
    current_section: str | None = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Parse job information
        if "Job ID:" in line:
            parsed_data["job"]["job_id"] = line.split(":", 1)[1].strip()
        elif "User ID:" in line:
            parsed_data["job"]["user_id"] = line.split(":", 1)[1].strip()
        elif "Start time:" in line:
            parsed_data["job"]["start_time"] = line.split(":", 1)[1].strip()
        elif "End time:" in line:
            parsed_data["job"]["end_time"] = line.split(":", 1)[1].strip()
        elif "Number of processes:" in line:
            parsed_data["job"]["nprocs"] = int(line.split(":", 1)[1].strip())
        elif "Modules in log:" in line:
            current_section = "modules"
        elif current_section == "modules" and line.startswith("-"):
            module_name = line.lstrip("- ").strip()
            parsed_data["modules"].append(module_name)

    return parsed_data


def weighted_size_stats(sizes: list[tuple[float, int]]) -> dict:
    """Describe per-file averages without allocating one entry per I/O call."""
    count = sum(weight for _, weight in sizes)
    mean = sum(size * weight for size, weight in sizes) / count
    return {
        "min": min(size for size, _ in sizes),
        "max": max(size for size, _ in sizes),
        "avg": mean,
        "std": (sum(weight * (size - mean) ** 2 for size, weight in sizes) / count)
        ** 0.5,
        "basis": "operation-weighted per-file averages; not individual request sizes",
    }


def job_runtime(job: dict) -> float | None:
    """Prefer the precise native runtime over second-resolution timestamps."""
    if job.get("runtime") is not None:
        return job["runtime"]
    try:
        start = datetime.fromisoformat(job["start_time"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(job["end_time"].replace("Z", "+00:00"))
        return (end - start).total_seconds()
    except (ValueError, TypeError, KeyError, AttributeError):
        return None
