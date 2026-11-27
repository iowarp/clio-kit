"""Dataset statistics and explicit coverage for agent-facing summaries."""

from typing import List

import numpy as np


def compute_dataset_stats(file_proxy, path: str, stats: List[str]) -> dict:
    """Compute statistics for a single dataset."""
    dataset = file_proxy[path]

    step = 1
    if dataset.nbytes > 500 * 1024 * 1024:
        sample_size = min(1000000, max(1000, dataset.size // 100))
        if len(dataset.shape) == 1:
            step = max(1, dataset.size // sample_size)
            data = dataset[::step]
        else:
            step = max(1, dataset.shape[0] // int(np.sqrt(sample_size)))
            data = dataset[::step]
    else:
        data = dataset[()]

    result = {
        "shape": dataset.shape,
        "dtype": str(dataset.dtype),
        "size_mb": dataset.nbytes / (1024 * 1024),
        "sampled": data.size < dataset.size,
        "elements_processed": int(data.size),
        "total_elements": int(dataset.size),
        "sampling_step": step,
    }

    if np.issubdtype(data.dtype, np.number):
        if "mean" in stats:
            result["mean"] = float(np.mean(data))
        if "std" in stats:
            result["std"] = float(np.std(data))
        if "min" in stats:
            result["min"] = float(np.min(data))
        if "max" in stats:
            result["max"] = float(np.max(data))
        if "sum" in stats:
            result["sum"] = float(np.sum(data))
        if "count" in stats:
            result["count"] = int(data.size)
        if "median" in stats:
            result["median"] = float(np.median(data))
    else:
        result["note"] = (
            f"Non-numeric data type ({data.dtype}), limited statistics available"
        )
        if "count" in stats:
            result["count"] = int(data.size)

    return result


def format_statistics(results: dict, stats_list: list[str]) -> str:
    # Aggregate results
    successful_stats = {k: v for k, v in results.items() if "error" not in v}

    summary = f"Aggregate statistics for {len(results)} datasets:\n\n"

    for path, stat_result in results.items():
        if "error" in stat_result:
            summary += f"✗ {path}: {stat_result['error']}\n"
        else:
            summary += f"✓ {path}:\n"
            summary += f"  Shape: {stat_result['shape']}, Size: {stat_result['size_mb']:.2f} MB\n"
            processed = stat_result["elements_processed"]
            total = stat_result["total_elements"]
            if stat_result["sampled"]:
                summary += (
                    f"  SAMPLED: {processed:,} of {total:,} elements "
                    f"({processed / total:.2%} coverage).\n"
                    f"  Selection: every {stat_result['sampling_step']}th entry along axis 0.\n"
                    "  All statistics below describe the sample only; sum/count are not "
                    "full-dataset totals and min/max are not full-dataset bounds.\n"
                )
            else:
                summary += f"  FULL DATA: {processed:,} of {total:,} elements.\n"
            if "note" in stat_result:
                summary += f"  {stat_result['note']}\n"
            for stat_name in stats_list:
                if stat_name in stat_result:
                    summary += f"  {stat_name}: {stat_result[stat_name]:.6f}\n"
            summary += "\n"

    # Cross-dataset aggregation
    if len(successful_stats) > 1:
        if any(result["sampled"] for result in successful_stats.values()):
            return summary + (
                "Cross-dataset aggregation omitted: at least one dataset is sampled.\n"
                "Full-data totals or global bounds cannot be inferred from these samples.\n"
            )
        summary += "Cross-dataset aggregation:\n"

        for stat_name in ["mean", "sum", "count"]:
            if all(
                stat_name in stats_list and stat_name in result
                for result in successful_stats.values()
            ):
                values = [result[stat_name] for result in successful_stats.values()]
                if stat_name == "mean":
                    counts = [
                        result["elements_processed"]
                        for result in successful_stats.values()
                    ]
                    total_count = sum(counts)
                    weighted_mean = (
                        sum(v * c for v, c in zip(values, counts)) / total_count
                        if total_count > 0
                        else 0
                    )
                    summary += f"  Overall {stat_name}: {weighted_mean:.6f}\n"
                elif stat_name == "sum":
                    summary += f"  Total {stat_name}: {sum(values):.6f}\n"
                elif stat_name == "count":
                    summary += f"  Total {stat_name}: {sum(values):,}\n"

        if all("min" in result for result in successful_stats.values()):
            global_min = min(result["min"] for result in successful_stats.values())
            summary += f"  Global min: {global_min:.6f}\n"

        if all("max" in result for result in successful_stats.values()):
            global_max = max(result["max"] for result in successful_stats.values())
            summary += f"  Global max: {global_max:.6f}\n"

    return summary


PROFILE_STATS = ("mean", "std", "min", "max", "sum", "median")
DEFAULT_PROFILE_STATS = ["mean", "std", "min", "max"]
MAX_PROFILE_INDICES = 1000
MAX_PROFILE_BYTES = 4 * 1024 * 1024 * 1024
PROFILE_BLOCK_BYTES = 64 * 1024 * 1024


def compute_axis_profile(file_proxy, path: str, axis: int, stats: List[str]) -> dict:
    """Per-index statistics along one axis, reduced over every other axis.

    Reads the full dataset in bounded blocks along ``axis`` so a profile such as
    per-timestep min/max never needs the whole array in memory.
    """
    dataset = file_proxy[path]
    if not np.issubdtype(dataset.dtype, np.number):
        raise ValueError(f"axis profiles need numeric data, got {dataset.dtype}")
    ndim = dataset.ndim
    if ndim < 2:
        raise ValueError("axis profiles need a dataset with at least 2 dimensions")
    if not -ndim <= axis < ndim:
        raise ValueError(f"axis {axis} is out of range for {ndim} dimensions")
    axis = axis % ndim
    length = dataset.shape[axis]
    if length > MAX_PROFILE_INDICES:
        raise ValueError(
            f"axis {axis} has {length} indices; profiles are limited to "
            f"{MAX_PROFILE_INDICES}. Use read_partial_dataset or a shorter axis."
        )
    if dataset.nbytes > MAX_PROFILE_BYTES:
        raise ValueError(
            f"dataset is {dataset.nbytes / 1024**3:.1f} GiB; axis profiles scan the "
            f"full data and are limited to {MAX_PROFILE_BYTES // 1024**3} GiB"
        )
    chosen = [name for name in stats if name in PROFILE_STATS]
    if not chosen:
        chosen = list(DEFAULT_PROFILE_STATS)
    other_axes = tuple(index for index in range(ndim) if index != axis)
    per_index_bytes = max(1, dataset.nbytes // max(1, length))
    block = max(1, PROFILE_BLOCK_BYTES // per_index_bytes)
    profile: dict[str, list[float]] = {name: [] for name in chosen}
    for start in range(0, length, block):
        selection = [slice(None)] * ndim
        selection[axis] = slice(start, min(length, start + block))
        data = dataset[tuple(selection)]
        for name in chosen:
            reduced = getattr(np, name)(data, axis=other_axes)
            profile[name].extend(float(value) for value in reduced)
    return {
        "shape": dataset.shape,
        "dtype": str(dataset.dtype),
        "axis": axis,
        "length": length,
        "reduced_axes": other_axes,
        "stats": chosen,
        "profile": profile,
    }


def format_axis_profiles(results: dict) -> str:
    """Render per-index profiles as compact tables, one per dataset."""
    lines: list[str] = [f"Axis profiles for {len(results)} datasets:", ""]
    for path, result in results.items():
        if "error" in result:
            lines.extend([f"✗ {path}: {result['error']}", ""])
            continue
        chosen = result["stats"]
        lines.append(
            f"✓ {path}: shape {tuple(result['shape'])}, {result['dtype']}; "
            f"axis {result['axis']} profile over {result['length']} indices, "
            f"full data reduced over axes {tuple(result['reduced_axes'])}"
        )
        if "max" in result["profile"]:
            maxima = result["profile"]["max"]
            peak = int(np.argmax(maxima))
            lines.append(f"  Largest max: {maxima[peak]:.6g} at index {peak}")
        if "min" in result["profile"]:
            minima = result["profile"]["min"]
            low = int(np.argmin(minima))
            lines.append(f"  Smallest min: {minima[low]:.6g} at index {low}")
        lines.append("  index " + " ".join(f"{name:>12}" for name in chosen))
        for index in range(result["length"]):
            values = " ".join(
                f"{result['profile'][name][index]:>12.6g}" for name in chosen
            )
            lines.append(f"  {index:>5} {values}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
