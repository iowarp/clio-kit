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
