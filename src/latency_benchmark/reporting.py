import csv
import json
import os
from typing import Dict, List, Optional

import numpy as np


def compute_stats(latencies_ms: List[int]) -> Dict[str, float]:
    """Compute summary statistics for a list of latencies."""
    if not latencies_ms:
        return {"mean": 0, "median": 0, "p90": 0, "p99": 0, "std_dev": 0, "count": 0}

    arr = np.array(latencies_ms)
    return {
        "mean": round(float(np.mean(arr)), 1),
        "median": round(float(np.median(arr)), 1),
        "p90": round(float(np.percentile(arr, 90)), 1),
        "p99": round(float(np.percentile(arr, 99)), 1),
        "std_dev": round(float(np.std(arr, ddof=1)), 1) if len(arr) > 1 else 0,
        "count": len(latencies_ms),
    }


def _format_block(title: str, stats: Dict[str, float]) -> List[str]:
    return [
        title,
        "=" * 40,
        f"  Samples:             {stats['count']}",
        f"  Mean:                {stats['mean']:.1f} ms",
        f"  Median:              {stats['median']:.1f} ms",
        f"  P90:                 {stats['p90']:.1f} ms",
        f"  P99:                 {stats['p99']:.1f} ms",
        f"  Std Dev:             {stats['std_dev']:.1f} ms",
    ]


def format_stats_table(
    stats: Dict[str, float],
    session_init_ms: int = 0,
    wer: float = 0.0,
    ttct_stats: Optional[Dict[str, float]] = None,
) -> str:
    """Format stats as a readable text table."""
    lines = _format_block("Emission Latency Results", stats)
    lines.append("=" * 40)

    if ttct_stats:
        lines.append("")
        lines.extend(_format_block("TTCT (Time to Complete Transcript) Results", ttct_stats))
        lines.append("=" * 40)

    lines.append(f"  Session Init:        {session_init_ms} ms")
    lines.append(f"  WER:                 {wer:.2%}")
    lines.append("=" * 40)
    return "\n".join(lines)


def write_csv(
    path: str,
    latencies_ms: List[int],
    session_init_ms: int = 0,
    wer: float = 0.0,
    ttct_ms: Optional[List[int]] = None,
) -> None:
    """Write results to a CSV file."""
    stats = compute_stats(latencies_ms)
    ttct_stats = compute_stats(ttct_ms) if ttct_ms is not None else None

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["emission_mean", stats["mean"]])
        writer.writerow(["emission_median", stats["median"]])
        writer.writerow(["emission_p90", stats["p90"]])
        writer.writerow(["emission_p99", stats["p99"]])
        writer.writerow(["emission_std_dev", stats["std_dev"]])
        writer.writerow(["emission_count", stats["count"]])
        if ttct_stats is not None:
            writer.writerow(["ttct_mean", ttct_stats["mean"]])
            writer.writerow(["ttct_median", ttct_stats["median"]])
            writer.writerow(["ttct_p90", ttct_stats["p90"]])
            writer.writerow(["ttct_p99", ttct_stats["p99"]])
            writer.writerow(["ttct_std_dev", ttct_stats["std_dev"]])
            writer.writerow(["ttct_count", ttct_stats["count"]])
        writer.writerow(["session_init_ms", session_init_ms])
        writer.writerow(["wer", round(wer, 4)])
        writer.writerow([])
        writer.writerow(["word_index", "latency_ms"])
        for i, lat in enumerate(latencies_ms):
            writer.writerow([i, lat])
        if ttct_ms is not None:
            writer.writerow([])
            writer.writerow(["turn_index", "ttct_ms"])
            for i, lat in enumerate(ttct_ms):
                writer.writerow([i, lat])


def write_json(
    path: str,
    latencies_ms: List[int],
    session_init_ms: int = 0,
    wer: float = 0.0,
    ttct_ms: Optional[List[int]] = None,
) -> None:
    """Write results to a JSON file."""
    data = {
        "emission_latency": {
            "stats": compute_stats(latencies_ms),
            "per_word_latencies_ms": latencies_ms,
        },
        "session_init_ms": session_init_ms,
        "wer": wer,
    }
    if ttct_ms is not None:
        data["ttct"] = {
            "stats": compute_stats(ttct_ms),
            "per_turn_ttct_ms": ttct_ms,
        }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def plot_latencies(latencies_ms: List[int], output_path: str, title: str = "Emission Latency Distribution") -> None:
    """Generate and save a histogram of emission latencies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bucket_size = 50
    min_lat = min(0, min(latencies_ms))  # include negatives if present
    max_lat = max(latencies_ms)
    bins = np.arange(min_lat, max_lat + bucket_size, bucket_size)
    if len(bins) < 2:
        bins = np.array([min_lat, min_lat + bucket_size])
    hist, edges = np.histogram(latencies_ms, bins=bins)
    total = len(latencies_ms)
    pcts = (hist / total) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(edges[:-1], pcts, width=bucket_size, align="edge", edgecolor="black")
    ax.set_title(f"{title} ({total} samples)")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("% of Total")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
