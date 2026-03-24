import csv
import json
import os
from typing import Dict, List

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


def format_stats_table(stats: Dict[str, float], session_init_ms: int = 0, wer: float = 0.0) -> str:
    """Format stats as a readable text table."""
    lines = [
        "Emission Latency Results",
        "=" * 40,
        f"  Samples:             {stats['count']}",
        f"  Mean:                {stats['mean']:.1f} ms",
        f"  Median:              {stats['median']:.1f} ms",
        f"  P90:                 {stats['p90']:.1f} ms",
        f"  P99:                 {stats['p99']:.1f} ms",
        f"  Std Dev:             {stats['std_dev']:.1f} ms",
        f"  Session Init:        {session_init_ms} ms",
        f"  WER:                 {wer:.2%}",
        "=" * 40,
    ]
    return "\n".join(lines)


def write_csv(path: str, latencies_ms: List[int], session_init_ms: int = 0, wer: float = 0.0) -> None:
    """Write results to a CSV file."""
    stats = compute_stats(latencies_ms)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["mean", stats["mean"]])
        writer.writerow(["median", stats["median"]])
        writer.writerow(["p90", stats["p90"]])
        writer.writerow(["p99", stats["p99"]])
        writer.writerow(["std_dev", stats["std_dev"]])
        writer.writerow(["count", stats["count"]])
        writer.writerow(["session_init_ms", session_init_ms])
        writer.writerow(["wer", round(wer, 4)])
        writer.writerow([])
        writer.writerow(["word_index", "latency_ms"])
        for i, lat in enumerate(latencies_ms):
            writer.writerow([i, lat])


def write_json(path: str, latencies_ms: List[int], session_init_ms: int = 0, wer: float = 0.0) -> None:
    """Write results to a JSON file."""
    stats = compute_stats(latencies_ms)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "stats": stats,
        "session_init_ms": session_init_ms,
        "wer": wer,
        "per_word_latencies_ms": latencies_ms,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def plot_latencies(latencies_ms: List[int], output_path: str, title: str = "Emission Latency Distribution") -> None:
    """Generate and save a histogram of emission latencies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bucket_size = 50
    max_lat = max(latencies_ms)
    bins = np.arange(0, max_lat + bucket_size, bucket_size)
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
