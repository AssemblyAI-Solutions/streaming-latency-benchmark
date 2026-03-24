import os
import sys

import click
from dotenv import load_dotenv

load_dotenv()
import numpy as np

from latency_benchmark.audio import load_and_chunk_audio
from latency_benchmark.benchmarker import LatencyBenchmarker
from latency_benchmark.dataset import discover_dataset, load_ground_truth
from latency_benchmark.models import BenchmarkResult
from latency_benchmark.reporting import (
    compute_stats,
    format_stats_table,
    plot_latencies,
    write_csv,
    write_json,
)
from latency_benchmark.session import run_streaming_session


@click.command()
@click.option("--api-key", required=True, envvar="ASSEMBLYAI_API_KEY", help="AssemblyAI API key")
@click.option("--endpoint", default="wss://streaming.assemblyai.com/v3/ws", help="AssemblyAI streaming WebSocket endpoint")
@click.option("--dataset", required=True, type=click.Path(exists=True), help="Directory with audio + JSON pairs")
@click.option("--output", default="./results", help="Output directory for results")
@click.option("--sample-rate", default=16000, help="Audio sample rate in Hz")
@click.option("--chunk-size-ms", default=100, help="Audio chunk duration in ms")
@click.option("--speech-model", default="u3-rt-pro", help="Speech model override (default: u3-rt-pro)")
@click.option("--num-files", default=0, help="Number of files to process (0 = all)")
@click.option("--plot/--no-plot", default=True, help="Generate histogram plot")
@click.option("--format", "output_format", type=click.Choice(["csv", "json", "both"]), default="both")
def main(api_key, endpoint, dataset, output, sample_rate, chunk_size_ms, speech_model, num_files, plot, output_format):
    """Measure emission latency for AssemblyAI's streaming speech-to-text API."""
    pairs = discover_dataset(dataset)
    if not pairs:
        click.echo("No audio+JSON pairs found in dataset directory.", err=True)
        sys.exit(1)

    if num_files > 0:
        pairs = pairs[:num_files]

    click.echo(f"Found {len(pairs)} audio files to benchmark.\n")

    benchmarker = LatencyBenchmarker()
    all_latencies = []
    all_init_latencies = []
    all_wers = []

    failed_files = []
    for i, (audio_path, json_path) in enumerate(pairs, 1):
        filename = os.path.basename(audio_path)
        click.echo(f"[{i}/{len(pairs)}] Processing {filename}...")

        try:
            chunks = load_and_chunk_audio(audio_path, sample_rate, chunk_size_ms)
            ground_truth = load_ground_truth(json_path)

            run_output = run_streaming_session(
                api_endpoint=endpoint,
                api_key=api_key,
                audio_chunks=chunks,
                sample_rate=sample_rate,
                speech_model=speech_model,
            )

            result = benchmarker.run(run_output, ground_truth)

            all_latencies.extend(result.per_word_latencies_ms)
            all_init_latencies.append(result.session_init_latency_ms)
            all_wers.append(result.wer)

            file_stats = compute_stats(result.per_word_latencies_ms)
            click.echo(
                f"  -> {file_stats['count']} words, "
                f"mean={file_stats['mean']:.0f}ms, "
                f"median={file_stats['median']:.0f}ms, "
                f"p90={file_stats['p90']:.0f}ms, "
                f"WER={result.wer:.2%}"
            )
        except Exception as e:
            click.echo(f"  -> FAILED: {e}", err=True)
            failed_files.append(filename)

    if failed_files:
        click.echo(f"\n{len(failed_files)} file(s) failed: {', '.join(failed_files)}")

    if not all_latencies:
        click.echo("No latency data collected.", err=True)
        sys.exit(1)

    click.echo()
    stats = compute_stats(all_latencies)
    avg_init = int(np.mean(all_init_latencies)) if all_init_latencies else 0
    avg_wer = float(np.mean(all_wers)) if all_wers else 0

    click.echo(format_stats_table(stats, session_init_ms=avg_init, wer=avg_wer))

    os.makedirs(output, exist_ok=True)

    if output_format in ("csv", "both"):
        csv_path = os.path.join(output, "results.csv")
        write_csv(csv_path, all_latencies, session_init_ms=avg_init, wer=avg_wer)
        click.echo(f"\nCSV saved to: {csv_path}")

    if output_format in ("json", "both"):
        json_path = os.path.join(output, "results.json")
        write_json(json_path, all_latencies, session_init_ms=avg_init, wer=avg_wer)
        click.echo(f"JSON saved to: {json_path}")

    if plot and all_latencies:
        plot_path = os.path.join(output, "latency_histogram.png")
        plot_latencies(all_latencies, plot_path)
        click.echo(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
