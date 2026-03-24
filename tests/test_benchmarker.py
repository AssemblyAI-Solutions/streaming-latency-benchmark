import pytest

from latency_benchmark.benchmarker import LatencyBenchmarker
from latency_benchmark.models import (
    AudioChunkProcessing,
    BenchmarkResult,
    RunOutput,
    StreamingTranscript,
    TranscribedWord,
)


def _make_ground_truth():
    """Ground truth: 'the cat sat' with precise timestamps."""
    return [
        TranscribedWord(text="the", start_ms=100, end_ms=250),
        TranscribedWord(text="cat", start_ms=300, end_ms=500),
        TranscribedWord(text="sat", start_ms=600, end_ms=800),
    ]


def _make_chunks_processing():
    """Audio chunks sent every 100ms. Chunk at 300ms covers 200-300ms, etc."""
    base_ts = 1000.0
    return [
        AudioChunkProcessing(audio_end_ts_ms=100, processing_ts=base_ts + 0.1),
        AudioChunkProcessing(audio_end_ts_ms=200, processing_ts=base_ts + 0.2),
        AudioChunkProcessing(audio_end_ts_ms=300, processing_ts=base_ts + 0.3),
        AudioChunkProcessing(audio_end_ts_ms=400, processing_ts=base_ts + 0.4),
        AudioChunkProcessing(audio_end_ts_ms=500, processing_ts=base_ts + 0.5),
        AudioChunkProcessing(audio_end_ts_ms=600, processing_ts=base_ts + 0.6),
        AudioChunkProcessing(audio_end_ts_ms=700, processing_ts=base_ts + 0.7),
        AudioChunkProcessing(audio_end_ts_ms=800, processing_ts=base_ts + 0.8),
        AudioChunkProcessing(audio_end_ts_ms=900, processing_ts=base_ts + 0.9),
        AudioChunkProcessing(audio_end_ts_ms=1000, processing_ts=base_ts + 1.0),
    ]


def _make_transcripts():
    """Simulated transcript messages: partial with 'the cat', final with 'the cat sat'."""
    base_ts = 1000.0
    return [
        StreamingTranscript(
            words=[
                TranscribedWord(text="the", start_ms=100, end_ms=250),
                TranscribedWord(text="cat", start_ms=300, end_ms=500),
            ],
            text="the cat",
            is_final=False,
            abs_processing_ts=base_ts + 0.6,
        ),
        StreamingTranscript(
            words=[
                TranscribedWord(text="the", start_ms=100, end_ms=250),
                TranscribedWord(text="cat", start_ms=300, end_ms=500),
                TranscribedWord(text="sat", start_ms=600, end_ms=800),
            ],
            text="the cat sat",
            is_final=True,
            abs_processing_ts=base_ts + 0.95,
        ),
    ]


def test_benchmark_returns_result():
    ground_truth = _make_ground_truth()
    output = RunOutput(
        session_id="test-1",
        session_init_latency_ms=100,
        chunks_processing=_make_chunks_processing(),
        transcripts=_make_transcripts(),
    )

    benchmarker = LatencyBenchmarker()
    result = benchmarker.run(output, ground_truth)

    assert isinstance(result, BenchmarkResult)
    assert result.session_init_latency_ms == 100
    assert len(result.per_word_latencies_ms) == 3


def test_benchmark_latency_values():
    """Verify per-word emission latency is computed correctly.

    'the' ends at 250ms -> chunk at 300ms sent at base+0.3 -> first transcript at base+0.6
       latency = 0.6 - 0.3 = 0.3s = 300ms
    'cat' ends at 500ms -> chunk at 500ms sent at base+0.5 -> first transcript at base+0.6
       latency = 0.6 - 0.5 = 0.1s = 100ms
    'sat' ends at 800ms -> chunk at 800ms sent at base+0.8 -> first transcript at base+0.95
       latency = 0.95 - 0.8 = 0.15s = 150ms
    """
    ground_truth = _make_ground_truth()
    output = RunOutput(
        session_id="test-1",
        session_init_latency_ms=100,
        chunks_processing=_make_chunks_processing(),
        transcripts=_make_transcripts(),
    )

    benchmarker = LatencyBenchmarker()
    result = benchmarker.run(output, ground_truth)

    assert result.per_word_latencies_ms == [300, 100, 150]


def test_benchmark_with_mismatched_transcript():
    """If the transcript has a substitution, only matched words get latency."""
    ground_truth = _make_ground_truth()
    output = RunOutput(
        session_id="test-2",
        session_init_latency_ms=100,
        chunks_processing=_make_chunks_processing(),
        transcripts=[
            StreamingTranscript(
                words=[
                    TranscribedWord(text="the", start_ms=100, end_ms=250),
                    TranscribedWord(text="bat", start_ms=300, end_ms=500),
                    TranscribedWord(text="sat", start_ms=600, end_ms=800),
                ],
                text="the bat sat",
                is_final=True,
                abs_processing_ts=1000.0 + 0.95,
            ),
        ],
    )

    benchmarker = LatencyBenchmarker()
    result = benchmarker.run(output, ground_truth)

    assert len(result.per_word_latencies_ms) == 2
