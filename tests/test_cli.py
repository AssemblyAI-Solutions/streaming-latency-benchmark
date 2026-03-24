from unittest.mock import patch, MagicMock

from click.testing import CliRunner

from latency_benchmark.cli import main
from latency_benchmark.models import (
    AudioChunkProcessing,
    BenchmarkResult,
    RunOutput,
    StreamingTranscript,
    TranscribedWord,
)


def _mock_run_output():
    return RunOutput(
        session_id="test-1",
        session_init_latency_ms=150,
        chunks_processing=[
            AudioChunkProcessing(audio_end_ts_ms=100, processing_ts=1000.1),
        ],
        transcripts=[
            StreamingTranscript(
                words=[TranscribedWord(text="hello", start_ms=50, end_ms=90)],
                text="hello",
                is_final=True,
                abs_processing_ts=1000.4,
            ),
        ],
    )


@patch("latency_benchmark.cli.run_streaming_session")
@patch("latency_benchmark.cli.discover_dataset")
@patch("latency_benchmark.cli.load_ground_truth")
@patch("latency_benchmark.cli.load_and_chunk_audio")
def test_cli_runs(mock_chunk, mock_gt, mock_discover, mock_session, tmp_path):
    mock_discover.return_value = [("audio.wav", "audio.json")]
    mock_gt.return_value = [TranscribedWord(text="hello", start_ms=50, end_ms=90)]
    mock_chunk.return_value = [MagicMock(data=b"\x00", duration_ms=100)]
    mock_session.return_value = _mock_run_output()

    runner = CliRunner()
    result = runner.invoke(main, [
        "--api-key", "test-key",
        "--dataset", str(tmp_path),
        "--output", str(tmp_path / "results"),
    ])

    assert result.exit_code == 0
    assert "Emission Latency Results" in result.output
