from latency_benchmark.models import (
    AudioChunk,
    AudioChunkProcessing,
    TranscribedWord,
    StreamingTranscript,
    MatchedTranscript,
    RunOutput,
    BenchmarkResult,
)


def test_audio_chunk_creation():
    chunk = AudioChunk(data=b"\x00" * 100, duration_ms=100)
    assert chunk.data == b"\x00" * 100
    assert chunk.duration_ms == 100


def test_transcribed_word_creation():
    word = TranscribedWord(text="hello", start_ms=1200.0, end_ms=1500.0)
    assert word.text == "hello"
    assert word.start_ms == 1200.0
    assert word.end_ms == 1500.0


def test_audio_chunk_processing():
    acp = AudioChunkProcessing(audio_end_ts_ms=1500, processing_ts=1000.5)
    assert acp.audio_end_ts_ms == 1500
    assert acp.processing_ts == 1000.5


def test_streaming_transcript():
    words = [TranscribedWord(text="hello", start_ms=1200.0, end_ms=1500.0)]
    st = StreamingTranscript(
        words=words,
        text="hello",
        is_final=True,
        abs_processing_ts=1000.6,
    )
    assert st.is_final is True
    assert st.text == "hello"


def test_run_output():
    output = RunOutput(
        session_id="test-1",
        session_init_latency_ms=150,
        chunks_processing=[],
        transcripts=[],
    )
    assert output.session_id == "test-1"
    assert output.session_init_latency_ms == 150


def test_benchmark_result():
    result = BenchmarkResult(
        session_init_latency_ms=150,
        first_partial_latency_ms=320,
        per_word_latencies_ms=[300, 320, 280, 350],
        wer=0.05,
    )
    assert result.first_partial_latency_ms == 320
    assert len(result.per_word_latencies_ms) == 4
