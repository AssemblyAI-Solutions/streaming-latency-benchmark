import os
import struct
import wave

import pytest

from latency_benchmark.audio import load_and_chunk_audio
from latency_benchmark.session import run_streaming_session

API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")

pytestmark = pytest.mark.skipif(not API_KEY, reason="ASSEMBLYAI_API_KEY not set")


@pytest.fixture
def sample_audio(tmp_path):
    """Create a 2-second audio file with some non-silence content."""
    filepath = tmp_path / "test.wav"
    sample_rate = 16000
    num_samples = sample_rate * 2
    import math
    samples = [int(16000 * math.sin(2 * math.pi * 440 * t / sample_rate)) for t in range(num_samples)]
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{num_samples}h", *samples))
    return str(filepath)


def test_end_to_end_session(sample_audio):
    """Verify we can connect, stream, and get transcripts back."""
    chunks = load_and_chunk_audio(sample_audio, sample_rate=16000, chunk_duration_ms=100)

    output = run_streaming_session(
        api_endpoint="wss://streaming.assemblyai.com",
        api_key=API_KEY,
        audio_chunks=chunks,
        sample_rate=16000,
    )

    assert output.session_id != ""
    assert output.session_init_latency_ms > 0
    assert len(output.chunks_processing) == 20
