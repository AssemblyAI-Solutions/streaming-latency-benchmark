import struct
import io
import wave

import pytest

from latency_benchmark.audio import load_and_chunk_audio
from latency_benchmark.models import AudioChunk


@pytest.fixture
def wav_file(tmp_path):
    """Create a 1-second mono 16-bit PCM WAV at 16kHz."""
    filepath = tmp_path / "test.wav"
    sample_rate = 16000
    num_samples = sample_rate  # 1 second
    samples = [0] * num_samples  # silence
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{num_samples}h", *samples))
    return filepath


def test_chunk_count(wav_file):
    """1 second of audio at 100ms chunks = 10 chunks."""
    chunks = load_and_chunk_audio(str(wav_file), sample_rate=16000, chunk_duration_ms=100)
    assert len(chunks) == 10


def test_chunk_type(wav_file):
    chunks = load_and_chunk_audio(str(wav_file), sample_rate=16000, chunk_duration_ms=100)
    assert all(isinstance(c, AudioChunk) for c in chunks)


def test_chunk_duration(wav_file):
    chunks = load_and_chunk_audio(str(wav_file), sample_rate=16000, chunk_duration_ms=100)
    assert all(c.duration_ms == 100 for c in chunks)


def test_chunk_byte_size(wav_file):
    """100ms at 16kHz, 16-bit mono = 1600 samples * 2 bytes = 3200 bytes."""
    chunks = load_and_chunk_audio(str(wav_file), sample_rate=16000, chunk_duration_ms=100)
    assert all(len(c.data) == 3200 for c in chunks)


def test_50ms_chunks(wav_file):
    """1 second at 50ms = 20 chunks."""
    chunks = load_and_chunk_audio(str(wav_file), sample_rate=16000, chunk_duration_ms=50)
    assert len(chunks) == 20
