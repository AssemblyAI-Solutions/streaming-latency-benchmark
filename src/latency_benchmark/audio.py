import wave
from typing import List

from latency_benchmark.models import AudioChunk


def load_and_chunk_audio(
    filepath: str,
    sample_rate: int = 16000,
    chunk_duration_ms: int = 100,
) -> List[AudioChunk]:
    """Load a WAV file and split it into fixed-duration PCM16 chunks."""
    with wave.open(filepath, "rb") as wf:
        assert wf.getnchannels() == 1, f"Expected mono audio, got {wf.getnchannels()} channels"
        assert wf.getsampwidth() == 2, f"Expected 16-bit audio, got {wf.getsampwidth() * 8}-bit"
        assert wf.getframerate() == sample_rate, (
            f"Expected {sample_rate}Hz, got {wf.getframerate()}Hz"
        )
        raw_data = wf.readframes(wf.getnframes())

    bytes_per_chunk = int(sample_rate * (chunk_duration_ms / 1000)) * 2
    chunks = []
    for offset in range(0, len(raw_data), bytes_per_chunk):
        chunk_data = raw_data[offset : offset + bytes_per_chunk]
        if len(chunk_data) == bytes_per_chunk:
            chunks.append(AudioChunk(data=chunk_data, duration_ms=chunk_duration_ms))

    return chunks
