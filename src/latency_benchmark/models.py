from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class AudioChunk:
    """A chunk of raw audio bytes with its duration."""
    data: bytes
    duration_ms: int


@dataclass(frozen=True)
class AudioChunkProcessing:
    """Records when an audio chunk (covering up to audio_end_ts_ms) was sent."""
    audio_end_ts_ms: int
    processing_ts: float  # monotonic wall-clock time when chunk was sent


@dataclass(frozen=True)
class TranscribedWord:
    """A word with its ground-truth or model-reported timestamps."""
    text: str
    start_ms: float
    end_ms: float


@dataclass(frozen=True)
class StreamingTranscript:
    """A single transcript message received from the streaming API."""
    words: List[TranscribedWord]
    text: str
    is_final: bool
    abs_processing_ts: float  # monotonic wall-clock time when received


@dataclass(frozen=True)
class MatchedTranscript:
    """An aligned word matched to the transcript message it first appeared in."""
    transcript: StreamingTranscript
    is_last_word_of_final: bool


@dataclass(frozen=True)
class RunOutput:
    """Complete output from a single streaming session."""
    session_id: str
    session_init_latency_ms: int
    chunks_processing: List[AudioChunkProcessing]
    transcripts: List[StreamingTranscript]


@dataclass
class BenchmarkResult:
    """Aggregated benchmark result for a single audio file."""
    session_init_latency_ms: int
    first_partial_latency_ms: int
    per_word_latencies_ms: List[int]
    ttct_ms: List[int]
    wer: float
