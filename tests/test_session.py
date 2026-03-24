import json
import time
from unittest.mock import MagicMock, patch, call

import pytest

from latency_benchmark.models import AudioChunk, RunOutput, StreamingTranscript
from latency_benchmark.session import run_streaming_session


@pytest.fixture
def audio_chunks():
    """3 chunks of 100ms each."""
    return [AudioChunk(data=b"\x00" * 3200, duration_ms=100) for _ in range(3)]


def _make_ws_messages():
    """Simulate Begin, Turn, Termination sequence."""
    return [
        json.dumps({"type": "Begin", "id": "sess-123", "expires_at": "9999999999"}),
        json.dumps({
            "type": "Turn",
            "words": [{"text": "hello", "start": 500, "end": 800}],
            "end_of_turn": False,
            "end_of_turn_confidence": 0.0,
            "turn_is_formatted": False,
        }),
        json.dumps({
            "type": "Turn",
            "words": [{"text": "hello", "start": 500, "end": 800},
                      {"text": "world", "start": 900, "end": 1200}],
            "end_of_turn": True,
            "end_of_turn_confidence": 0.95,
            "turn_is_formatted": True,
        }),
        json.dumps({
            "type": "Termination",
            "audio_duration_seconds": 0.3,
            "session_duration_seconds": 0.5,
        }),
    ]


@patch("latency_benchmark.session.connect")
def test_run_session_returns_output(mock_connect, audio_chunks):
    mock_ws = MagicMock()
    mock_ws.__iter__ = MagicMock(return_value=iter(_make_ws_messages()))
    mock_ws.__enter__ = MagicMock(return_value=mock_ws)
    mock_ws.__exit__ = MagicMock(return_value=False)
    mock_connect.return_value = mock_ws

    output = run_streaming_session(
        api_endpoint="wss://streaming.assemblyai.com",
        api_key="test-key",
        audio_chunks=audio_chunks,
        sample_rate=16000,
        realtime=False,
    )

    assert isinstance(output, RunOutput)
    assert output.session_id == "sess-123"
    assert len(output.transcripts) == 2
    assert len(output.chunks_processing) == 3


@patch("latency_benchmark.session.connect")
def test_session_sends_terminate(mock_connect, audio_chunks):
    mock_ws = MagicMock()
    mock_ws.__iter__ = MagicMock(return_value=iter(_make_ws_messages()))
    mock_ws.__enter__ = MagicMock(return_value=mock_ws)
    mock_ws.__exit__ = MagicMock(return_value=False)
    mock_connect.return_value = mock_ws

    run_streaming_session(
        api_endpoint="wss://streaming.assemblyai.com",
        api_key="test-key",
        audio_chunks=audio_chunks,
        sample_rate=16000,
        realtime=False,
    )

    send_calls = mock_ws.send.call_args_list
    string_sends = [c for c in send_calls if isinstance(c.args[0], str)]
    assert "Terminate" in string_sends[-1].args[0]
