import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import List, Optional
from urllib.parse import urlencode

from websockets.sync.client import ClientConnection, connect

from latency_benchmark.models import (
    AudioChunk,
    AudioChunkProcessing,
    RunOutput,
    StreamingTranscript,
    TranscribedWord,
)


def _buffer_audio(
    audio_chunks: List[AudioChunk],
    buffer: Queue,
    done: threading.Event,
    realtime: bool = True,
) -> None:
    """Feed audio chunks into buffer at real-time rate (or instantly if realtime=False)."""
    if realtime and audio_chunks:
        start = time.monotonic()
        for i, chunk in enumerate(audio_chunks):
            target = start + (i + 1) * chunk.duration_ms / 1000
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            buffer.put(chunk)
    else:
        for chunk in audio_chunks:
            buffer.put(chunk)
    done.set()


def _send_audio(
    ws: ClientConnection,
    buffer: Queue,
    done: threading.Event,
    silence_threshold_ms: int,
) -> List[AudioChunkProcessing]:
    """Read chunks from buffer and send to WebSocket. Returns send timestamps."""
    ws.send(json.dumps({"type": "EndpointSilenceThreshold", "value_ms": silence_threshold_ms}))

    audio_end_time_ms = 0
    processing_times = []

    while not done.is_set() or not buffer.empty():
        try:
            chunk = buffer.get(timeout=0.1)
        except queue.Empty:
            continue
        buffer.task_done()
        audio_end_time_ms += chunk.duration_ms
        ts = time.monotonic()
        ws.send(chunk.data)
        processing_times.append(
            AudioChunkProcessing(audio_end_ts_ms=audio_end_time_ms, processing_ts=ts)
        )

    ws.send(json.dumps({"type": "Terminate"}))
    return processing_times


def _receive_transcripts(
    ws: ClientConnection,
) -> tuple:
    """Read all messages from WebSocket. Returns (session_id, transcripts)."""
    transcripts = []
    session_id = ""

    for message in ws:
        recv_ts = time.monotonic()
        data = json.loads(message)
        msg_type = data.get("type", "")

        if msg_type == "Begin":
            session_id = data.get("id", "")

        elif msg_type == "Turn" and data.get("words"):
            words = [
                TranscribedWord(
                    text=w["text"],
                    start_ms=w["start"],
                    end_ms=w["end"],
                )
                for w in data["words"]
            ]
            text = " ".join(w["text"] for w in data["words"])
            is_final = data.get("end_of_turn", False)

            transcripts.append(
                StreamingTranscript(
                    words=words,
                    text=text,
                    is_final=is_final,
                    abs_processing_ts=recv_ts,
                )
            )

        elif msg_type == "Termination":
            break

    return session_id, transcripts


def run_streaming_session(
    api_endpoint: str,
    api_key: str,
    audio_chunks: List[AudioChunk],
    sample_rate: int = 16000,
    silence_threshold_ms: int = 700,
    speech_model: Optional[str] = None,
    realtime: bool = True,
) -> RunOutput:
    """Run a single streaming transcription session.

    Streams audio at real-time rate, collects all transcript messages,
    and records precise timestamps for latency computation.

    Set realtime=False to send audio as fast as possible (useful for testing).
    """
    params = {"sample_rate": sample_rate}
    if speech_model:
        params["speech_model"] = speech_model

    endpoint = f"{api_endpoint}/v3/ws?{urlencode(params)}"
    headers = {"Authorization": api_key}

    buffer = Queue()
    done = threading.Event()

    start_ts = time.monotonic()
    ws = connect(endpoint, additional_headers=headers)
    session_init_latency_ms = int((time.monotonic() - start_ts) * 1000)

    with ThreadPoolExecutor(max_workers=3) as pool:
        pool.submit(_buffer_audio, audio_chunks, buffer, done, realtime)
        send_future = pool.submit(_send_audio, ws, buffer, done, silence_threshold_ms)
        recv_future = pool.submit(_receive_transcripts, ws)

        chunks_processing = send_future.result()
        session_id, transcripts = recv_future.result()

    try:
        ws.close()
    except Exception:
        pass

    return RunOutput(
        session_id=session_id,
        session_init_latency_ms=session_init_latency_ms,
        chunks_processing=chunks_processing,
        transcripts=transcripts,
    )
