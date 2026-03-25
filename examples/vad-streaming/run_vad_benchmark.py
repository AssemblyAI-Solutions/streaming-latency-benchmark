"""
Stream audio to AssemblyAI with Silero VAD running alongside.
Measures VAD-based latency metrics to validate the approach before recommending to customers.

Usage:
    pip install -r requirements.txt
    cp .env.example .env  # add your API key
    python run_vad_benchmark.py --dataset ../streaming-latency-benchmark/data/sample --num-files 5

What it measures:
    - VAD TTFB: speech_start (from VAD) → first transcript received
    - VAD Emission Latency: speech_end (from VAD) → last partial received
    - Session init latency: WebSocket connect → Begin message
"""

import argparse
import json
import os
import queue
import struct
import threading
import time
import wave
from typing import List, Optional, Tuple
from urllib.parse import urlencode

import warnings
warnings.filterwarnings("ignore", message=".*non-writable.*")

import numpy as np
import torch
from dotenv import load_dotenv
from silero_vad import load_silero_vad, VADIterator
from websockets.sync.client import connect

load_dotenv()

# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_wav(filepath: str) -> Tuple[bytes, int]:
    """Load a WAV file. Returns (raw_pcm16_bytes, sample_rate)."""
    with wave.open(filepath, "rb") as wf:
        assert wf.getnchannels() == 1, "Expected mono audio"
        assert wf.getsampwidth() == 2, "Expected 16-bit audio"
        sr = wf.getframerate()
        data = wf.readframes(wf.getnframes())
    return data, sr


def chunk_audio(raw_data: bytes, chunk_duration_ms: int, sample_rate: int) -> list:
    """Split raw PCM16 audio into fixed-duration chunks."""
    bytes_per_chunk = int(sample_rate * (chunk_duration_ms / 1000)) * 2
    chunks = []
    for offset in range(0, len(raw_data), bytes_per_chunk):
        chunk = raw_data[offset:offset + bytes_per_chunk]
        if len(chunk) == bytes_per_chunk:
            chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# VAD processing
# ---------------------------------------------------------------------------

def run_vad_on_audio(raw_data: bytes, sample_rate: int, vad_iterator: VADIterator) -> dict:
    """
    Run Silero VAD on the full audio to find all speech segments.
    Returns dict with list of speech segments and audio duration.

    Silero VAD requires 512-sample frames at 16kHz.
    """
    frame_size = 512  # samples (required by Silero at 16kHz)
    frame_bytes = frame_size * 2  # 16-bit = 2 bytes per sample

    vad_iterator.reset_states()

    segments = []  # list of {"start_ms": int, "end_ms": int}
    current_start = None
    total_samples = len(raw_data) // 2
    audio_duration_ms = (total_samples / sample_rate) * 1000

    for offset in range(0, len(raw_data), frame_bytes):
        frame = raw_data[offset:offset + frame_bytes]
        if len(frame) < frame_bytes:
            break

        # Convert PCM16 bytes to float32 tensor
        audio_int16 = torch.frombuffer(frame, dtype=torch.int16).clone()
        audio_float = audio_int16.float() / 32768.0

        # return_seconds=True gives timestamps in seconds
        result = vad_iterator(audio_float, return_seconds=True)

        if result is not None:
            if "start" in result:
                current_start = round(result["start"] * 1000)
            if "end" in result and current_start is not None:
                segments.append({
                    "start_ms": current_start,
                    "end_ms": round(result["end"] * 1000),
                })
                current_start = None

    return {
        "segments": segments,
        "speech_start_ms": segments[0]["start_ms"] if segments else None,
        "speech_end_ms": segments[0]["end_ms"] if segments else None,
        "audio_duration_ms": round(audio_duration_ms),
    }


# ---------------------------------------------------------------------------
# Streaming session with timestamps
# ---------------------------------------------------------------------------

def run_streaming_session(
    api_endpoint: str,
    api_key: str,
    audio_chunks: list,
    chunk_duration_ms: int,
    sample_rate: int,
    speech_model: Optional[str] = None,
) -> dict:
    """
    Stream audio to AssemblyAI WebSocket API at real-time rate.
    Returns session metadata and transcript timestamps.
    """
    params = {"sample_rate": sample_rate}
    if speech_model:
        params["speech_model"] = speech_model

    endpoint = f"{api_endpoint}?{urlencode(params)}"
    headers = {"Authorization": api_key}

    # Results collected by threads
    send_timestamps = []  # (audio_end_ms, monotonic_time)
    transcripts = []      # (text, is_final, monotonic_time)
    session_id = ""

    buffer = queue.Queue()
    done = threading.Event()

    def buffer_audio():
        start = time.monotonic()
        for i, chunk in enumerate(audio_chunks):
            target = start + (i + 1) * chunk_duration_ms / 1000
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            buffer.put(chunk)
        done.set()

    def send_audio(ws):
        ws.send(json.dumps({"type": "EndpointSilenceThreshold", "value_ms": 700}))
        audio_end_ms = 0
        while not done.is_set() or not buffer.empty():
            try:
                chunk = buffer.get(timeout=0.1)
            except queue.Empty:
                continue
            buffer.task_done()
            audio_end_ms += chunk_duration_ms
            ts = time.monotonic()
            ws.send(chunk)
            send_timestamps.append((audio_end_ms, ts))
        ws.send(json.dumps({"type": "Terminate"}))

    def receive_transcripts(ws):
        nonlocal session_id
        for message in ws:
            recv_ts = time.monotonic()
            data = json.loads(message)
            msg_type = data.get("type", "")

            if msg_type == "Begin":
                session_id = data.get("id", "")
            elif msg_type == "Turn" and data.get("words"):
                text = " ".join(w["text"] for w in data["words"])
                is_final = data.get("end_of_turn", False)
                transcripts.append((text, is_final, recv_ts))
            elif msg_type == "Termination":
                break

    # Connect and run
    start_ts = time.monotonic()
    ws = connect(endpoint, additional_headers=headers)
    session_init_ms = round((time.monotonic() - start_ts) * 1000)

    with ThreadPoolExecutor(max_workers=3) as pool:
        pool.submit(buffer_audio)
        pool.submit(send_audio, ws)
        pool.submit(receive_transcripts, ws)

    try:
        ws.close()
    except Exception:
        pass

    return {
        "session_id": session_id,
        "session_init_ms": session_init_ms,
        "send_timestamps": send_timestamps,
        "transcripts": transcripts,
    }


from concurrent.futures import ThreadPoolExecutor


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_vad_metrics(vad_result: dict, session_result: dict, chunk_duration_ms: int) -> dict:
    """
    Compute VAD-anchored latency metrics.

    VAD TTFB: How long after the first speech_start does the first transcript arrive?
    VAD Emission: For the first speech segment, how long after speech_end does
                  the next transcript arrive? This measures how quickly the API
                  responds after a complete utterance.

    We anchor VAD events to the audio timeline, then find the corresponding
    send timestamp (when that audio was actually sent to the API).
    """
    send_timestamps = session_result["send_timestamps"]
    transcripts = session_result["transcripts"]

    if not transcripts or not send_timestamps:
        return {"vad_ttfb_ms": None, "vad_emission_ms": None, "error": "No transcripts received"}

    speech_start_ms = vad_result.get("speech_start_ms")
    speech_end_ms = vad_result.get("speech_end_ms")
    segments = vad_result.get("segments", [])

    first_transcript_ts = transcripts[0][2]   # monotonic time of first transcript

    vad_ttfb_ms = None
    vad_emission_ms = None

    if speech_start_ms is not None:
        # Find the send timestamp of the chunk that covers speech_start
        for audio_end_ms, send_ts in send_timestamps:
            if audio_end_ms >= speech_start_ms:
                vad_ttfb_ms = round((first_transcript_ts - send_ts) * 1000)
                break

    if speech_end_ms is not None:
        # Find the send timestamp of the chunk that covers speech_end
        speech_end_send_ts = None
        for audio_end_ms, send_ts in send_timestamps:
            if audio_end_ms >= speech_end_ms:
                speech_end_send_ts = send_ts
                break

        if speech_end_send_ts is not None:
            # Find the first transcript that arrives AFTER the speech_end chunk was sent
            for text, is_final, recv_ts in transcripts:
                if recv_ts >= speech_end_send_ts:
                    vad_emission_ms = round((recv_ts - speech_end_send_ts) * 1000)
                    break

    return {
        "vad_ttfb_ms": vad_ttfb_ms,
        "vad_emission_ms": vad_emission_ms,
        "speech_start_ms": speech_start_ms,
        "speech_end_ms": speech_end_ms,
        "num_segments": len(segments),
        "audio_duration_ms": vad_result["audio_duration_ms"],
        "num_transcripts": len(transcripts),
        "num_finals": sum(1 for _, is_final, _ in transcripts if is_final),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_audio_files(dataset_dir: str) -> list:
    """Find WAV files in a directory."""
    return sorted(
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.endswith(".wav")
    )


def main():
    parser = argparse.ArgumentParser(description="Stream audio with Silero VAD + AssemblyAI")
    parser.add_argument("--dataset", required=True, help="Directory with WAV files")
    parser.add_argument("--num-files", type=int, default=5, help="Number of files to process")
    parser.add_argument("--endpoint", default="wss://streaming.assemblyai.com/v3/ws",
                        help="AssemblyAI WebSocket endpoint")
    parser.add_argument("--api-key", default=os.environ.get("ASSEMBLYAI_API_KEY"),
                        help="AssemblyAI API key (or set ASSEMBLYAI_API_KEY)")
    parser.add_argument("--speech-model", default=None, help="Speech model override")
    parser.add_argument("--chunk-size-ms", type=int, default=100, help="Audio chunk size in ms")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate in Hz")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: Set ASSEMBLYAI_API_KEY in .env or pass --api-key")
        return

    # Load VAD model once
    print("Loading Silero VAD model...")
    vad_model = load_silero_vad()
    vad_iterator = VADIterator(vad_model, sampling_rate=args.sample_rate)
    print("  Done.\n")

    # Find audio files
    audio_files = find_audio_files(args.dataset)
    if not audio_files:
        print(f"No WAV files found in {args.dataset}")
        return
    audio_files = audio_files[:args.num_files]
    print(f"Processing {len(audio_files)} files...\n")

    all_ttfb = []
    all_emission = []
    all_init = []

    for i, filepath in enumerate(audio_files, 1):
        filename = os.path.basename(filepath)
        print(f"[{i}/{len(audio_files)}] {filename}")

        # Load and chunk audio
        raw_data, sr = load_wav(filepath)
        chunks = chunk_audio(raw_data, args.chunk_size_ms, sr)

        # Run VAD offline to find speech boundaries
        vad_result = run_vad_on_audio(raw_data, sr, vad_iterator)
        print(f"  VAD: speech_start={vad_result['speech_start_ms']}ms, "
              f"speech_end={vad_result['speech_end_ms']}ms, "
              f"{len(vad_result['segments'])} segments, "
              f"duration={vad_result['audio_duration_ms']:.0f}ms")

        # Stream to AssemblyAI
        try:
            session = run_streaming_session(
                api_endpoint=args.endpoint,
                api_key=args.api_key,
                audio_chunks=chunks,
                chunk_duration_ms=args.chunk_size_ms,
                sample_rate=sr,
                speech_model=args.speech_model,
            )
        except Exception as e:
            print(f"  FAILED: {e}\n")
            continue

        # Compute metrics
        metrics = compute_vad_metrics(vad_result, session, args.chunk_size_ms)

        print(f"  Session init: {session['session_init_ms']}ms")
        print(f"  Transcripts: {metrics['num_transcripts']} ({metrics['num_finals']} finals)")
        print(f"  VAD TTFB: {metrics['vad_ttfb_ms']}ms")
        print(f"  VAD Emission: {metrics['vad_emission_ms']}ms")
        print()

        all_init.append(session["session_init_ms"])
        if metrics["vad_ttfb_ms"] is not None:
            all_ttfb.append(metrics["vad_ttfb_ms"])
        if metrics["vad_emission_ms"] is not None:
            all_emission.append(metrics["vad_emission_ms"])

    # Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)

    if all_ttfb:
        print(f"\nVAD TTFB (speech_start → first transcript):")
        print(f"  Median: {int(np.median(all_ttfb))}ms")
        print(f"  Mean:   {int(np.mean(all_ttfb))}ms")
        print(f"  P90:    {int(np.percentile(all_ttfb, 90))}ms")
        print(f"  Samples: {len(all_ttfb)}")
    else:
        print("\nNo VAD TTFB measurements (speech_start not detected)")

    if all_emission:
        print(f"\nVAD Emission Latency (speech_end → last partial):")
        print(f"  Median: {int(np.median(all_emission))}ms")
        print(f"  Mean:   {int(np.mean(all_emission))}ms")
        print(f"  P90:    {int(np.percentile(all_emission, 90))}ms")
        print(f"  Samples: {len(all_emission)}")
    else:
        print("\nNo VAD Emission measurements (speech_end not detected)")

    if all_init:
        print(f"\nSession Init Latency:")
        print(f"  Median: {int(np.median(all_init))}ms")
        print(f"  Mean:   {int(np.mean(all_init))}ms")

    print()


if __name__ == "__main__":
    main()
