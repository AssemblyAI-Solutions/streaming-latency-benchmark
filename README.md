# Streaming Latency Benchmark

Measure **emission latency** and **TTCT (Time to Complete Transcript)** for AssemblyAI's streaming speech-to-text API.

## What is Emission Latency?

Emission latency measures how quickly a streaming STT API recognizes a word
after it has received all the audio containing that word. It's the processing
delay — the gap between "the system has all the audio it needs" and "the system
tells you the word."

### Why emission latency over TTFB?

**Time to First Byte (TTFB)** is popular in LLM benchmarking, but for speech-to-text
it can be misleading:

- It only measures the **first** word, not ongoing performance
- It's heavily affected by initial silence/ringing before speech starts
- It depends on model timestamp accuracy for speech-adjusted calculations
- It can be trivially "gamed" by emitting low-confidence junk early

**Emission latency** is more useful because:

- It measures **every word**, giving you a distribution of real performance
- It uses human-labeled ground truth, so measurements are accurate
- For use cases like voicemail detection, it directly answers: "how fast do I
  get words back after they're spoken?"

## What is TTCT?

**TTCT (Time to Complete Transcript)** measures how quickly the system returns
a *finalized* transcript after the user has finished speaking. That's the
moment a downstream voice agent can actually start generating a response, so
TTCT is the actionable end-to-end latency for voice-agent use cases.

### When to use TTCT vs emission latency

- **Voice agents**: use **TTCT** — the agent
  cannot act on partial results that may still be revised, so what matters is
  how fast the final transcript arrives after end-of-speech.
- **Live captioning, voicemail detection, transcription monitoring**: use
  **emission latency** — these consume words as they stream and don't need to
  wait for finalization.

TTCT is computed per finalized turn, not per word. It includes the
silence-threshold wait (700ms by default), end-of-turn detection, final-pass
decoding, and network round-trip — i.e., everything the user actually waits
through. As a result, TTCT is typically much larger than emission latency.

### How it works

1. Stream audio to AssemblyAI's API at real-time rate, recording when each chunk is sent
2. Collect all transcript messages, recording when each is received
3. Align the API's output against human-labeled ground truth using edit distance
4. For each matched word: `emission_latency = time_transcript_received - time_last_audio_chunk_sent`
5. For each finalized turn: `ttct = time_final_turn_received - time_audio_chunk_sent_for_last_word_in_turn`
6. Aggregate both into mean, median, P90, P99

## Architecture

### How the benchmark works

The tool runs a pipeline for each audio file in the dataset:

```
Load audio        Stream to API       Align & compute            Report
+ ground truth -> via WebSocket    -> emission latency / TTCT -> results
                  at real-time        per word / per turn
```

**Step 1 — Load:** Read a WAV file and split it into fixed-duration chunks (default 100ms). Load the matching JSON file containing human-labeled word timestamps.

**Step 2 — Stream:** Open a WebSocket connection to AssemblyAI's streaming API. Three threads run concurrently:
- **Buffer thread** feeds audio chunks at real-time rate (drift-free scheduling)
- **Send thread** writes chunks to the WebSocket and records the monotonic timestamp of each send
- **Receive thread** reads transcript messages from the WebSocket and records the monotonic timestamp of each receive

**Step 3 — Align & compute (emission latency):** The API's transcript might not perfectly match ground truth (some words may be missing, substituted, or added). The tool uses [jiwer](https://github.com/jitsi/jiwer) to perform word-level edit-distance alignment between the normalized ground truth and the normalized API output, keeping only words that match in both. For each matched word:
- Find the audio chunk that covers the word's end timestamp ("Moment A" — when all audio for this word was sent)
- Find the first transcript message containing this word ("Moment B" — when the API first returned this word)
- **Emission latency = Moment B − Moment A**

**Step 4 — Compute (TTCT):** For each `Turn` message with `end_of_turn=true`:
- Take the audio chunk covering the turn's last word's `end` timestamp ("Moment A" — when the audio for the end of speech in that turn finished sending)
- Take the wall-clock time when the final-turn message arrived ("Moment B")
- **TTCT = Moment B − Moment A**

**Step 5 — Report:** Aggregate per-word emission latencies and per-turn TTCT values across all files into summary statistics (mean, median, P90, P99), write CSV/JSON output, and generate histogram plots for each metric.

### Project structure

```
src/latency_benchmark/
├── cli.py           # Click CLI — wires everything together
├── audio.py         # Loads WAV files, splits into fixed-duration PCM16 chunks
├── dataset.py       # Discovers audio+JSON pairs in a directory, loads ground truth
├── session.py       # WebSocket session — streams audio, collects transcripts with timestamps
├── benchmarker.py   # Core algorithm — word alignment (jiwer) + emission latency & TTCT computation
├── reporting.py     # Stats aggregation, CSV/JSON output, histogram plotting
└── models.py        # Dataclasses (AudioChunk, RunOutput, StreamingTranscript, etc.)

scripts/
└── download_sample_data.py  # Downloads LibriSpeech audio + alignments for benchmarking

data/sample/          # 100 pre-downloaded LibriSpeech test files with ground-truth timestamps
tests/                # Unit tests for each module + optional integration test
```

### Key design decisions

- **`time.monotonic()`** is used for all timestamps, avoiding issues with wall-clock adjustments
- **Text normalization** via `whisper-normalizer` ensures consistent comparison between ground truth and API output (handles casing, punctuation, number formatting)
- **Negative latencies are filtered out** (both emission and TTCT) — these occur when the API speculatively emits a word in a partial transcript before the audio covering that word has been fully sent (predictive decoding). This is valid API behavior but not a meaningful latency measurement.
- **`Terminate` is sent immediately after the last audio chunk** to close the session cleanly. This can short-circuit the API's normal silence-threshold wait at end-of-stream, so reported TTCT values should be read as a **lower bound** vs. what a continuously-streaming production agent would experience.
- **Per-file error handling** — if a WebSocket session fails for one file, the tool logs the error and continues with the remaining files

## Quick Start

```bash
# Install
pip install -e .

# Set up your API key
cp .env.example .env
# Edit .env and add your AssemblyAI API key

# Run with sample data
latency-benchmark --dataset ./data/sample --output ./results
```

The tool automatically loads your API key from the `.env` file. You can also
pass it directly via `--api-key` or the `ASSEMBLYAI_API_KEY` environment variable.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--api-key` | `$ASSEMBLYAI_API_KEY` | AssemblyAI API key |
| `--endpoint` | `wss://streaming.assemblyai.com/v3/ws` | WebSocket endpoint |
| `--dataset` | (required) | Directory with audio + JSON pairs |
| `--output` | `./results` | Base output directory; each run lands in `{output}/{speech_model}/{timestamp}/` |
| `--sample-rate` | `16000` | Audio sample rate (Hz) |
| `--chunk-size-ms` | `100` | Audio chunk duration (ms) |
| `--speech-model` | `u3-rt-pro` | Speech model |
| `--num-files` | `0` (all) | Limit number of files |
| `--plot/--no-plot` | `--plot` | Generate histogram |
| `--format` | `both` | Output format: csv, json, or both |

## Included Sample Data

The `data/sample/` directory contains 100 pre-downloaded audio files from
[LibriSpeech](https://www.openslr.org/12/) (`test-clean` split) — an open-source
corpus of read English speech derived from public domain audiobooks. Word-level
timestamps were generated by the [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/)
and sourced from [librispeech-alignments](https://github.com/CorentinJ/librispeech-alignments)
(available on [HuggingFace](https://huggingface.co/datasets/gilkeyio/librispeech-alignments)).

To download more files (or refresh the dataset):

```bash
python scripts/download_sample_data.py --num-files 200 --output data/expanded
```

This script downloads LibriSpeech audio from OpenSLR and fetches the corresponding
word-level alignments from the HuggingFace API. Requires `ffmpeg` for FLAC-to-WAV
conversion.

## Dataset Format

Place audio files (`.wav`) and matching JSON ground-truth files in a directory:

```
data/my-test/
├── call_001.wav
├── call_001.json
├── call_002.wav
└── call_002.json
```

Each JSON file contains word-level timestamps:

```json
[
  {"text": "you", "start": 1200, "end": 1350},
  {"text": "have", "start": 1400, "end": 1550},
  {"text": "reached", "start": 1600, "end": 1900}
]
```

## Output

Each run writes into its own directory under `--output`, partitioned by model
and timestamp so runs never overwrite each other:

```
{--output}/{speech_model}/{YYYY-MM-DDTHH-MM-SS}/
├── results.csv              # summary stats + per-word emission + per-turn TTCT
├── results.json              # machine-readable, with `emission_latency` and `ttct` sections
├── latency_histogram.png     # emission-latency distribution
└── ttct_histogram.png        # TTCT distribution
```

The tool also prints per-file stats and an aggregate summary table to the console.

## Testing specific endpoints

Test different AssemblyAI regions:

```bash
# US East
latency-benchmark --endpoint wss://streaming.use1.assemblyai.com/v3/ws --dataset ./data/sample

# US West
latency-benchmark --endpoint wss://streaming.usw2.assemblyai.com/v3/ws --dataset ./data/sample
```

## Improving production monitoring

For ongoing production monitoring (not just benchmarking), consider adding a
**Voice Activity Detector (VAD)** to your pipeline. See the
[VAD streaming example](examples/vad-streaming/) for a working implementation
using Silero VAD that:

1. Detects speech start/end events on each audio chunk
2. Measures `speech_start → first_transcript_received` for real-time TTFB
3. Measures `speech_end → next_transcript_received` for real-time emission latency approximation
4. Can be sampled on a subset of production traffic (5-10%) for continuous monitoring
