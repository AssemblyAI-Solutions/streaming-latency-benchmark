# Streaming Latency Benchmark

Measure **emission latency** for AssemblyAI's streaming speech-to-text API.

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

### How it works

1. Stream audio to AssemblyAI's API at real-time rate, recording when each chunk is sent
2. Collect all transcript messages, recording when each is received
3. Align the API's output against human-labeled ground truth using edit distance
4. For each matched word: `latency = time_transcript_received - time_last_audio_chunk_sent`
5. Aggregate into mean, median, P90, P99

## Architecture

### How the benchmark works

The tool runs a pipeline for each audio file in the dataset:

```
Load audio        Stream to API       Align & compute       Report
+ ground truth -> via WebSocket    -> emission latency   -> results
                  at real-time        per word
```

**Step 1 — Load:** Read a WAV file and split it into fixed-duration chunks (default 100ms). Load the matching JSON file containing human-labeled word timestamps.

**Step 2 — Stream:** Open a WebSocket connection to AssemblyAI's streaming API. Three threads run concurrently:
- **Buffer thread** feeds audio chunks at real-time rate (drift-free scheduling)
- **Send thread** writes chunks to the WebSocket and records the monotonic timestamp of each send
- **Receive thread** reads transcript messages from the WebSocket and records the monotonic timestamp of each receive

**Step 3 — Align & compute:** The API's transcript won't perfectly match ground truth (some words may be missing, substituted, or added). The tool uses [jiwer](https://github.com/jitsi/jiwer) to perform word-level edit-distance alignment between the normalized ground truth and the normalized API output, keeping only words that match in both. For each matched word:
- Find the audio chunk that covers the word's end timestamp ("Moment A" — when all audio for this word was sent)
- Find the first transcript message containing this word ("Moment B" — when the API first returned this word)
- **Emission latency = Moment B − Moment A**

**Step 4 — Report:** Aggregate per-word latencies across all files into summary statistics (mean, median, P90, P99), write CSV/JSON output, and generate a histogram plot.

### Project structure

```
src/latency_benchmark/
├── cli.py           # Click CLI — wires everything together
├── audio.py         # Loads WAV files, splits into fixed-duration PCM16 chunks
├── dataset.py       # Discovers audio+JSON pairs in a directory, loads ground truth
├── session.py       # WebSocket session — streams audio, collects transcripts with timestamps
├── benchmarker.py   # Core algorithm — word alignment (jiwer) + emission latency computation
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
- **Negative latencies are filtered out** — these occur when the API speculatively emits a word in a partial transcript before the audio covering that word has been fully sent (predictive decoding). This is valid API behavior but not a meaningful latency measurement.
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
| `--output` | `./results` | Output directory |
| `--sample-rate` | `16000` | Audio sample rate (Hz) |
| `--chunk-size-ms` | `100` | Audio chunk duration (ms) |
| `--speech-model` | `u3-rt-pro` | Speech model (see note below) |
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

The tool produces:

- **Console output**: Per-file stats and aggregate summary
- **results.csv**: Summary stats + per-word latency values
- **results.json**: Machine-readable results
- **latency_histogram.png**: Distribution plot

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
**Voice Activity Detector (VAD)** to your pipeline:

1. **Silero VAD** or **WebRTC VAD** can detect speech start/end events
2. Use `speech_start → first_transcript_received` for real-time TTFB approximation
3. Use `speech_end → last_partial_received` for real-time emission latency approximation
4. Run on a subset of production traffic for continuous monitoring
