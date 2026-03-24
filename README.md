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

## Quick Start

```bash
# Install
pip install -e .

# Run with sample data
latency-benchmark \
  --api-key YOUR_ASSEMBLYAI_API_KEY \
  --dataset ./data/sample \
  --output ./results

# Or use env var
export ASSEMBLYAI_API_KEY=your_key
latency-benchmark --dataset ./data/sample
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--api-key` | `$ASSEMBLYAI_API_KEY` | AssemblyAI API key |
| `--endpoint` | `wss://streaming.assemblyai.com` | WebSocket endpoint |
| `--dataset` | (required) | Directory with audio + JSON pairs |
| `--output` | `./results` | Output directory |
| `--sample-rate` | `16000` | Audio sample rate (Hz) |
| `--chunk-size-ms` | `100` | Audio chunk duration (ms) |
| `--speech-model` | (default) | Model override |
| `--num-files` | `0` (all) | Limit number of files |
| `--plot/--no-plot` | `--plot` | Generate histogram |
| `--format` | `both` | Output format: csv, json, or both |

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
latency-benchmark --endpoint wss://streaming.use1.assemblyai.com --dataset ./data/sample

# US West
latency-benchmark --endpoint wss://streaming.usw2.assemblyai.com --dataset ./data/sample
```

## Improving production monitoring

For ongoing production monitoring (not just benchmarking), consider adding a
**Voice Activity Detector (VAD)** to your pipeline:

1. **Silero VAD** or **WebRTC VAD** can detect speech start/end events
2. Use `speech_start → first_transcript_received` for real-time TTFB approximation
3. Use `speech_end → last_partial_received` for real-time emission latency approximation
4. Run on a subset of production traffic for continuous monitoring
