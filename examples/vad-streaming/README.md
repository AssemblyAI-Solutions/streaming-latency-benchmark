# VAD Streaming Example

Streams audio to AssemblyAI's streaming API with [Silero VAD](https://github.com/snakers4/silero-vad) running alongside to detect speech boundaries. Computes VAD-anchored latency metrics that are more accurate than TTFB.

## What it measures

- **VAD TTFB:** Time from when VAD detects speech start to when the first transcript is received
- **VAD Emission Latency:** Time from when VAD detects speech end to when the last partial transcript is received
- **Session Init Latency:** WebSocket connection time

## Setup

```bash
pip install -r requirements.txt

# Set up API key
cp ../../.env.example .env
# Edit .env with your AssemblyAI API key
# Or copy from the root: cp ../../.env .
```

## Usage

```bash
# Run against the benchmark sample data (5 files)
python run_vad_benchmark.py --dataset ../../data/sample --num-files 5

# Run against all 100 sample files
python run_vad_benchmark.py --dataset ../../data/sample --num-files 100

# Specify a model
python run_vad_benchmark.py --dataset ../../data/sample --speech-model universal-streaming-english
```

## How it works

1. **VAD analysis:** Runs Silero VAD on the audio offline (before streaming) to find speech start/end timestamps. Uses 512-sample frames as required by Silero at 16kHz.

2. **Audio streaming:** Sends audio chunks to AssemblyAI at real-time rate via WebSocket, recording monotonic timestamps for each chunk sent and each transcript received.

3. **Metric computation:** Maps VAD speech events to the audio timeline, finds the corresponding chunk send timestamps, and computes latency as the difference between transcript receive time and chunk send time.

## Notes

- Silero VAD requires exactly 512-sample frames at 16kHz (32ms). The script handles splitting 100ms audio chunks into 512-sample frames internally.
- The VAD runs on the full audio before streaming — in production you'd run it on each chunk as it arrives.
- Requires PyTorch (CPU-only is fine, no GPU needed).
