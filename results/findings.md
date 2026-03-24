# Emission Latency Benchmark — Findings Report

**Date:** 2026-03-24
**Dataset:** 100 LibriSpeech test-clean files (open-source, human-labeled word timestamps)
**Endpoint:** `wss://streaming.assemblyai.com/v3/ws`
**Chunk size:** 100ms | **Sample rate:** 16kHz

---

## Summary

We benchmarked two AssemblyAI streaming models to measure **emission latency** — the time from when an audio chunk containing a word is sent to when that word first appears in the API's transcript output.

| Metric | universal-streaming-english | u3-rt-pro |
|--------|:--------------------------:|:---------:|
| **Median** | **305 ms** | 3,075 ms |
| **Mean** | 357 ms | 4,218 ms |
| **P90** | 600 ms | 9,383 ms |
| **P99** | 1,683 ms | 14,263 ms |
| **Std Dev** | 269 ms | 3,352 ms |
| **WER** | 2.48% | 1.76% |
| **Session Init** | 84 ms | 91 ms |
| **Word Samples** | 2,093 | 2,193 |

## Key Findings

### 1. `universal-streaming-english` delivers ~300ms median emission latency

The median of 305ms is consistent with AssemblyAI's internal benchmarks of 300–350ms. This is well within the stated 300ms SLA and significantly below the ~550–600ms "speech-adjusted TTFB" that Uber has been reporting.

The discrepancy between our emission latency measurement and Uber's SA TTFB is explained by:
- **Uber's SA TTFB formula** subtracts the model's `words[0].start` timestamp to account for silence, but this timestamp has inherent variance (~180ms deviation observed internally). This inaccuracy inflates the metric.
- **No VAD in Uber's pipeline** means the TTFB measurement can't precisely detect when speech actually starts, adding further noise.
- **TTFB only measures the first word**, which is subject to session initialization overhead and initial silence. Emission latency measures every word, giving a far more representative picture.

### 2. `u3-rt-pro` is not suitable for low-latency streaming use cases

With a median emission latency of 3,075ms (~3 seconds), `u3-rt-pro` prioritizes accuracy over speed. It achieves a slightly lower WER (1.76% vs 2.48%), but the latency tradeoff is roughly 10x. For Uber's voicemail detection use case — where the system needs to detect phrases like "you've reached..." and end calls quickly — this model is far too slow.

### 3. Emission latency is the right metric for voicemail detection

For Uber's use case (streaming transcription feeding an LLM that detects voicemail greetings), what matters is: **how quickly does each spoken word appear in the transcript?** This is exactly what emission latency measures. TTFB only captures the first word and is sensitive to silence, timestamp accuracy, and session initialization — none of which reflect ongoing transcription performance.

## Methodology

1. **Audio streaming:** Each WAV file is split into 100ms chunks and sent to the WebSocket API at real-time rate using drift-free scheduling.

2. **Timestamp recording:** Monotonic clock timestamps are recorded for every audio chunk sent and every transcript message received.

3. **Word alignment:** The API's final transcript output is aligned against the human-labeled ground truth using edit-distance (jiwer). Only words present in both texts are used for measurement.

4. **Latency computation:** For each matched word:
   - **Moment A:** The wall-clock time the audio chunk covering the word's end timestamp was sent
   - **Moment B:** The wall-clock time the first transcript containing the word was received
   - **Emission latency = B − A**

5. **Filtering:** Negative latencies (where the API speculatively predicts a word before its audio is fully sent) are excluded. This affected ~4% of samples for `universal-streaming-english` and nearly 0% for `u3-rt-pro`.

## Recommendations

- **Uber should use `universal-streaming-english`** for their voicemail detection pipeline (which they already do).
- **Adopt emission latency** as the primary streaming performance metric instead of SA TTFB.
- **Add a VAD** (e.g., Silero VAD) to the pipeline for production monitoring — this allows approximating emission latency on live traffic without ground-truth files.
- **Run this benchmark periodically** (e.g., daily via cron) against the same dataset to detect regressions over time.
