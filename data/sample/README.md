# Sample Data

This directory should contain audio files (`.wav`) paired with ground-truth
timestamp JSON files (same base name, `.json` extension).

## JSON format

Each JSON file is an array of word objects:

```json
[
  {"text": "hello", "start": 1200, "end": 1500},
  {"text": "world", "start": 1600, "end": 1900}
]
```

- `text`: The spoken word
- `start`: Word start time in milliseconds (from beginning of audio)
- `end`: Word end time in milliseconds

## Sourcing data

For benchmarking, use audio files with human-verified transcriptions and precise
word-level timestamps. The LibriSpeech dataset is a good open-source option.

AssemblyAI can provide a curated subset — contact your account team.
