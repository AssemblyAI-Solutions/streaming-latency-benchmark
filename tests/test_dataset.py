import json
import struct
import wave

import pytest

from latency_benchmark.dataset import discover_dataset, load_ground_truth
from latency_benchmark.models import TranscribedWord


@pytest.fixture
def dataset_dir(tmp_path):
    """Create a minimal dataset with 2 audio+json pairs."""
    for i in range(1, 3):
        filepath = tmp_path / f"sample_{i:03d}.wav"
        with wave.open(str(filepath), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(struct.pack("<160h", *([0] * 160)))

        gt = [
            {"text": "hello", "start": 500, "end": 800},
            {"text": "world", "start": 900, "end": 1200},
        ]
        (tmp_path / f"sample_{i:03d}.json").write_text(json.dumps(gt))

    return tmp_path


def test_discover_dataset(dataset_dir):
    pairs = discover_dataset(str(dataset_dir))
    assert len(pairs) == 2
    for audio_path, json_path in pairs:
        assert audio_path.endswith(".wav")
        assert json_path.endswith(".json")


def test_discover_skips_unmatched(dataset_dir):
    """An audio file without a matching JSON is skipped."""
    filepath = dataset_dir / "orphan.wav"
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<160h", *([0] * 160)))
    pairs = discover_dataset(str(dataset_dir))
    assert len(pairs) == 2


def test_load_ground_truth(dataset_dir):
    pairs = discover_dataset(str(dataset_dir))
    words = load_ground_truth(pairs[0][1])
    assert len(words) == 2
    assert all(isinstance(w, TranscribedWord) for w in words)
    assert words[0].text == "hello"
    assert words[0].start_ms == 500
    assert words[0].end_ms == 800
