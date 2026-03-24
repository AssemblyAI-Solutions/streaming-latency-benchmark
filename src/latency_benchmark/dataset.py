import json
import os
from typing import List, Tuple

from latency_benchmark.models import TranscribedWord

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}


def discover_dataset(dataset_dir: str) -> List[Tuple[str, str]]:
    """Find audio+JSON pairs in a directory."""
    audio_files = sorted(
        f for f in os.listdir(dataset_dir)
        if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
    )

    pairs = []
    for audio_file in audio_files:
        base_name = os.path.splitext(audio_file)[0]
        json_file = f"{base_name}.json"
        json_path = os.path.join(dataset_dir, json_file)
        if os.path.exists(json_path):
            pairs.append((os.path.join(dataset_dir, audio_file), json_path))

    return pairs


def load_ground_truth(json_path: str) -> List[TranscribedWord]:
    """Load ground-truth word timestamps from a JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    return [
        TranscribedWord(text=entry["text"], start_ms=entry["start"], end_ms=entry["end"])
        for entry in data
    ]
