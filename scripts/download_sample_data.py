"""Download a sample subset from LibriSpeech with word-level alignments for benchmarking.

Usage:
    python scripts/download_sample_data.py --num-files 20 --output data/sample

Downloads audio from LibriSpeech test-clean (OpenSLR) and word-level
alignments from HuggingFace API, then converts to our benchmark format
(WAV + JSON pairs). No torch/GPU required — only needs ffmpeg.
"""

import argparse
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlretrieve, urlopen


LIBRISPEECH_TEST_CLEAN_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
HF_API_URL = "https://datasets-server.huggingface.co/rows?dataset=gilkeyio%2Flibrispeech-alignments&config=default&split=test_clean&offset={offset}&length={length}"


def download_if_missing(url: str, dest: str, label: str) -> None:
    if not os.path.exists(dest):
        print(f"Downloading {label}...")
        urlretrieve(url, dest)
        print(f"  Done.")


def get_audio_map(cache_dir: str) -> dict:
    """Download and extract LibriSpeech test-clean. Returns utterance_id -> flac_path."""
    tar_path = os.path.join(cache_dir, "test-clean.tar.gz")
    extract_dir = os.path.join(cache_dir, "audio")

    download_if_missing(LIBRISPEECH_TEST_CLEAN_URL, tar_path, "LibriSpeech test-clean (~350MB)")

    if not os.path.exists(extract_dir):
        print("Extracting audio...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_dir, filter="data")

    audio_map = {}
    for flac_file in Path(extract_dir).rglob("*.flac"):
        audio_map[flac_file.stem] = str(flac_file)

    print(f"Found {len(audio_map)} audio files.")
    return audio_map


def fetch_alignments_from_hf(num_needed: int, min_words: int) -> list:
    """Fetch alignment rows from HuggingFace datasets API (no audio decoding needed)."""
    print(f"Fetching alignments from HuggingFace API...")
    results = []
    offset = 0
    batch_size = 100

    while len(results) < num_needed:
        url = HF_API_URL.format(offset=offset, length=batch_size)
        with urlopen(url) as resp:
            data = json.loads(resp.read())

        rows = data.get("rows", [])
        if not rows:
            break

        for row in rows:
            r = row["row"]
            words = r.get("words", [])
            aligned = [w for w in words if w.get("word", "") != "<unk>"]
            if len(aligned) >= min_words:
                results.append({
                    "id": r["id"],
                    "words": aligned,
                })
                if len(results) >= num_needed:
                    break

        offset += batch_size
        print(f"  Fetched {offset} rows, {len(results)} usable so far...")

    return results


def flac_to_wav(flac_path: str, wav_path: str) -> None:
    """Convert FLAC to 16-bit PCM WAV at 16kHz mono using ffmpeg."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", flac_path, "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", "-loglevel", "error", wav_path],
        check=True,
    )


def save_ground_truth(words: list, output_path: str) -> None:
    """Save word-level timestamps as JSON (seconds -> milliseconds)."""
    ground_truth = [
        {
            "text": w["word"],
            "start": round(w["start"] * 1000),
            "end": round(w["end"] * 1000),
        }
        for w in words
    ]
    with open(output_path, "w") as f:
        json.dump(ground_truth, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Download LibriSpeech sample data for benchmarking")
    parser.add_argument("--num-files", type=int, default=20, help="Number of audio files to download")
    parser.add_argument("--output", type=str, default="data/sample", help="Output directory")
    parser.add_argument("--min-words", type=int, default=5, help="Minimum words per utterance")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory for downloads")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    cache_dir = args.cache_dir or os.path.join(tempfile.gettempdir(), "librispeech-cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Step 1: Get alignments from HuggingFace API (lightweight, no audio decoding)
    alignments = fetch_alignments_from_hf(args.num_files, args.min_words)

    # Step 2: Download LibriSpeech audio
    audio_map = get_audio_map(cache_dir)

    # Step 3: Match and convert
    saved = 0
    for entry in alignments:
        utterance_id = entry["id"]
        if utterance_id not in audio_map:
            print(f"  Skipping {utterance_id} (audio not found)")
            continue

        wav_path = os.path.join(args.output, f"{utterance_id}.wav")
        json_path = os.path.join(args.output, f"{utterance_id}.json")

        flac_to_wav(audio_map[utterance_id], wav_path)
        save_ground_truth(entry["words"], json_path)

        saved += 1
        print(f"  [{saved}/{args.num_files}] {utterance_id} ({len(entry['words'])} words)")

    print(f"\nDone! Saved {saved} files to {args.output}/")
    print(f"Each file is a .wav + .json pair ready for: latency-benchmark --dataset {args.output}")


if __name__ == "__main__":
    main()
