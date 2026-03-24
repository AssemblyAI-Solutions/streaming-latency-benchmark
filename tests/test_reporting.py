import json
import os

import pytest

from latency_benchmark.reporting import compute_stats, write_csv, write_json


@pytest.fixture
def sample_latencies():
    return [250, 300, 280, 320, 350, 400, 500, 270, 290, 310]


def test_compute_stats(sample_latencies):
    stats = compute_stats(sample_latencies)
    assert "mean" in stats
    assert "median" in stats
    assert "p90" in stats
    assert "p99" in stats
    assert "std_dev" in stats
    assert "count" in stats
    assert stats["count"] == 10
    assert 250 <= stats["mean"] <= 400
    assert 250 <= stats["median"] <= 350


def test_compute_stats_empty():
    stats = compute_stats([])
    assert stats["count"] == 0
    assert stats["mean"] == 0


def test_write_csv(tmp_path, sample_latencies):
    path = str(tmp_path / "results.csv")
    write_csv(path, sample_latencies, session_init_ms=150, wer=0.05)
    assert os.path.exists(path)
    with open(path) as f:
        content = f.read()
    assert "mean" in content
    assert "median" in content


def test_write_json(tmp_path, sample_latencies):
    path = str(tmp_path / "results.json")
    write_json(path, sample_latencies, session_init_ms=150, wer=0.05)
    assert os.path.exists(path)
    with open(path) as f:
        data = json.load(f)
    assert "stats" in data
    assert "per_word_latencies_ms" in data
