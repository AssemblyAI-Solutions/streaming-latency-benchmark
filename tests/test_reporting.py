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
    assert "emission_mean" in content
    assert "emission_median" in content


def test_write_csv_with_ttct(tmp_path, sample_latencies):
    path = str(tmp_path / "results.csv")
    ttct = [1100, 1180, 1250, 1300, 1400]
    write_csv(path, sample_latencies, session_init_ms=150, wer=0.05, ttct_ms=ttct)
    assert os.path.exists(path)
    with open(path) as f:
        content = f.read()
    assert "ttct_mean" in content
    assert "turn_index,ttct_ms" in content


def test_write_json(tmp_path, sample_latencies):
    path = str(tmp_path / "results.json")
    write_json(path, sample_latencies, session_init_ms=150, wer=0.05)
    assert os.path.exists(path)
    with open(path) as f:
        data = json.load(f)
    assert "emission_latency" in data
    assert "stats" in data["emission_latency"]
    assert "per_word_latencies_ms" in data["emission_latency"]
    assert "ttct" not in data


def test_write_json_with_ttct(tmp_path, sample_latencies):
    path = str(tmp_path / "results.json")
    ttct = [1100, 1180, 1250, 1300, 1400]
    write_json(path, sample_latencies, session_init_ms=150, wer=0.05, ttct_ms=ttct)
    with open(path) as f:
        data = json.load(f)
    assert "ttct" in data
    assert data["ttct"]["per_turn_ttct_ms"] == ttct
    assert data["ttct"]["stats"]["count"] == 5
