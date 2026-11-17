import threading
import time
from unittest.mock import patch

import pytest

from gdpx.providers.vasp import restart


def loop(seconds):
    return f"     LOOP+: cpu time 12.0: real time {seconds}\n"


def test_incremental_complete_records_exclude_first(tmp_path):
    path = tmp_path / "OUTCAR"
    reader = restart.OutcarTimings(path, window=2)
    assert reader.poll(0) == []
    path.write_text(loop(1000) + " LOOP: cpu time 5: real time 6\n" + loop(10)[:-1])
    assert reader.poll(1) == [(1, 1000)]
    assert list(reader.samples) == []
    assert reader.poll(2) == []
    with path.open("a") as stream:
        stream.write("\n" + loop("2.0D+01") + loop(30))
    assert reader.poll(3) == [(2, 10), (3, 20), (4, 30)]
    assert list(reader.samples) == [20, 30]
    assert reader.poll(4) == []


@pytest.mark.parametrize("replace", [False, True])
def test_outcar_replacement_or_truncation_resets_history(tmp_path, replace):
    path = tmp_path / "OUTCAR"
    path.write_text(loop(1000) + loop(200) + loop(300))
    reader = restart.OutcarTimings(path, window=5)
    reader.poll(0)
    if replace:
        path.rename(tmp_path / "old")
    path.write_text(loop(10))
    assert reader.poll(1) == [(1, 10)]
    assert list(reader.samples) == []


def test_invalid_timings_are_ignored_and_partial_memory_is_bounded(tmp_path):
    path = tmp_path / "OUTCAR"
    path.write_text(loop("-2") + loop("1e999") + loop("bad") + loop(0) + "x" * 2000000)
    reader = restart.OutcarTimings(path, window=5)
    assert reader.poll(0) == []
    assert len(reader.partial) <= 65536
    assert reader.offset == path.stat().st_size


def test_adaptive_reserve_and_long_current_cycle(tmp_path):
    path = tmp_path / "OUTCAR"
    reader = restart.OutcarTimings(path, window=5)
    options = restart.validate_graceful_stop({})
    assert reader.reserve(0, options) == (7200, "fallback")
    path.write_text(loop(10000) + loop(100))
    reader.poll(0)
    assert reader.reserve(0, options) == (7200, "fallback")
    with path.open("a") as stream:
        stream.write(loop(200))
    reader.poll(10)
    # max(1.5 * mean(100,200), max(100,200)) + 300 I/O + 30 polling.
    assert reader.reserve(10, options) == (555, "OUTCAR")
    # A cycle that runs longer than the history increases the reserve.
    assert reader.reserve(410, options) == (930, "OUTCAR")


def test_watcher_stops_from_appended_timings_before_fallback(tmp_path):
    logs = []
    requested = threading.Event()
    def log(message):
        logs.append(message)
        if message.startswith("Requested"):
            requested.set()
    settings = dict(
        enabled=True, margin_seconds=10, poll_interval_seconds=0.01,
        checkpoint_margin_seconds=1, min_samples=2,
    )
    with patch.object(restart, "slurm_end_time", return_value=time.time() + 60):
        with restart.graceful_stop(tmp_path, log, **settings):
            path = tmp_path / "OUTCAR"
            path.write_text(loop(1000) + loop(50))
            assert not requested.wait(0.05)  # First cycle is excluded; only one sample.
            with path.open("a") as stream:
                stream.write(loop(50))
            assert requested.wait(3)
            assert (tmp_path / "STOPCAR").read_text() == "LSTOP = .TRUE.\n"
    assert not (tmp_path / "STOPCAR").exists()
    assert any("(OUTCAR)" in message for message in logs)
    assert not any(t.name == "vasp-outcar-watcher" for t in threading.enumerate())


def test_watcher_preserves_external_stopcar(tmp_path):
    with patch.object(restart, "slurm_end_time", return_value=time.time() + 60):
        with restart.graceful_stop(
            tmp_path, enabled=True, margin_seconds=10,
            poll_interval_seconds=0.01, checkpoint_margin_seconds=1,
        ):
            (tmp_path / "STOPCAR").write_text("LABORT = .TRUE.\n")
            (tmp_path / "OUTCAR").write_text(loop(1000) + loop(50) + loop(50))
            time.sleep(0.05)
    assert (tmp_path / "STOPCAR").read_text() == "LABORT = .TRUE.\n"


@pytest.mark.parametrize("options", [
    {"poll_interval_seconds": 0}, {"checkpoint_margin_seconds": -1},
    {"safety_factor": 0.9}, {"window": 0}, {"window": 2.5},
    {"min_samples": True}, {"window": 2, "min_samples": 3},
])
def test_invalid_adaptive_options(options):
    with pytest.raises(ValueError):
        restart.validate_graceful_stop(options)


def test_missing_outcar_discards_old_estimate(tmp_path):
    path = tmp_path / "OUTCAR"
    path.write_text(loop(1000) + loop(10) + loop(20))
    reader = restart.OutcarTimings(path, window=5)
    options = restart.validate_graceful_stop({})
    reader.poll(0)
    assert reader.reserve(0, options)[1] == "OUTCAR"
    path.unlink()
    assert reader.poll(1) == []
    assert reader.reserve(1, options) == (7200, "fallback")
