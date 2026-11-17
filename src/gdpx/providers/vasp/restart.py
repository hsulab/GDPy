"""Electronic checkpoints and opt-in graceful stopping for VASP relaxation."""
import contextlib
from collections import deque
import datetime
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import threading
import time


GRACEFUL_STOP_DEFAULTS = {
    "enabled": False,
    "margin_seconds": 7200,
    "poll_interval_seconds": 30,
    "checkpoint_margin_seconds": 300,
    "safety_factor": 1.5,
    "window": 5,
    "min_samples": 2,
}


def validate_graceful_stop(options):
    from collections.abc import Mapping
    if not isinstance(options, Mapping):
        raise ValueError("setup.graceful_stop must be a mapping.")
    unknown = set(options) - GRACEFUL_STOP_DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown setup.graceful_stop options: {sorted(unknown)}")
    result = {**GRACEFUL_STOP_DEFAULTS, **options}
    if not isinstance(result["enabled"], bool):
        raise ValueError("setup.graceful_stop.enabled must be a boolean.")
    for key in ("margin_seconds", "poll_interval_seconds", "checkpoint_margin_seconds", "safety_factor"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"setup.graceful_stop.{key} must be a finite positive number.")
    if result["safety_factor"] < 1:
        raise ValueError("setup.graceful_stop.safety_factor must be at least 1.")
    for key in ("window", "min_samples"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f"setup.graceful_stop.{key} must be a positive integer.")
    if result["min_samples"] > result["window"]:
        raise ValueError("setup.graceful_stop.min_samples must not exceed window.")
    return result


def restore_electronic_checkpoint(calc, checkpoint, directory):
    """Restore only the current geometry's checkpoint, never an older run."""
    checkpoint, directory = Path(checkpoint), Path(directory)
    present = {}
    for name in ("WAVECAR", "CHGCAR"):
        source = checkpoint / name
        present[name] = source.is_file() and source.stat().st_size > 0
        if present[name]:
            shutil.copy2(source, directory / name)
    calc.set(
        istart=1 if present["WAVECAR"] else 0,
        icharg=1 if present["CHGCAR"] else (0 if present["WAVECAR"] else 2),
    )
    return present


def slurm_end_time():
    """Read scheduler metadata, not user tuning, from the allocation."""
    value = os.environ.get("SLURM_JOB_END_TIME")
    if value:
        return float(value)
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("VASP graceful stop requires a Slurm job deadline.")
    result = subprocess.run(
        ["scontrol", "show", "job", "-o", job_id],
        check=True, capture_output=True, text=True, timeout=15,
    )
    match = re.search(r"\bEndTime=(\S+)", result.stdout)
    if not match or match[1] in {"Unknown", "N/A", "None"}:
        raise RuntimeError("Cannot determine Slurm job EndTime for VASP.")
    return datetime.datetime.fromisoformat(match[1]).timestamp()


class OutcarTimings:
    """Tail complete LOOP+ records with bounded memory; omit the first cycle."""
    pattern = re.compile(
        rb"^\s*LOOP\+:.*?real time\s+([0-9.+EeDd-]+)\s*$"
    )

    def __init__(self, path, window):
        self.path = Path(path)
        self.samples = deque(maxlen=window)
        self.offset = 0
        self.identity = None
        self.partial = b""
        self.cycles = 0
        self.last_completion = None

    def poll(self, now):
        records = []
        try:
            stream = self.path.open("rb")
        except FileNotFoundError:
            self.offset = 0
            self.identity = None
            self.partial = b""
            self.cycles = 0
            self.samples.clear()
            self.last_completion = None
            return records
        with stream:
            stat = os.fstat(stream.fileno())
            identity = (stat.st_dev, stat.st_ino)
            if self.identity != identity or stat.st_size < self.offset:
                self.offset = 0
                self.partial = b""
                self.cycles = 0
                self.samples.clear()
                self.last_completion = None
            self.identity = identity
            stream.seek(self.offset)
            # Read only the snapshot available at this poll, not an endless producer.
            remaining = stat.st_size - self.offset
            while remaining > 0:
                chunk = stream.read(min(remaining, 1024 * 1024))
                if not chunk:
                    break
                remaining -= len(chunk)
                self.offset += len(chunk)
                lines = (self.partial + chunk).split(b"\n")
                self.partial = lines.pop()[-65536:]
                for line in lines:
                    match = self.pattern.match(line)
                    if not match:
                        continue
                    try:
                        seconds = float(match[1].replace(b"D", b"E").replace(b"d", b"e"))
                    except ValueError:
                        continue
                    if not math.isfinite(seconds) or seconds <= 0:
                        continue
                    self.cycles += 1
                    if self.cycles > 1:
                        self.samples.append(seconds)
                    records.append((self.cycles, seconds))
                    self.last_completion = now
        return records

    def reserve(self, now, options):
        if len(self.samples) < options["min_samples"]:
            return options["margin_seconds"], "fallback"
        # Do not subtract progress: reserve a whole cycle even if the next one
        # started between polls. Inflate the estimate if the current cycle stalls.
        elapsed = max(0.0, now - self.last_completion)
        estimate = max(
            options["safety_factor"] * statistics.mean(self.samples),
            max(self.samples),
            options["safety_factor"] * elapsed,
        )
        return (
            estimate + options["checkpoint_margin_seconds"] + options["poll_interval_seconds"],
            "OUTCAR",
        )


@contextlib.contextmanager
def graceful_stop(directory, log=print, **settings):
    """Watch OUTCAR and request an ionic-step stop; disabled by default."""
    options = validate_graceful_stop(settings)
    if not options["enabled"]:
        yield
        return
    remaining = slurm_end_time() - time.time()
    if remaining <= options["margin_seconds"]:
        raise RuntimeError("Insufficient allocation time for the VASP fallback stop margin.")
    deadline = time.monotonic() + remaining
    stopcar = Path(directory) / "STOPCAR"
    if stopcar.exists():
        raise RuntimeError(f"Existing STOPCAR must be reviewed before running: {stopcar}")
    timings = OutcarTimings(Path(directory) / "OUTCAR", options["window"])
    finished = threading.Event()
    created = []

    def watch():
        while not finished.is_set():
            now = time.monotonic()
            try:
                records = timings.poll(now)
            except OSError as exc:
                log(f"Could not read OUTCAR timings: {exc}; using fallback margin.")
                records = []
                timings.samples.clear()
            for cycle, seconds in records:
                suffix = " (excluded from estimate)" if cycle == 1 else ""
                log(f"VASP SCF cycle {cycle}: {seconds:.1f}s{suffix}")
            reserve, source = timings.reserve(now, options)
            remaining = deadline - time.monotonic()
            if remaining <= reserve:
                if finished.is_set():
                    return
                try:
                    with stopcar.open("x") as stream:
                        stream.write("LSTOP = .TRUE.\n")
                    created.append(True)
                    log(
                        f"Requested VASP graceful stop: {remaining:.0f}s remaining, "
                        f"{reserve:.0f}s reserve ({source}); waiting for ionic step and checkpoint output."
                    )
                    return
                except FileExistsError:
                    return  # Preserve a user's stop request.
                except OSError as exc:
                    log(f"Failed to write VASP STOPCAR: {exc}; retrying.")
                    finished.wait(options["poll_interval_seconds"])
            else:
                finished.wait(min(options["poll_interval_seconds"], remaining - reserve))

    watcher = threading.Thread(target=watch, name="vasp-outcar-watcher", daemon=True)
    log("VASP OUTCAR watcher enabled; first SCF cycle excluded from timing estimate.")
    watcher.start()
    try:
        yield
    finally:
        finished.set()
        watcher.join()
        if created and stopcar.exists() and stopcar.read_text() == "LSTOP = .TRUE.\n":
            stopcar.unlink()
