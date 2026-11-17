"""Electronic checkpoints and opt-in graceful stopping for VASP relaxation."""
import contextlib
import datetime
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time



def restore_electronic_checkpoint(calc, checkpoint, directory):
    """Restore only the current geometry's checkpoint, never an older run."""
    checkpoint, directory = Path(checkpoint), Path(directory)
    present = {}
    for name in ("WAVECAR", "CHGCAR"):
        source = checkpoint / name
        present[name] = source.is_file() and source.stat().st_size > 0
        if present[name]:
            shutil.copy2(source, directory / name)
    # A density checkpoint retains the magnetic density and PAW occupancies.
    calc.set(
        istart=1 if present["WAVECAR"] else 0,
        icharg=1 if present["CHGCAR"] else (0 if present["WAVECAR"] else 2),
    )
    return present


def slurm_end_time():
    """Read the allocation's absolute deadline; fail closed when opted in."""
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


@contextlib.contextmanager
def graceful_stop(directory, log=print, *, enabled=False, margin_seconds=7200):
    """Request LSTOP early enough to complete an ionic step and write files.

    Explicit executor input enables this feature; it is disabled by default.
    No electronic-step abort is used because it can leave unconverged forces.
    """
    if not enabled:
        yield
        return
    margin = margin_seconds
    delay = slurm_end_time() - time.time() - margin
    if delay <= 0:
        raise RuntimeError("Insufficient allocation time for the VASP stop margin.")
    stopcar = Path(directory) / "STOPCAR"
    if stopcar.exists():
        raise RuntimeError(f"Existing STOPCAR must be reviewed before running: {stopcar}")
    created = []

    def request_stop():
        try:
            # Do not overwrite a user's stop request.
            with stopcar.open("x") as stream:
                stream.write("LSTOP = .TRUE.\n")
            created.append(True)
            log("Requested VASP graceful stop (LSTOP); waiting for ionic step and checkpoint output.")
        except FileExistsError:
            pass
        except OSError as exc:
            log(f"Failed to write VASP STOPCAR: {exc}")

    timer = threading.Timer(delay, request_stop)
    timer.daemon = True
    log(f"VASP graceful stop scheduled in {delay:.0f}s; reserve {margin:.0f}s before Slurm deadline.")
    timer.start()
    try:
        yield
    finally:
        timer.cancel()
        timer.join()
        if created and stopcar.exists() and stopcar.read_text() == "LSTOP = .TRUE.\n":
            stopcar.unlink()
