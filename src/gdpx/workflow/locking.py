"""Exclusive controller ownership of a workflow directory."""

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import socket


@contextmanager
def workflow_lock(directory):
    """Reject competing controllers; the OS releases ownership on exit/crash."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # Keep this inode permanently: unlinking it could allow two owners.
    path = directory / ".workflow.lock"
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.seek(0)
            owner = stream.read().strip() or "another controller"
            raise RuntimeError(
                f"Workflow is already running at {directory} ({owner}). "
                "Wait for that controller to exit before resuming."
            ) from exc
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(f"host={socket.gethostname()} pid={os.getpid()}\n")
            stream.flush()
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
