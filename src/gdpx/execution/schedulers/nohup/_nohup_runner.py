"""Standalone nohup supervisor; intentionally uses only the standard library."""

import argparse
import base64
import json
import os
import pathlib
import subprocess
import tempfile
import time
import traceback


def atomic_json(path, value):
    path = pathlib.Path(path)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = pathlib.Path(handle.name)
        json.dump(value, handle)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--context", required=True)
    args = parser.parse_args()
    context = json.loads(base64.urlsafe_b64decode(args.context).decode())
    attempt = pathlib.Path(context["attempt_directory"])
    atomic_json(attempt / "started.json", {"job_id": args.job_id, "pid": os.getpid()})
    exit_code = 1
    try:
        exit_code = subprocess.run(
            ["bash", "-l", context["script"]], cwd=pathlib.Path(context["script"]).parent,
        ).returncode
    except BaseException:
        traceback.print_exc()
    finally:
        atomic_json(attempt / "completed.json", dict(
            job_id=args.job_id, exit_code=exit_code, finished_at=time.time(),
        ))


if __name__ == "__main__":
    main()
