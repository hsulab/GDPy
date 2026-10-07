"""Standalone nohup supervisor; intentionally uses only the standard library."""

import argparse
import base64
import json
import os
import pathlib
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--context", required=True)
    args = parser.parse_args()
    context = json.loads(base64.urlsafe_b64decode(args.context).decode())
    child = subprocess.Popen(
        ["bash", "-l", context["script"]], cwd=pathlib.Path(context["script"]).parent,
    )
    try:
        os.write(context["ready_fd"], b"R")
    except BrokenPipeError:
        pass  # Keep the allocation alive if its submitting process exits early.
    finally:
        os.close(context["ready_fd"])
    exit_code = child.wait()
    if exit_code:
        print(f"Nohup job {args.job_id} exited with status {exit_code}.", file=sys.stderr, flush=True)
    return exit_code if exit_code >= 0 else 128 - exit_code


if __name__ == "__main__":
    raise SystemExit(main())
