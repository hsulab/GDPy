"""Read-only, host-wide discovery of GDPy simulation processes."""

import base64
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass

try:
    import pwd
except ImportError:  # Keep ordinary GDPy imports usable on non-POSIX hosts.
    pwd = None


@dataclass(frozen=True)
class Process:
    pid: int
    ppid: int
    uid: int
    state: str
    elapsed: str
    command: str

    @property
    def active(self):
        return not self.state.startswith(("Z", "X"))


def encode_context(context):
    return base64.urlsafe_b64encode(json.dumps(context).encode()).decode()


def decode_context(value):
    return json.loads(base64.urlsafe_b64decode(value).decode())


def parse_processes(output):
    processes = []
    for line in output.splitlines():
        fields = line.split(None, 5)
        if len(fields) != 6:
            continue
        try:
            processes.append(Process(*(int(value) for value in fields[:3]), *fields[3:]))
        except ValueError:
            continue
    return processes


def snapshot_processes():
    if os.name != "posix":
        raise RuntimeError("GDPy process discovery requires a POSIX host.")
    try:
        result = subprocess.run(
            ["ps", "-axww", "-o", "pid=,ppid=,uid=,stat=,etime=,args="],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Unable to query host processes with ps: {error}") from error
    return parse_processes(result.stdout)


def _launcher_arguments(command):
    """Recognize executables, rather than arbitrary command substrings."""
    try:
        args = shlex.split(command)
    except ValueError:
        return []
    if not args:
        return []
    executable = pathlib.Path(args[0]).name
    if executable == "gdp":
        return args[1:]
    if re.fullmatch(r"(?:python(?:\d+(?:\.\d+)*)?|pypy\d*)", executable):
        if len(args) > 1 and pathlib.Path(args[1]).name == "gdp":
            return args[2:]
        if args[1:3] == ["-m", "gdpx.main"]:
            return args[3:]
    return []


def supervisor_context(process):
    try:
        # ps does not quote argv: allow spaces in installation paths, while
        # keeping the interpreter and supervisor anchored at the command start.
        match = re.fullmatch(
            r"(?:.*?/)?(?:python(?:\d+(?:\.\d+)*)?|pypy\d*) "
            r"(?:.*?/)?_nohup_runner\.py --job-id (\S+) --context (\S+)",
            process.command,
        )
        if match is None:
            return None
        job_id, encoded = match.groups()
        context = decode_context(encoded)
        if not isinstance(context, dict) or not job_id.startswith("nohup-"):
            return None
        return dict(context, job_id=job_id)
    except (ValueError, IndexError, UnicodeError, TypeError):
        return None


def _is_simulation(command):
    args = _launcher_arguments(command)
    if "--help" in args or "-h" in args:
        return False
    # Skip the global CLI options without interpreting their values as actions.
    value_options = {"-rs", "--random_seed", "-d", "--directory", "-r", "--runtime",
                     "-nj", "--n_jobs", "--log"}
    index = 0
    while index < len(args) and args[index].startswith("-"):
        option = args[index]
        index += 2 if option in value_options else 1
    if index >= len(args):
        return False
    action, remaining = args[index], args[index + 1:]
    if action == "explore":
        return bool(remaining)
    if action == "workflow":
        return bool(remaining) and remaining[0] == "run"
    if action == "compute":
        value_options = {"-b", "--batch", "--plan", "--job", "--task", "--worker"}
        index = 0
        while index < len(remaining) and remaining[index].startswith("-"):
            index += 2 if remaining[index] in value_options else 1
        return index >= len(remaining) or remaining[index] not in {
            "prepare", "submit", "status", "resubmit", "collect",
        }
    return False


def process_directory(pid):
    try:
        if sys.platform.startswith("linux"):
            return os.readlink(f"/proc/{pid}/cwd")
        if sys.platform == "darwin":
            result = subprocess.run(
                ["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
                capture_output=True, text=True, timeout=2,
            )
            return next((line[1:] for line in result.stdout.splitlines() if line.startswith("n")), None)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def running_jobs(*, all_users=False, processes=None):
    processes = snapshot_processes() if processes is None else processes
    by_pid = {process.pid: process for process in processes}
    contexts = {process.pid: supervisor_context(process) for process in processes if process.active}
    candidates = {pid for pid, context in contexts.items() if context is not None}
    candidates.update(process.pid for process in processes if process.active and _is_simulation(process.command))
    managed = {pid for pid, context in contexts.items() if context is not None}
    rows = []
    for pid in sorted(candidates):
        process = by_pid[pid]
        if not all_users and process.uid != os.getuid():
            continue
        # Keep each allocation, but suppress its tasks and nested CLI helpers.
        if pid not in managed:
            ancestor = process.ppid
            seen = {pid}
            while ancestor in by_pid and ancestor not in seen:
                seen.add(ancestor)
                if ancestor in candidates:
                    break
                ancestor = by_pid[ancestor].ppid
            if ancestor in candidates:
                continue
        context = contexts.get(pid) or {}
        try:
            user = pwd.getpwuid(process.uid).pw_name if pwd else str(process.uid)
        except KeyError:
            user = str(process.uid)
        rows.append(dict(
            job_id=context.get("job_id", f"pid:{pid}"), pid=pid, uid=process.uid, user=user,
            state=process.state, elapsed=process.elapsed,
            directory=context.get("directory") or process_directory(pid),
            command=(f"bash -l {shlex.quote(context['script'])}" if context.get("script") else process.command),
        ))
    return rows
