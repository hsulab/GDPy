"""SSH transport for direct and queue-based schedulers."""

from __future__ import annotations

import os
import errno
import pathlib
import shlex
import shutil
import stat
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Optional, Union

import paramiko

from .scheduler import BaseScheduler


def _should_sync_file(sftp: paramiko.SFTPClient, remote_file_path, local_file_path) -> bool:
    if not os.path.exists(local_file_path):
        return True
    remote_attr = sftp.lstat(remote_file_path)
    local_stat = os.stat(local_file_path)
    return remote_attr.st_size != local_stat.st_size or remote_attr.st_mtime != local_stat.st_mtime


def _sync_latest_recursive(sftp: paramiko.SFTPClient, remote_dir: str, local_dir: str, skipped_items,
                           protected_paths=()) -> int:
    files_synced = 0
    for item in sftp.listdir_attr(remote_dir):
        remote_item = os.path.join(remote_dir, item.filename)
        local_item = os.path.join(local_dir, item.filename)
        if item.filename in skipped_items or pathlib.Path(local_item).resolve() in protected_paths:
            continue
        if stat.S_ISREG(item.st_mode):
            os.makedirs(local_dir, exist_ok=True)
            if _should_sync_file(sftp, remote_item, local_item):
                sftp.get(remote_item, local_item)
                remote_attr = sftp.lstat(remote_item)
                os.utime(local_item, (remote_attr.st_atime, remote_attr.st_mtime))
                files_synced += 1
        elif stat.S_ISDIR(item.st_mode):
            files_synced += _sync_latest_recursive(sftp, remote_item, local_item, skipped_items, protected_paths)
    return files_synced


def _remove_outdated_recursive(
    sftp: paramiko.SFTPClient,
    remote_dir: str,
    local_dir: str,
    skipped_items: list[str],
    print_func=print,
    debug_func=print,
    protected_paths=(),
) -> int:
    del debug_func
    if not os.path.isdir(local_dir):
        return 0
    items_removed = 0
    for item in os.listdir(local_dir):
        remote_item = os.path.join(remote_dir, item)
        local_item = os.path.join(local_dir, item)
        if item in skipped_items or pathlib.Path(local_item).resolve() in protected_paths:
            continue
        try:
            sftp.stat(remote_item)
        except IOError:
            print_func(f"removing {local_item}")
            if os.path.isfile(local_item) or os.path.islink(local_item):
                os.remove(local_item)
            else:
                shutil.rmtree(local_item)
            items_removed += 1
            continue
        if os.path.isdir(local_item):
            items_removed += _remove_outdated_recursive(
                sftp, remote_item, local_item, skipped_items, print_func=print_func,
                protected_paths=protected_paths
            )
    return items_removed


class SshTransport(BaseScheduler):
    """Run a scheduler through SSH and synchronize its working tree."""

    transport_name = "ssh"

    def __init__(
        self,
        scheduler: BaseScheduler,
        hostname: str,
        remote_wdir: Union[str, pathlib.Path],
        *,
        ssh_client_factory: Optional[Callable[[], paramiko.SSHClient]] = None,
    ) -> None:
        if not isinstance(scheduler, BaseScheduler):
            raise TypeError("SshTransport requires a BaseScheduler instance.")
        if isinstance(scheduler, SshTransport):
            raise ValueError("SshTransport cannot wrap another SSH transport.")
        if not hostname:
            raise ValueError("SSH transport hostname cannot be empty.")
        remote_path = pathlib.PurePosixPath(str(remote_wdir))
        if not remote_path.is_absolute():
            raise ValueError("remote_wdir must be an absolute POSIX path.")
        self.scheduler = scheduler
        self.hostname = hostname
        self.remote_wdir = remote_path
        self.local_root: Optional[pathlib.Path] = None
        self.output_root: Optional[pathlib.Path] = None
        self.sync_excludes: set[pathlib.Path] = set()
        # Optional exact exclusions for callers sharing a staging root. The
        # default retains legacy name-based job-store exclusions.
        self.staging_excludes: Optional[set[pathlib.Path]] = None
        self._ssh_client_factory = ssh_client_factory or paramiko.SSHClient

    @property
    def is_direct(self) -> bool:
        return self.scheduler.is_direct

    @property
    def name(self) -> str:
        return self.scheduler.name

    @property
    def script(self) -> pathlib.Path:
        return self.scheduler.script

    @script.setter
    def script(self, value: Union[str, pathlib.Path]) -> None:
        self.scheduler.script = value

    @property
    def job_name(self) -> str:
        return self.scheduler.job_name

    @job_name.setter
    def job_name(self, value: str) -> None:
        self.scheduler.job_name = value

    @property
    def environs(self):
        return self.scheduler.environs

    @environs.setter
    def environs(self, value) -> None:
        self.scheduler.environs = value

    @property
    def machine_prefix(self) -> str:
        return self.scheduler.machine_prefix

    @machine_prefix.setter
    def machine_prefix(self, value: str) -> None:
        self.scheduler.machine_prefix = value

    @property
    def concurrent_tasks(self) -> int:
        return self.scheduler.concurrent_tasks

    @property
    def user_commands(self) -> str:
        return self.scheduler.user_commands

    @user_commands.setter
    def user_commands(self, value: str) -> None:
        self.scheduler.user_commands = value

    @property
    def parameters(self) -> dict:
        return self.scheduler.parameters

    @property
    def submit_timeout(self) -> float:
        return self.scheduler.submit_timeout

    @property
    def is_dry_run(self) -> bool:
        return self.scheduler.is_dry_run

    def __str__(self) -> str:
        return str(self.scheduler)

    def set(self, **kwargs) -> None:
        self.scheduler.set(**kwargs)

    def write(self) -> None:
        self.scheduler.write()

    def build_submit_command(self, script_name: str) -> str:
        return self.scheduler.build_submit_command(script_name)

    def parse_submit_output(self, output: str) -> str:
        return self.scheduler.parse_submit_output(output)

    def is_finished_from_output(self, output: str) -> bool:
        return self.scheduler.is_finished_from_output(output)

    def as_dict(self) -> dict:
        data = self.scheduler.as_dict()
        data["transport"] = {
            "provider": "ssh",
            "parameters": {
                "hostname": self.hostname,
                "remote_wdir": str(self.remote_wdir),
            },
        }
        return data

    def _client(self):
        client = self._ssh_client_factory()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        password = os.environ.get(f"{self.hostname.upper()}_PASSWORD")
        client.connect(hostname=self.hostname, password=password)
        return client

    def _roots(self) -> tuple[pathlib.Path, pathlib.PurePosixPath, pathlib.Path]:
        if pathlib.PurePath(self.job_name).name != self.job_name or self.job_name in {"", ".", ".."}:
            raise ValueError(f"Unsafe remote job name {self.job_name!r}.")
        local_root = pathlib.Path(self.local_root or self.script.parent).resolve()
        script = self.script.resolve()
        try:
            script_relative = script.relative_to(local_root)
        except ValueError as error:
            raise ValueError(f"Job script {script} is outside staging root {local_root}.") from error
        return local_root, self.remote_wdir / self.job_name, script_relative

    @staticmethod
    def _mkdir_p(sftp, remote_dir: pathlib.PurePosixPath) -> None:
        current = pathlib.PurePosixPath("/")
        for part in remote_dir.parts[1:]:
            current /= part
            try:
                sftp.stat(str(current))
            except IOError:
                sftp.mkdir(str(current))

    def _transfer(self, sftp, local_root: pathlib.Path, remote_root: pathlib.PurePosixPath) -> None:
        self._mkdir_p(sftp, remote_root)
        skipped = {f"_{self.name}_jobs.json"}
        for path in local_root.rglob("*"):
            relative = path.relative_to(local_root)
            if (path.resolve() in self.staging_excludes if self.staging_excludes is not None
                    else relative.name in skipped):
                continue
            remote_path = remote_root.joinpath(*relative.parts)
            if path.is_dir():
                self._mkdir_p(sftp, remote_path)
            elif path.is_file():
                self._mkdir_p(sftp, remote_path.parent)
                sftp.put(str(path), str(remote_path))

    @staticmethod
    def _command_result(client, command: str) -> tuple[str, str]:
        _, stdout, stderr = client.exec_command(command)
        # Drain both streams concurrently: long direct jobs can fill stderr
        # while stdout is waiting for EOF.
        with ThreadPoolExecutor(max_workers=2) as pool:
            output_future = pool.submit(stdout.read)
            error_future = pool.submit(stderr.read)
            output = output_future.result().decode()
            error = error_future.result().decode()
        channel = getattr(stdout, "channel", None)
        if channel is not None and channel.recv_exit_status() != 0:
            raise RuntimeError(f"Remote command failed: {error.strip() or command}")
        return output, error

    def submit(self, func_to_execute: Optional[Callable] = None) -> str:
        del func_to_execute
        if not self.script.exists():
            self.write()
        if self.is_dry_run:
            return f"Attempt to submit {self.script.name} remotely to {self.hostname}."
        local_root, remote_root, script_relative = self._roots()
        client = self._client()
        sftp = None
        try:
            sftp = client.open_sftp()
            self._transfer(sftp, local_root, remote_root)
            remote_cwd = remote_root.joinpath(*script_relative.parent.parts)
            if self.scheduler.is_direct:
                launch_command = f"bash -l {shlex.quote(script_relative.name)}"
            else:
                launch_command = self.scheduler.build_submit_command(
                    shlex.quote(script_relative.name)
                )
            command = f"cd {shlex.quote(str(remote_cwd))} && {launch_command}"
            output, error = self._command_result(client, command)
            if self.scheduler.is_direct:
                return "direct"
            if not output.strip():
                raise RuntimeError(f"Remote submission returned no output: {error.strip()}")
            return self.scheduler.parse_submit_output(output)
        finally:
            if sftp is not None:
                sftp.close()
            client.close()

    def is_finished(self) -> bool:
        if self.scheduler.is_direct:
            return True
        client = self._client()
        try:
            output, _ = self._command_result(client, self.scheduler.ENQUIRE_COMMAND)
            return self.scheduler.is_finished_from_output(output)
        finally:
            client.close()

    def read_remote_file(self, relative_path: str) -> Optional[str]:
        """Read a job-owned artifact without replacing the controller's copy."""
        _, remote_root, _ = self._roots()
        relative = pathlib.PurePosixPath(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Remote artifact path must be relative to the staging root.")
        client = self._client()
        sftp = None
        try:
            sftp = client.open_sftp()
            with sftp.open(str(remote_root / relative), "r") as handle:
                content = handle.read()
                return content.decode("utf-8") if isinstance(content, bytes) else content
        except OSError as error:
            if error.errno == errno.ENOENT:
                return None
            raise
        finally:
            if sftp is not None:
                sftp.close()
            client.close()

    def sync(self, wdir_names: Iterable[str] = (), *, root_relative: bool = False) -> None:
        """Retrieve outputs; root-relative mode protects shared exploration metadata."""
        local_root, remote_root, _ = self._roots()
        skipped = [f"_{self.name}_jobs.json", "_scheduler.json"]
        client = self._client()
        sftp = None
        try:
            sftp = client.open_sftp()
            if root_relative:
                # Exploration jobs share metadata but own disjoint output trees.
                protected = {(local_root / '_meta').resolve()}
                count = removed = 0
                for name in wdir_names:
                    local_item = (local_root / name).resolve()
                    relative = local_item.relative_to(local_root)
                    if local_item in protected or (local_root / '_meta') in local_item.parents:
                        raise ValueError('Cannot synchronize shared exploration metadata as outputs.')
                    remote_item = remote_root.joinpath(*relative.parts)
                    count += _sync_latest_recursive(
                        sftp, str(remote_item), str(local_item), [], protected
                    )
                    removed += _remove_outdated_recursive(
                        sftp, str(remote_item), str(local_item), [], print_func=self._print,
                        protected_paths=protected,
                    )
                self._debug(f'synced {count} files; removed {removed} outdated items.')
                return
            count = _sync_latest_recursive(
                sftp, str(remote_root), str(local_root), skipped, self.sync_excludes
            )
            self._print(f"synced {count} files from {remote_root}.")
            removed = 0
            output_root = self.output_root or self.script.parent
            if self.output_root is None and output_root.name == "_meta":
                output_root = output_root.parent
            for item_name in wdir_names:
                local_item = (
                    output_root
                    if output_root.name == item_name
                    else output_root / item_name
                ).resolve()
                relative_item = local_item.relative_to(local_root)
                remote_item = remote_root.joinpath(*relative_item.parts)
                removed += _remove_outdated_recursive(
                    sftp, str(remote_item), str(local_item), skipped, print_func=self._print,
                    protected_paths=self.sync_excludes,
                )
            self._print(f"removed {removed} outdated items.")
        finally:
            if sftp is not None:
                sftp.close()
            client.close()


__all__ = ["SshTransport"]
