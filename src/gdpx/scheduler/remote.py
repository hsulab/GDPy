#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import pathlib
import re
import shutil
import stat
import traceback
from typing import Callable, Optional

import paramiko

from .slurm import SlurmScheduler


def _should_sync_file(sftp: paramiko.SFTPClient, remote_file_path, local_file_path) -> bool:
    """
    If the remote_file should be synced - if it was not downloaded or it is out of sync with the remote version.

    Args:
        sftp:                Connection to the sftp server.
        remote_file_path:    Remote file path.
        local_file_path:     Local file path.

    Returns:
        True if the remote file should be synced, False otherwise.

    """
    if not os.path.exists(local_file_path):
        return True
    else:
        remote_attr = sftp.lstat(remote_file_path)
        local_stat = os.stat(local_file_path)
        return remote_attr.st_size != local_stat.st_size or remote_attr.st_mtime != local_stat.st_mtime


def _sync_latest_recursive(sftp: paramiko.SFTPClient, remote_dir: str, local_dir: str, skipped_items) -> int:
    """
    Recursively sync the sftp contents starting at remote dir to the local dir,
    and return the number of files synced.

    Args:
        sftp:        Connection to the sftp server.
        remote_dir:  Remote dir to start sync from.
        local_dir:   To sync to.

    Returns:
        The number of files synced.

    """
    files_synced = 0
    for item in sftp.listdir_attr(remote_dir):
        remote_dir_item = os.path.join(remote_dir, item.filename)
        local_dir_item = os.path.join(local_dir, item.filename)
        if item.filename in skipped_items:
            continue
        if stat.S_ISREG(item.st_mode):
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            if _should_sync_file(sftp, remote_dir_item, local_dir_item):
                sftp.get(remote_dir_item, local_dir_item)
                times = (
                    sftp.lstat(remote_dir_item).st_atime,
                    sftp.lstat(remote_dir_item).st_mtime,
                )
                os.utime(local_dir_item, times)
                files_synced += 1
        else:
            files_synced += _sync_latest_recursive(sftp, remote_dir_item, local_dir_item, skipped_items=skipped_items)

    return files_synced


def _remove_outdated_recursive(
    sftp: paramiko.SFTPClient,
    remote_dir: str,
    local_dir: str,
    skipped_items: list[str],
    print_func=print,
    debug_func=print,
) -> int:
    """Remove outdated items.

    Since the sync process may leave behind files that are no longer present on the remote machine,
    we need remove items that do not exist on the remote machine.
    For example, one calculation on remote moves previous outputs to a new directory but the sync process
    downloads the new directory and keep the previous outputs, which may make the read_convergence fail
    due to inconsistency between the last frame of the previous trajectory and the initial structure of
    the current trajectory.
    The above error will not occur if the previous outputs are overwritten by the new calculation but
    sometimes the calculation failed to restart and the previous outputs are still there and will not
    be update by sync as they are the same.

    Args:
        sftp:            Connection to the sftp server.
        remote_dir:      Remote dir to check for outdated items.
        local_dir:       Local dir to check for outdated items.
        skipped_items:   List of items to skip during removal.
        print_func:      Function to print messages (default is print).
        debug_func:      Function to print debug messages (default is print).

    Returns:
        The number of items removed.

    """
    items_removed = 0
    for item in os.listdir(local_dir):
        remote_dir_item = os.path.join(remote_dir, item)
        local_dir_item = os.path.join(local_dir, item)
        if item in skipped_items:
            continue
        try:
            if os.path.isdir(local_dir_item):
                items_removed += _remove_outdated_recursive(
                    sftp,
                    remote_dir_item,
                    local_dir_item,
                    skipped_items=skipped_items,
                    print_func=print_func,
                    debug_func=debug_func,
                )
            # Check if the remote item exists.
            # If it does not exist, the sftp.stat will throw a FileNotFoundError,
            # which is a subclass of IOError.
            _ = sftp.stat(remote_dir_item)
        except IOError:
            print_func("removing {}".format(local_dir_item))
            try:
                if os.path.isfile(local_dir_item):
                    os.remove(local_dir_item)  # remove file
                else:
                    shutil.rmtree(local_dir_item)  # remove directory
            except Exception as e:
                print_func(f"could not remove {local_dir_item}, error {str(e)}")
            items_removed += 1

    return items_removed


class RemoteSlurmScheduler(SlurmScheduler):

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        self.remote_wdir = pathlib.Path(self.remote_wdir)
        if not self.remote_wdir.is_absolute():
            raise RuntimeError("Remote_wdir must be an absolute path.")
        else:
            ...

        return

    def _transfer(self, sftp, remote_dir, skipped_items):
        """"""
        # Check if the remote_wdir exists before sending files
        # If several batches are submitted, this transfer will
        # just run once.
        try:
            rdir_stat = sftp.stat(str(remote_dir))
            self._print(f"remote_dir `{rdir_stat =}` has already been transferred.")
        except IOError:
            sftp.mkdir(str(remote_dir))

            local_dir = self.script.parent
            for p in local_dir.rglob("*"):
                relative_path = p.relative_to(local_dir)
                if relative_path.name in skipped_items:
                    continue
                remote_path = remote_dir / relative_path
                if p.is_dir():
                    try:
                        sftp.mkdir(str(remote_path))
                    except IOError:
                        ...
                else:
                    sftp.put(str(p), str(remote_path))

        return

    def submit(self, func_to_execute: Optional[Callable] = None) -> str:
        """Submit job to a remote machine."""
        if func_to_execute is not None and self.name == "local":
            raise Exception("Cannot run a function on the remote machine locally.")

        job_id = f"REMOTE -> {self.hostname.upper()} "
        sftp = None
        try:
            password = os.environ.get(f"{self.hostname.upper()}_PASSWORD")
            self.ssh.connect(hostname=self.hostname, password=password)
            sftp = self.ssh.open_sftp()

            juid = self.script.name.split(".")[0][4:]  # run-{uid}.script
            remote_dir = pathlib.Path(self.remote_wdir) / juid

            # The jobs.json file is used to keep track of the jobs submitted to the remote machine,
            # thus, we do not need upload it.
            self._transfer(sftp, remote_dir, skipped_items=[f"_{self.name}_jobs.json"])

            command = f"cd {str(remote_dir)}; {self.SUBMIT_COMMAND} {self.script.name}"
            _, stdout, stderr = self.ssh.exec_command(command)

            output = stdout.read().decode()
            _ = stderr.read().decode()
            if not output:
                raise RuntimeError("Job submission failed.")

            job_id += output.strip().split()[-1]
        except Exception:
            self._print(f"{traceback.format_exc()}")
            job_id += f"FAILED: sshconnection"
        finally:
            self.ssh.close()
            if sftp is not None:
                sftp.close()

        return job_id

    def _sync_remote(self, wdir_names: list[str]) -> None:
        """Syncronize the working directories from the remote machine.

        Normally, this method should be called after the remote job is finished.

        """
        password = os.environ.get(f"{self.hostname.upper()}_PASSWORD")
        try:
            self.ssh.connect(hostname=self.hostname, password=password)

            # pull results from the remote
            sftp = self.ssh.open_sftp()

            local_dir = str(self.script.parent)

            juid = self.script.name.split(".")[0][4:]  # run-{uid}.script
            remote_dir = str(pathlib.Path(self.remote_wdir) / juid)

            try:
                # download all the files from the remote machine
                self._print(f"start syncing '{remote_dir}'...")
                num_files_synced = _sync_latest_recursive(
                    sftp,
                    remote_dir,
                    local_dir,
                    skipped_items=[f"_{self.name}_jobs.json"],
                )
                self._print(f"synced {num_files_synced} files.")

                # remove_outdated (only outdated files in wdirs will be removed)
                self._print(f"start removing outdated items...")
                num_outdated_removed = 0
                for item_name in wdir_names:
                    num_outdated_removed += _remove_outdated_recursive(
                        sftp,
                        str(pathlib.Path(remote_dir) / item_name),
                        str(pathlib.Path(local_dir) / item_name),
                        skipped_items=[f"_{self.name}_jobs.json"],
                        print_func=self._print,
                        debug_func=self._debug,
                    )
                self._print(f"removed {num_outdated_removed} outdated items.")
            except Exception as e:
                self._print(f"error syncing: {str(e)}")
            finally:
                sftp.close()
        finally:
            self.ssh.close()

        return

    def is_finished(self) -> bool:
        """"""
        finished = False

        password = os.environ.get(f"{self.hostname.upper()}_PASSWORD")
        try:
            self.ssh.connect(hostname=self.hostname, password=password)

            _, stdout, _ = self.ssh.exec_command(self.ENQUIRE_COMMAND)
            output = stdout.read().decode()

            pattern = re.compile(r"\s+(\d+)\s+\S+\s+(\S+)\s+[A-Z]+\s+\S+\s+\S+\s+\d+\s+\d+")
            matches = pattern.findall(output)
            names = [m[1] for m in matches]

            if self.job_name not in names:
                finished = True
            else:
                finished = False

        finally:
            self.ssh.close()

        return finished


if __name__ == "__main__":
    ...
