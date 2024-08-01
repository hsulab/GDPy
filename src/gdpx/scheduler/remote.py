#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import pathlib
import re
import shutil
import stat
import traceback

import paramiko

from .slurm import SlurmScheduler


def _should_sync_file(sftp: paramiko.SFTPClient, remote_file_path, local_file_path):
    """
    If the remote_file should be synced - if it was not downloaded or it is out of sync with the remote version.

    Args:
        sftp:                Connection to the sftp server.
        remote_file_path:    Remote file path.
        local_file_path:     Local file path.

    Returns:

    """
    if not os.path.exists(local_file_path):
        return True
    else:
        remote_attr = sftp.lstat(remote_file_path)
        local_stat = os.stat(local_file_path)
        return (
            remote_attr.st_size != local_stat.st_size
            or remote_attr.st_mtime != local_stat.st_mtime
        )


def _sync_r(sftp: paramiko.SFTPClient, remote_dir, local_dir, skipp_items):
    """
    Recursively sync the sftp contents starting at remote dir to the local dir and return the number of files synced.

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
        if item.filename in skipp_items:
            continue
        # print("check {} => {}".format(remote_dir_item, local_dir_item))
        if stat.S_ISREG(item.st_mode):
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            if _should_sync_file(sftp, remote_dir_item, local_dir_item):
                # print("sync {} => {}".format(remote_dir_item, local_dir_item))
                sftp.get(remote_dir_item, local_dir_item)
                times = (
                    sftp.lstat(remote_dir_item).st_atime,
                    sftp.lstat(remote_dir_item).st_mtime,
                )
                os.utime(local_dir_item, times)
                files_synced += 1
        else:
            files_synced += _sync_r(sftp, remote_dir_item, local_dir_item, skipp_items)

    return files_synced


def _remove_outdated_r(sftp, remote_dir, local_dir, skipped_items):
    items_removed = 0
    for item in os.listdir(local_dir):
        remote_dir_item = os.path.join(remote_dir, item)
        local_dir_item = os.path.join(local_dir, item)
        if item in skipped_items:
            continue
        try:
            rdir_stat = sftp.stat(str(remote_dir))
            if os.path.isdir(local_dir_item):
                items_removed += _remove_outdated_r(
                    sftp, remote_dir_item, local_dir_item, skipped_items
                )
        except IOError:
            print("removing {}".format(local_dir_item))
            _remove(local_dir_item)
            items_removed += 1

    return items_removed


def _remove(path):
    """param <path> could either be relative or absolute."""
    try:
        if os.path.isfile(path):
            os.remove(path)  # remove file
        else:
            shutil.rmtree(path)  # remove directory
    except Exception as e:
        print("could not remove {}, error {0}".format(path, str(e)))


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
        sftp = self.ssh.open_sftp()

        local_dir = self.script.parent

        try:
            rdir_stat = sftp.stat(str(remote_dir))
            print(f"transfer remote_dir {rdir_stat =}")
        except IOError:
            sftp.mkdir(str(remote_dir))

            for p in local_dir.rglob("*"):
                # print(f"put {p}")
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

    def submit(self) -> str:
        """Submit job to a remote machine."""
        password = os.environ.get(f"{self.hostname.upper()}_PASSWORD")

        job_id = f"REMOTE -> {self.hostname.upper()} "
        try:
            self.ssh.connect(hostname=self.hostname, password=password)
            sftp = self.ssh.open_sftp()

            juid = self.script.name.split(".")[0][4:]  # run-{uid}.script
            remote_dir = pathlib.Path(self.remote_wdir) / juid
            self._transfer(sftp, remote_dir, [f"_{self.name}_jobs.json"])

            command = f"cd {str(remote_dir)}; {self.SUBMIT_COMMAND} {self.script.name}"
            stdin, stdout, stderr = self.ssh.exec_command(command)

            output = stdout.read().decode()
            error = stderr.read().decode()
            if not output:
                raise RuntimeError("Job submission failed.")

            job_id += output.strip().split()[-1]
        except Exception:
            print(f"{traceback.format_exc()}")
            job_id += f"FAILED: {error}"
        finally:
            self.ssh.close()
            sftp.close()

        return job_id

    def is_finished(self) -> bool:
        """"""
        finished = False

        password = os.environ.get(f"{self.hostname.upper()}_PASSWORD")
        try:
            self.ssh.connect(hostname=self.hostname, password=password)

            stdin, stdout, stderr = self.ssh.exec_command(self.ENQUIRE_COMMAND)
            output = stdout.read().decode()

            pattern = re.compile(
                r"\s+(\d+)\s+\S+\s+(\S+)\s+[A-Z]+\s+\S+\s+\S+\s+\d+\s+\d+"
            )
            matches = pattern.findall(output)
            names = [m[1] for m in matches]

            if self.job_name not in names:
                finished = True
            else:
                finished = False
            # print(f"{finished =}")

            # pull results from the remote
            sftp = None
            if finished:
                sftp = self.ssh.open_sftp()

                local_dir = str(self.script.parent)

                juid = self.script.name.split(".")[0][4:]  # run-{uid}.script
                remote_dir = str(pathlib.Path(self.remote_wdir) / juid)

                try:
                    rdir_stat = sftp.stat(str(remote_dir))

                    files_synced = _sync_r(
                        sftp, remote_dir, local_dir, [f"_{self.name}_jobs.json"]
                    )
                    print(
                        "synced {} file(s) from '{}'".format(files_synced, remote_dir)
                    )
                    # remove_outdated
                    print(f"cleaning up outdated items of '{remote_dir}' starting...")
                    outdated_removed = _remove_outdated_r(
                        sftp, remote_dir, local_dir, [f"_{self.name}_jobs.json"]
                    )
                    print(
                        f"removed {outdated_removed} outdated item(s) of '{remote_dir}'"
                    )
                except:
                    ...
                finally:
                    sftp.close()
            else:
                ...
        finally:
            self.ssh.close()
            if sftp is not None:
                sftp.close()

        return finished


if __name__ == "__main__":
    ...
