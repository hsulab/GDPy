import sys
import types

import pytest

from gdpx.providers import CapabilityKind, ProviderManager, get_provider_manager
from gdpx.execution.schedulers.local import LocalScheduler
from gdpx.execution.schedulers.lsf import LsfScheduler
from gdpx.execution.schedulers.pbs import PbsScheduler
from gdpx.execution.schedulers.slurm import SlurmScheduler
from gdpx.providers.schedulers import scheduler_providers


def test_scheduler_is_resolved_as_a_provider_capability():
    factory = get_provider_manager().require("local", CapabilityKind.SCHEDULER, "default")

    assert isinstance(factory.create({}), LocalScheduler)


def test_queue_schedulers_parse_transport_independent_output():
    slurm = SlurmScheduler()
    slurm.job_name = "target"
    assert not slurm.is_finished_from_output("123 compute target R 00:01 1:00 1 4\n")
    assert slurm.is_finished_from_output("123 compute another R 00:01 1:00 1 4\n")

    lsf = LsfScheduler()
    lsf.job_name = "target"
    assert lsf.parse_submit_output("Job <456> is submitted to queue <normal>.\n") == "456"
    assert not lsf.is_finished_from_output(
        "JOBID USER STAT QUEUE FROM_HOST EXEC_HOST JOB_NAME SUBMIT_TIME\n"
        "456 user RUN normal host node target Sep 3\n"
    )

    pbs = PbsScheduler()
    pbs.job_name = "target"
    assert "#PBS -N target" in str(pbs)
    assert pbs.parse_submit_output("789.server\n") == "789.server"
    assert not pbs.is_finished_from_output("789.server user queue target 1 1 -- 01:00 R 00:01\n")


def test_remote_provider_wraps_nested_scheduler_lazily(monkeypatch, tmp_path):
    fake_paramiko = types.ModuleType("paramiko")
    fake_paramiko.SSHClient = object
    fake_paramiko.SFTPClient = object
    fake_paramiko.AutoAddPolicy = object
    monkeypatch.setitem(sys.modules, "paramiko", fake_paramiko)
    sys.modules.pop("gdpx.execution.schedulers.remote", None)

    manager = ProviderManager()
    for provider in scheduler_providers():
        manager.register(provider)
    factory = manager.require("remote", CapabilityKind.SCHEDULER, "default")
    remote = factory.create(
        {
            "hostname": "cluster",
            "remote_wdir": "/scratch/jobs",
            "scheduler": {
                "provider": "slurm",
                "parameters": {"partition": "compute", "is_dry_run": True},
            },
        },
        providers=manager,
    )

    assert remote.name == "slurm"
    assert isinstance(remote.scheduler, SlurmScheduler)
    assert remote.scheduler.parameters["partition"] == "compute"

    with pytest.raises(ValueError, match="non-local"):
        factory.create(
            {
                "hostname": "cluster",
                "remote_wdir": "/scratch/jobs",
                "scheduler": {"provider": "local", "parameters": {}},
            },
            providers=manager,
        )

    class Channel:
        @staticmethod
        def recv_exit_status():
            return 0

    class Stream:
        channel = Channel()

        def __init__(self, value):
            self.value = value

        def read(self):
            return self.value.encode()

    class Sftp:
        def __init__(self):
            self.paths = {"/", "/scratch", "/scratch/jobs"}
            self.uploads = []

        def stat(self, path):
            if path not in self.paths:
                raise IOError(path)
            return object()

        def mkdir(self, path):
            self.paths.add(path)

        def put(self, local, remote_path):
            self.uploads.append((local, remote_path))

        def close(self):
            pass

    class Client:
        def __init__(self):
            self.sftp = Sftp()
            self.commands = []

        def set_missing_host_key_policy(self, policy):
            pass

        def connect(self, **kwargs):
            pass

        def open_sftp(self):
            return self.sftp

        def exec_command(self, command):
            self.commands.append(command)
            output = (
                "Submitted batch job 123\n"
                if command.startswith("cd ")
                else "123 compute target R 00:01 1:00 1 4\n"
            )
            return None, Stream(output), Stream("")

        def close(self):
            pass

    client = Client()
    remote.scheduler.is_dry_run = False
    remote._ssh_client_factory = lambda: client
    remote.job_name = "target"
    script_path = tmp_path / "run.script"
    script_path.write_text("#!/bin/bash\n")
    remote.script = script_path

    assert remote.submit() == "123"
    assert not remote.is_finished()
    assert any("sbatch run.script" in command for command in client.commands)
    assert client.sftp.uploads
    sys.modules.pop("gdpx.execution.schedulers.remote", None)
