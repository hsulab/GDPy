import sys
import types

import pytest

from gdpx.providers import CapabilityKind, ProviderManager, get_provider_manager
from gdpx.execution.schedulers.direct import DirectScheduler
from gdpx.execution.schedulers.lsf import LsfScheduler
from gdpx.execution.schedulers.pbs import PbsScheduler
from gdpx.execution.schedulers.slurm import SlurmScheduler
from gdpx.providers.schedulers import scheduler_providers, transport_providers


def test_scheduler_is_resolved_as_a_provider_capability():
    factory = get_provider_manager().require("direct", CapabilityKind.SCHEDULER, "default")

    assert isinstance(factory.create({}), DirectScheduler)

    transport = get_provider_manager().require(
        "local", CapabilityKind.TRANSPORT, "default"
    )
    scheduler = DirectScheduler()
    assert transport.create({}, scheduler=scheduler) is scheduler


def test_direct_scheduler_renders_an_executable_script():
    scheduler = DirectScheduler(environs="source ~/.profile\n")
    scheduler.user_commands = "gdp compute run\n"

    content = str(scheduler)

    assert content.startswith("#!/bin/bash -l")
    assert "source ~/.profile" in content
    assert "gdp compute run" in content


def test_concurrent_tasks_are_validated_and_not_rendered_as_directives():
    scheduler = SlurmScheduler(concurrent_tasks=4, ntasks=256)

    assert scheduler.concurrent_tasks == 4
    assert "#SBATCH --ntasks=256" in str(scheduler)
    assert "concurrent-tasks" not in str(scheduler)
    assert scheduler.as_dict()["parameters"]["concurrent_tasks"] == 4

    assert DirectScheduler(concurrent_tasks=2).concurrent_tasks == 2
    with pytest.raises(ValueError, match="positive integer"):
        SlurmScheduler(concurrent_tasks=0)
    with pytest.raises(ValueError, match="does not support"):
        PbsScheduler(concurrent_tasks=2)


def test_direct_script_exit_status_and_dry_run(tmp_path):
    import subprocess

    scheduler = DirectScheduler()
    scheduler.script = tmp_path / "run.script"
    scheduler.user_commands = "exit 7\n"
    with pytest.raises(subprocess.CalledProcessError):
        scheduler.submit()
    scheduler.is_dry_run = True
    assert "Attempt" in scheduler.submit(lambda: pytest.fail("dry run executed callback"))


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


def test_ssh_transport_wraps_queue_and_direct_schedulers_lazily(monkeypatch, tmp_path):
    fake_paramiko = types.ModuleType("paramiko")
    fake_paramiko.SSHClient = object
    fake_paramiko.SFTPClient = object
    fake_paramiko.AutoAddPolicy = object
    monkeypatch.setitem(sys.modules, "paramiko", fake_paramiko)
    sys.modules.pop("gdpx.execution.schedulers.remote", None)

    manager = ProviderManager()
    for provider in (*scheduler_providers(), *transport_providers()):
        manager.register(provider)
    scheduler_factory = manager.require("slurm", CapabilityKind.SCHEDULER, "default")
    scheduler = scheduler_factory.create(
        {"partition": "compute", "is_dry_run": True},
        providers=manager,
    )
    factory = manager.require("ssh", CapabilityKind.TRANSPORT, "default")
    remote = factory.create(
        {
            "hostname": "cluster",
            "remote_wdir": "/scratch/jobs",
        },
        providers=manager,
        scheduler=scheduler,
    )

    assert remote.name == "slurm"
    assert isinstance(remote.scheduler, SlurmScheduler)
    assert remote.scheduler.parameters["partition"] == "compute"

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
            if "sbatch" in command:
                output = "Submitted batch job 123\n"
            elif "bash" in command:
                output = "completed\n"
            else:
                output = "123 compute target R 00:01 1:00 1 4\n"
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

    direct = manager.require("direct", CapabilityKind.SCHEDULER, "default").create({})
    ssh_direct = factory.create(
        {"hostname": "cluster", "remote_wdir": "/scratch/jobs"},
        providers=manager,
        scheduler=direct,
    )
    ssh_direct._ssh_client_factory = lambda: client
    ssh_direct.job_name = "direct-target"
    direct_script = tmp_path / "direct.script"
    direct_script.write_text("#!/bin/bash\n")
    ssh_direct.script = direct_script

    called = False

    def local_callback():
        nonlocal called
        called = True

    assert ssh_direct.submit(func_to_execute=local_callback) == "direct"
    assert not called
    assert ssh_direct.is_finished()
    assert any("bash -l direct.script" in command for command in client.commands)
    ssh_direct.scheduler.is_dry_run = True
    command_count = len(client.commands)
    assert "Attempt" in ssh_direct.submit()
    assert len(client.commands) == command_count
    ssh_direct.scheduler.is_dry_run = False

    monkeypatch.setattr(Channel, "recv_exit_status", staticmethod(lambda: 9))
    with pytest.raises(RuntimeError, match="Remote command failed"):
        ssh_direct.submit()

    with pytest.raises(ValueError, match="absolute POSIX"):
        factory.create(
            {"hostname": "cluster", "remote_wdir": "relative"}, scheduler=direct
        )
    sys.modules.pop("gdpx.execution.schedulers.remote", None)
