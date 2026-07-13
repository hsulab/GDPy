import pytest

from ase import Atoms

from gdpx.scheduler.scheduler import BaseScheduler


class MockScheduler(BaseScheduler):
    """Controllable scheduler that simulates HPC submission without execution."""

    name = "mock"
    PREFIX = ""
    SUFFIX = ""
    SHELL = ""
    SUBMIT_COMMAND = ""
    ENQUIRE_COMMAND = ""

    def __init__(self):
        super().__init__()
        self._finished = False
        self.submit_count = 0
        self.submitted_jobs: list[dict] = []

    def submit(self, func_to_execute=None):
        self.submit_count += 1
        self.script.parent.mkdir(parents=True, exist_ok=True)
        self.write()
        self.submitted_jobs.append(dict(
            script=str(self.script),
            func=func_to_execute,
            name=self.job_name,
        ))
        return f"mock_{self.submit_count}"

    def is_finished(self) -> bool:
        return self._finished

    def finish(self, val: bool = True):
        self._finished = val

    @property
    def job_name(self):
        return self._job_name

    @job_name.setter
    def job_name(self, name):
        self._job_name = name


@pytest.fixture
def mock_sched():
    return MockScheduler()


@pytest.fixture
def fake_driver():
    class FakeDriver:
        random_seed = 42
        setting = type("s", (), {"machine_prefix": ""})()
        def as_dict(self): return {"backend": "fake"}
        def set_rng(self, **kw): pass
        def reset(self): pass
        def run(self, atoms, **kw): pass
        def read_convergence(self): return True
        def read_trajectory(self, add_step_info=False, archive_path=None): return []
    return FakeDriver()


@pytest.fixture
def fake_structure():
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
