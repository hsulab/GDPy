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


class TestMockScheduler:
    def test_submit_creates_script(self, tmp_path):
        s = MockScheduler()
        s.script = tmp_path / "run.script"
        job_id = s.submit()
        assert job_id == "mock_1"
        assert (tmp_path / "run.script").exists()
        assert len(s.submitted_jobs) == 1
        assert s.submitted_jobs[0]["script"] == str(tmp_path / "run.script")

    def test_submit_increments_count(self, tmp_path):
        s = MockScheduler()
        s.script = tmp_path / "s1"
        s.submit()
        assert s.submit_count == 1
        s.script = tmp_path / "s2"
        s.submit()
        assert s.submit_count == 2
        assert s.submitted_jobs[1]["script"] == str(tmp_path / "s2")

    def test_submit_tracks_func_to_execute(self, tmp_path):
        s = MockScheduler()
        s.script = tmp_path / "s1"
        fn = lambda: None
        s.submit(func_to_execute=fn)
        assert s.submitted_jobs[0]["func"] is fn

    def test_submit_without_func(self, tmp_path):
        s = MockScheduler()
        s.script = tmp_path / "s1"
        s.submit()
        assert s.submitted_jobs[0]["func"] is None

    def test_is_finished_default_false(self):
        s = MockScheduler()
        assert s.is_finished() is False

    def test_is_finished_controllable(self):
        s = MockScheduler()
        s.finish(True)
        assert s.is_finished() is True
        s.finish(False)
        assert s.is_finished() is False

    def test_job_name_property(self):
        s = MockScheduler()
        s.job_name = "test-job"
        assert s.job_name == "test-job"
