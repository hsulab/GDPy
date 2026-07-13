from gdpx.worker.drive import DriverBasedWorker


def _create_computation_dirs(w, tmp_path):
    """Create the expected wdir directories so convergence checks pass."""
    for job in w.job_store.get_running():
        for wdir_name in job.wdir_names:
            (tmp_path / wdir_name).mkdir(parents=True, exist_ok=True)
from gdpx.worker.single import SingleWorker
from gdpx.worker.grid import GridDriverBasedWorker
from gdpx.worker.pairing import Pairing


class TestRun:
    def test_run_creates_jobs(self, mock_sched, fake_driver, fake_structure, tmp_path):
        w = DriverBasedWorker(scheduler=mock_sched, directory=tmp_path, batchsize=2)
        w.set_drivers(fake_driver)
        w.run([fake_structure] * 5)
        assert mock_sched.submit_count > 0
        assert len(w.job_store.get_running()) == mock_sched.submit_count

    def test_run_single_structure(self, mock_sched, fake_driver, fake_structure, tmp_path):
        w = DriverBasedWorker(scheduler=mock_sched, directory=tmp_path)
        w.set_drivers(fake_driver)
        w.run([fake_structure])
        assert len(w.job_store.get_running()) == 1


class TestInspect:
    def test_running_jobs_unchanged(self, mock_sched, fake_driver, fake_structure, tmp_path):
        w = DriverBasedWorker(scheduler=mock_sched, directory=tmp_path)
        w.set_drivers(fake_driver)
        w.run([fake_structure])
        assert len(w.job_store.get_running()) == 1
        w.inspect(resubmit=False)
        assert len(w.job_store.get_running()) == 1
        assert len(w.job_store.get_finished()) == 0

    def test_marks_finished(self, mock_sched, fake_driver, fake_structure, tmp_path):
        w = DriverBasedWorker(scheduler=mock_sched, directory=tmp_path)
        w.set_drivers(fake_driver)
        w.run([fake_structure])
        _create_computation_dirs(w, tmp_path)
        mock_sched.finish(True)
        w.inspect(resubmit=False)
        assert len(w.job_store.get_finished()) == 1
        assert len(w.job_store.get_running()) == 0

    def test_resubmit_on_unfinished(self, mock_sched, fake_driver, fake_structure, tmp_path):
        w = DriverBasedWorker(scheduler=mock_sched, directory=tmp_path)
        w.set_drivers(fake_driver)
        w.run([fake_structure])
        mock_sched.finish(True)
        w._check_job_convergence = lambda job: False
        prev = mock_sched.submit_count
        w.inspect(resubmit=True)
        assert mock_sched.submit_count > prev


class TestRetrieve:
    def test_retrieve_returns_list(self, mock_sched, fake_driver, fake_structure, tmp_path):
        w = DriverBasedWorker(scheduler=mock_sched, directory=tmp_path)
        w.set_drivers(fake_driver)
        w.run([fake_structure])
        _create_computation_dirs(w, tmp_path)
        mock_sched.finish(True)
        results = w.retrieve()
        assert isinstance(results, list)

    def test_retrieve_only_new(self, mock_sched, fake_driver, fake_structure, tmp_path):
        w = DriverBasedWorker(scheduler=mock_sched, directory=tmp_path)
        w.set_drivers(fake_driver)
        w.run([fake_structure])
        _create_computation_dirs(w, tmp_path)
        mock_sched.finish(True)
        first = w.retrieve()
        second = w.retrieve(include_retrieved=False)
        assert len(second) == 0


class TestPairing:
    def test_broadcast(self, mock_sched, fake_driver, tmp_path):
        w = DriverBasedWorker(scheduler=mock_sched, directory=tmp_path, pairing=Pairing.BROADCAST)
        w.set_drivers(fake_driver)
        plan = w._make_task_plan(5)
        assert len(plan) == 5
        assert all(di == 0 for di, _ in plan)

    def test_repeat(self, mock_sched, fake_driver, tmp_path):
        w = DriverBasedWorker(scheduler=mock_sched, directory=tmp_path, pairing=Pairing.REPEAT)
        w.set_drivers(fake_driver, fake_driver, fake_driver)
        plan = w._make_task_plan(1)
        assert len(plan) == 3

    def test_bijection(self, mock_sched, fake_driver, tmp_path):
        w = DriverBasedWorker(scheduler=mock_sched, directory=tmp_path, pairing=Pairing.BIJECTION)
        w.set_drivers(fake_driver, fake_driver, fake_driver)
        plan = w._make_task_plan(3)
        assert plan == [(0, 0), (1, 1), (2, 2)]

    def test_product(self, mock_sched, fake_driver, tmp_path):
        w = DriverBasedWorker(scheduler=mock_sched, directory=tmp_path, pairing=Pairing.PRODUCT)
        w.set_drivers(fake_driver, fake_driver)
        plan = w._make_task_plan(3)
        assert len(plan) == 6

    def test_partition(self, mock_sched, fake_driver, tmp_path):
        w = DriverBasedWorker(scheduler=mock_sched, directory=tmp_path, pairing=Pairing.PARTITION)
        w.set_drivers(fake_driver, fake_driver)
        plan = w._make_task_plan(5)
        assert plan == [(0, 0), (1, 1), (0, 2), (1, 3), (0, 4)]

    def test_auto_broadcast(self, mock_sched, fake_driver, tmp_path):
        w = DriverBasedWorker(scheduler=mock_sched, directory=tmp_path, pairing=Pairing.AUTO)
        w.set_drivers(fake_driver)
        plan = w._make_task_plan(10)
        assert len(plan) == 10
        assert all(di == 0 for di, _ in plan)

    def test_auto_bijection(self, mock_sched, fake_driver, tmp_path):
        w = DriverBasedWorker(scheduler=mock_sched, directory=tmp_path, pairing=Pairing.AUTO)
        w.set_drivers(fake_driver, fake_driver)
        plan = w._make_task_plan(2)
        assert plan == [(0, 0), (1, 1)]

    def test_auto_raises_on_mismatch(self, mock_sched, fake_driver, tmp_path):
        w = DriverBasedWorker(scheduler=mock_sched, directory=tmp_path, pairing=Pairing.AUTO)
        w.set_drivers(fake_driver, fake_driver)
        import pytest
        with pytest.raises(ValueError):
            w._make_task_plan(3)


class TestSubclasses:
    def test_single_worker(self, mock_sched, fake_driver, fake_structure, tmp_path):
        w = SingleWorker(driver=fake_driver, scheduler=mock_sched, directory=str(tmp_path))
        assert len(w.drivers) == 1
        assert isinstance(w, DriverBasedWorker)
        w.run([fake_structure])
        assert len(w.job_store.get_running()) > 0

    def test_grid_worker(self, mock_sched, fake_driver, tmp_path):
        class FakePotter:
            as_dict = lambda self: {"name": "test"}
        gw = GridDriverBasedWorker(
            [FakePotter(), FakePotter()],
            [fake_driver, fake_driver],
            scheduler=mock_sched, directory=str(tmp_path),
        )
        assert len(gw.drivers) == 2

    def test_single_from_a_worker(self, mock_sched, fake_driver, tmp_path):
        w = DriverBasedWorker(scheduler=mock_sched, directory=tmp_path)
        w.set_drivers(fake_driver)
        sw = SingleWorker.from_a_worker(w)
        assert len(sw.drivers) == 1

    def test_single_rewind(self, mock_sched, fake_driver, fake_structure, tmp_path):
        w = SingleWorker(driver=fake_driver, scheduler=mock_sched, directory=str(tmp_path))
        w.run([fake_structure])
        assert len(w.job_store.get_running()) == 1
        w.rewind_to_step(0)
        assert len(w.job_store.get_running()) == 1  # step 0 still runs
        w.rewind_to_step(-1)
        assert len(w.job_store.get_running()) == 0  # step -1 removes all


class TestBackwardCompat:
    def test_constructor_with_potter(self, mock_sched, fake_driver, tmp_path):
        class FakePotter:
            as_dict = lambda self: {"name": "test"}
        w = DriverBasedWorker(FakePotter(), fake_driver, mock_sched, directory=tmp_path)
        assert len(w.drivers) == 1
        assert w.potter is not None

    def test_constructor_with_driver_kwarg(self, mock_sched, fake_driver, tmp_path):
        w = DriverBasedWorker(driver=fake_driver, scheduler=mock_sched, directory=tmp_path)
        assert len(w.drivers) == 1


class TestJobStore:
    def test_persistence_across_instances(self, mock_sched, fake_driver, fake_structure, tmp_path):
        w = DriverBasedWorker(scheduler=mock_sched, directory=tmp_path)
        w.set_drivers(fake_driver)
        w.run([fake_structure])
        w2 = DriverBasedWorker(scheduler=mock_sched, directory=tmp_path)
        assert len(w2.job_store.get_running()) == len(w.job_store.get_running())

    def test_db_file_created(self, mock_sched, fake_driver, fake_structure, tmp_path):
        w = DriverBasedWorker(scheduler=mock_sched, directory=tmp_path)
        w.set_drivers(fake_driver)
        w.run([fake_structure])
        db_path = tmp_path / f"_{mock_sched.name}_jobs.json"
        assert db_path.exists()


class TestTemplateHooks:
    def test_convergence_hook_called(self, mock_sched, fake_driver, fake_structure, tmp_path):
        calls = []
        class TestWorker(DriverBasedWorker):
            def _check_job_convergence(self, job):
                calls.append("check")
                return True
        w = TestWorker(scheduler=mock_sched, directory=tmp_path)
        w.set_drivers(fake_driver)
        w.run([fake_structure])
        mock_sched.finish(True)
        w.inspect(resubmit=True)
        assert "check" in calls

    def test_resubmit_hook_called(self, mock_sched, fake_driver, fake_structure, tmp_path):
        calls = []
        class TestWorker(DriverBasedWorker):
            def _check_job_convergence(self, job):
                return False
            def _resubmit_job(self, job):
                calls.append("resubmit")
        w = TestWorker(scheduler=mock_sched, directory=tmp_path)
        w.set_drivers(fake_driver)
        w.run([fake_structure])
        mock_sched.finish(True)
        w.inspect(resubmit=True)
        assert "resubmit" in calls
