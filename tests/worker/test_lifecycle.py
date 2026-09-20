from gdpx.execution import Runtime
from gdpx.execution.workers.drive import DriverBasedWorker
from gdpx.execution.workers.single import SingleWorker
from gdpx.providers import ComponentConfig, RuntimeConfig
from gdpx.providers.specs import PotentialSpec
from gdpx.utils.archive import ZSTD_ARCHIVE_NAME


def _runtime(driver, scheduler, *, worker="batch"):
    config = RuntimeConfig(
        potential=ComponentConfig("test"),
        executor=ComponentConfig("test", "run"),
        options={"worker": worker},
    )
    return Runtime(
        potential=PotentialSpec("test", {}),
        materialization=None,
        executor=driver,
        modifiers=(),
        config=config,
        scheduler=scheduler,
    )


def _create_computation_dirs(worker, tmp_path):
    for job in worker.job_store.get_running():
        for workdir in job.wdir_names:
            (tmp_path / workdir).mkdir(parents=True, exist_ok=True)


def test_runtime_worker_creates_jobs(mock_sched, fake_driver, fake_structure, tmp_path):
    worker = DriverBasedWorker(_runtime(fake_driver, mock_sched), directory=tmp_path, batchsize=2)
    worker.run([fake_structure] * 5)
    assert mock_sched.submit_count > 0
    assert len(worker.job_store.get_running()) == mock_sched.submit_count


def test_runtime_worker_marks_finished(mock_sched, fake_driver, fake_structure, tmp_path):
    worker = DriverBasedWorker(_runtime(fake_driver, mock_sched), directory=tmp_path)
    worker.run([fake_structure])
    _create_computation_dirs(worker, tmp_path)
    mock_sched.finish(True)
    worker.inspect(resubmit=False)
    assert len(worker.job_store.get_finished()) == 1


def test_runtime_worker_retrieves_and_archives(mock_sched, fake_driver, fake_structure, tmp_path):
    worker = DriverBasedWorker(_runtime(fake_driver, mock_sched), directory=tmp_path)
    worker.run([fake_structure])
    _create_computation_dirs(worker, tmp_path)
    output_path = tmp_path / "cand0" / "output.txt"
    output_path.write_text("finished")
    mock_sched.finish(True)
    assert isinstance(worker.retrieve(use_archive=True), list)
    assert (tmp_path / ZSTD_ARCHIVE_NAME).is_file()


def test_single_worker_and_conversion(mock_sched, fake_driver, fake_structure, tmp_path):
    runtime = _runtime(fake_driver, mock_sched, worker="single")
    batch_worker = DriverBasedWorker(runtime, directory=tmp_path)
    worker = SingleWorker.from_a_worker(batch_worker)
    assert isinstance(worker, DriverBasedWorker)
    worker.run([fake_structure])
    assert len(worker.job_store.get_running()) == 1


def test_runtime_config_is_the_only_serialized_worker_input(mock_sched, fake_driver, tmp_path):
    worker = DriverBasedWorker(_runtime(fake_driver, mock_sched), directory=tmp_path)
    serialized = worker.as_dict()
    assert serialized["schema_version"] == 3
    assert "potter" not in serialized
    assert "driver" not in serialized
