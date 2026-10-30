"""Coordinate sampling chains and batch their basic calculation requests."""
import shlex

from gdpx.execution.schedulers.direct import DirectScheduler
from gdpx.execution.sampling_requests import coordinator
from gdpx.execution.workers.drive import DriverBasedWorker
from gdpx.execution.output import get_reporter


class MonteCarloWorker(DriverBasedWorker):
    """The controller is local; only emitted SPC/min/MD requests are scheduled."""
    def __init__(self, runtime, **kwargs):
        super().__init__(runtime, **kwargs)
        self.scheduler = DirectScheduler()

    def as_dict(self) -> dict:
        params = super().as_dict()
        # Sampling uses this worker for both dispatch kinds. Preserve the
        # configured kind so frozen runtimes still match their compute plan.
        params["dispatch"]["worker"] = self.runtime.config.dispatch.worker
        return params

    def _advance(self, batch):
        chains = []
        for atoms, name, seed in zip(batch[4], batch[1], batch[3]):
            engine = self.driver
            chains.append((engine, atoms, self.directory / name, seed))
        with coordinator(self.directory, self.batchsize) as broker:
            state = broker.advance(chains)
        for engine, _, directory, _ in chains:
            engine.directory = directory
            get_reporter(self).task_finished(str(engine.directory.relative_to(self.directory.resolve())),
                                            True if engine.read_convergence() else None)
        return state

    def _irun(self, batch_name, uid, identifier, frames, batch, *, submit=True):
        self.scheduler.job_name = uid + '-' + batch_name
        self.scheduler.script = self._script_path(uid)
        self.scheduler.script.parent.mkdir(parents=True, exist_ok=True)
        self.scheduler.user_commands = f'cd ../..\ngdp --log= compute run --job {shlex.quote(uid)}\n'
        self.scheduler.write()
        if submit:
            self._advance(batch)
            self.job_store.mark_submitted(self.scheduler.job_name, 'sampling')

    def _run_by_commandline(self, identifier, frames, batches, target_batch=None):
        return self._advance(batches[target_batch])

    def run_saved_job(self, path, task=None):
        if task is not None:
            raise ValueError('Sampling coordinates all chains; select basic calculation tasks instead.')
        _, batch = self._read_job_input(path)
        return self._advance(batch)
