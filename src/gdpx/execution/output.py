"""Provider-independent, bounded worker summaries from existing results."""
from contextvars import ContextVar
from functools import wraps
import time

import numpy as np

from gdpx.core.output import Box, message, quiet_logging


_SESSION = ContextVar('execution_reporting', default=None)


def reporting_session(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        token = _SESSION.set({}) if _SESSION.get() is None else None
        try:
            with quiet_logging():
                return function(*args, **kwargs)
        finally:
            if token is not None:
                _SESSION.reset(token)
    return wrapped


def get_reporter(worker):
    key = str(worker.directory)
    session = _SESSION.get()
    if session is not None:
        if key not in session:
            session[key] = WorkerReporter(worker)
        reporter = session[key]
        reporter.worker = worker
        return reporter
    reporter = getattr(worker, '_output_reporter', None)
    if reporter is None or reporter.directory != key:
        reporter = worker._output_reporter = WorkerReporter(worker)
    return reporter


def worker_output(action):
    def decorate(function):
        @wraps(function)
        def wrapped(worker, *args, **kwargs):
            if not hasattr(worker, 'runtime') or not hasattr(worker, '_drivers'):
                return function(worker, *args, **kwargs)
            reporter = get_reporter(worker)
            with quiet_logging():
                try:
                    result = function(worker, *args, **kwargs)
                except Exception as error:
                    for failure in getattr(error, 'failures', ()):
                        reporter.outcomes[str(failure.workdir)] = False
                    try:
                        reporter.summary('failed')
                    except Exception:
                        pass
                    raise
                if action == 'retrieve':
                    reporter.collect(result)
                elif action == 'inspect':
                    reporter.progress()
                return result
        return wrapped
    return decorate


class WorkerReporter:
    def __init__(self, worker):
        self.worker = worker
        self.directory = str(worker.directory)
        self.started = time.monotonic()
        self.last_progress = self.started
        self.total = self.batches = None
        self.values = {}
        self.last_summary = None
        self.selected = None
        self.outcomes = {}

    def configure(self, batches):
        selected = {str(name) for batch in batches for name in batch[1]}
        if self.selected != selected:
            self.values.clear()
            self.outcomes.clear()
            self.last_summary = None
        self.selected = selected
        self.batches = len(batches)
        self.total = sum(len(batch[1]) for batch in batches)

    def counts(self):
        jobs = self.worker.job_store.get_queued()
        finished = {job.gdir for job in self.worker.job_store.get_finished()}
        names = {str(name) for job in jobs for name in job.wdir_names}
        selected = self.selected if self.selected is not None else names
        done = {str(name) for job in jobs if job.gdir in finished for name in job.wdir_names} & selected
        failed = {name for name, success in self.outcomes.items() if not success} & selected
        done |= {name for name, success in self.outcomes.items() if success} & selected
        done -= failed
        return len(done), len(selected - done - failed), len(failed)

    def task_finished(self, name, success):
        self.outcomes[str(name)] = success
        if time.monotonic() - self.last_progress >= 30:
            done, pending, failed = self.counts()
            message(f'worker progress: {done} finished | {pending} pending | {failed} failed')
            self.last_progress = time.monotonic()

    def progress(self):
        _, pending, _ = self.counts()
        if pending and (self.last_summary is None or time.monotonic() - self.last_progress >= 30):
            self.summary('waiting')
            self.last_progress = time.monotonic()

    def collect(self, trajectories):
        # Borrow existing results. Never call get_energy/get_forces or load files.
        methods = {getattr(driver.setting, 'task', 'unknown') for driver in self.worker._drivers}
        spc = methods == {'spc'} or all(getattr(driver.setting, 'steps', None) == 0 for driver in self.worker._drivers)
        for index, trajectory in enumerate(trajectories):
            if not trajectory:
                continue
            frame = trajectory[-1]
            results = getattr(frame.calc, 'results', {})
            key = frame.info.get('wdir', frame.info.get('confid', index))
            energy = results.get('energy')
            step = None if spc else frame.info.get('step')
            force = None
            if results.get('forces') is not None:
                forces = np.array(results['forces'], copy=True)
                for constraint in frame.constraints:
                    if hasattr(constraint, 'adjust_forces'):
                        constraint.adjust_forces(frame, forces)
                if len(forces):
                    force = float(np.linalg.norm(forces, axis=1).max())
            self.values[str(key)] = (energy, step, force)
        self.summary('finished')

    @staticmethod
    def stats(values, digits=4, width=0):
        values = [float(value) for value in values if value is not None and np.isfinite(value)]
        if not values:
            return '—'
        return f'min {min(values):{width}.{digits}f}   avg {np.mean(values):{width}.{digits}f}   max {max(values):{width}.{digits}f}'

    def summary(self, state):
        done, pending, failed = self.counts()
        if state == 'finished' and pending:
            state = 'partial'
        if failed:
            state = 'failed'
        signature = (state, done, pending, failed, tuple(sorted(self.values.items())))
        if signature == self.last_summary:
            return
        self.last_summary = signature
        drivers = self.worker._drivers
        methods = sorted({getattr(driver.setting, 'task', 'unknown') for driver in drivers})
        runtime = self.worker.runtime.config.to_dict()
        executor = runtime.get('executor', {}).get('provider', '—')
        potential = runtime.get('potential', {}).get('provider', '—')
        title = 'worker | ' + ', '.join({'min': 'minimization', 'md': 'dynamics', 'spc': 'single point'}.get(m, m) for m in methods)
        total = self.total if self.total is not None else done + pending + failed
        title += f' | calculations: {total} | batches: {self.batches if self.batches is not None else "—"}'
        box = Box(title)
        box.line(f'executor: {executor}   potential: {potential}')
        box.line(f'state: {state}   finished: {done}   pending: {pending}   failed: {failed}')
        if state == 'failed':
            box.line(f'failure details: {self.directory}; see traceback and calculation logs')
        values = list(self.values.values())
        width = max((len(f'{float(value):.{digits}f}') for row in values
                     for value, digits in ((row[0], 4), (row[1], 1), (row[2], 4))
                     if value is not None and np.isfinite(value)), default=0)
        box.line('steps:         ' + self.stats((value[1] for value in values), digits=1, width=width))
        box.line('energy [eV]:   ' + self.stats((value[0] for value in values), width=width))
        box.line('maxfrc [eV/Å]: ' + self.stats((value[2] for value in values), width=width))
        box.line(f'results summarized: {len(values)}   elapsed this invocation: {time.monotonic()-self.started:.1f} s')
        box.border('bottom')
