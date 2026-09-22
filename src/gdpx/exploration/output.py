"""Run-level exploration framing, separate from algorithm progress."""

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import time

from gdpx import config
from gdpx.core.output import Box


_RUN = ContextVar('exploration_output', default=None)


class ExplorationReporter:
    def __init__(self, directory, input_path=None, random_seed=None):
        self.directory = Path(directory).resolve()
        self.input_path = input_path
        self.random_seed = random_seed
        self.started = time.monotonic()
        self.opened = self.closed = False
        self.total = self.pending = None

    def start(self, total=None, scheduler=None):
        if self.opened:
            return
        self.total = total
        box = Box('exploration')
        if self.input_path is not None:
            box.line(f'input: {self.input_path}')
        box.line(f'directory: {self.directory}')
        if total is not None:
            box.line(f'explorations: {total}   scheduler: {scheduler}')
        seed = f'random seed: {self.random_seed}   ' if self.random_seed is not None else ''
        box.line(f'{seed}processors: {config.NJOBS}')
        box.border('bottom')
        self.opened = True

    def finish(self, status, error=None):
        if self.closed:
            return
        self.start()
        box = Box('exploration')
        box.line(f'status: {status}')
        if self.total is not None and self.pending is not None:
            box.line(f'explorations: {self.total - self.pending} complete | {self.pending} pending')
        if error is not None:
            detail = ' '.join(str(error).split())
            box.line(f'error: {type(error).__name__}: {detail[:300]}')
        box.line(f'elapsed this invocation: {time.monotonic() - self.started:.1f} s')
        box.line(f'output directory: {self.directory}')
        box.border('bottom')
        self.closed = True


@contextmanager
def exploration_output(directory, input_path=None, random_seed=None):
    """Share one report across CLI input loading and exploration execution."""
    current = _RUN.get()
    if current is not None:
        yield current
        return
    report = ExplorationReporter(directory, input_path, random_seed)
    token = _RUN.set(report)
    try:
        yield report
    except (Exception, KeyboardInterrupt) as error:
        # Reporting must never replace the original failure or interruption.
        try:
            report.finish('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error)
        except Exception:
            pass
        raise
    else:
        report.finish('complete' if report.pending == 0 else 'waiting')
    finally:
        _RUN.reset(token)
