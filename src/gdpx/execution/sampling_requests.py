"""Durable basic-calculation requests emitted by sampling state machines."""
from contextlib import contextmanager
from contextvars import ContextVar
import copy
from dataclasses import replace
import fcntl
import os
from pathlib import Path

from gdpx.core.output import candidate_logging
from gdpx.execution.accepted_state import load_structure, save_structure
from gdpx.execution.checkpoint import load_data, save_data
from gdpx.execution.executor import RunState
from gdpx.execution.fingerprint import payload_digest, structure_digest

_ACTIVE = ContextVar('sampling_evaluations', default=None)


class EvaluationClient:
    """Request/result adapter; owns no worker, scheduler job, or submission loop."""
    def __init__(self, runtime):
        self.runtime = runtime
        self.directory = Path('.')
        self.wdir_name = '.'

    @property
    def driver(self):
        return self.runtime.executor

    @property
    def scheduler(self):
        return self.runtime.scheduler

    def run(self, frames, rng_states=(), **kwargs):
        if len(frames) != 1:
            raise ValueError('An evaluation request requires exactly one structure.')
        broker = _ACTIVE.get()
        if broker is None:
            raise RuntimeError('Sampling evaluations require a sampling coordinator.')
        broker.request(self.runtime, frames[0], self.directory,
                       rng_states[0] if rng_states else None)

    def inspect(self, **kwargs):
        _ACTIVE.get().bind(self.runtime)  # Only the coordinator submits or polls.

    def get_number_of_running_jobs(self):
        return int(_ACTIVE.get().result(self.directory) is None)

    def retrieve(self, **kwargs):
        result = _ACTIVE.get().result(self.directory)
        if result is None:
            raise RuntimeError('Evaluation is still pending.')
        return [[result]]


def evaluate(runtime, atoms, directory, seed=None):
    broker = _ACTIVE.get()
    if broker is None:
        raise RuntimeError('Sampling evaluations require a sampling coordinator.')
    broker.request(runtime, atoms, directory, seed)
    return broker.result(directory)


class EvaluationBroker:
    """Publish immutable requests and batches before any external submission."""
    def __init__(self, root, batch_size=1):
        self.root = Path(root).resolve()
        self.directory = self.root / '_requests'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'index.json'
        if not self.path.exists() and (self.directory / 'requests.json').exists():
            self.path = self.directory / 'requests.json'
        self.state = load_data(self.path) if self.path.exists() else dict(version=2, requests={}, batches={})
        if self.state.get('version') != 2:
            raise ValueError('Unsupported sampling request ledger; start a new run.')
        self.batch_size = batch_size
        self.runtimes = {}
        self.chain_seed = 0
        self.chain_directory = self.root

    def save(self):
        save_data(self.path, self.state)

    def key(self, directory):
        return str(Path(directory).resolve().relative_to(self.root))

    def next_id(self, kind):
        """Allocate within each namespace without reusing persisted or orphan IDs."""
        identifiers = (self.state.get(kind, {}) if kind in {'batches', 'runtimes'} else
                       [row['id'] for row in self.state['requests'].values()])
        names = set(identifiers) | {path.stem if kind == 'runtimes' else path.name
                                    for path in (self.directory / kind).glob('*')}
        sequence = max((int(name) for name in names if name.isdecimal()), default=0) + 1
        return f'{sequence:06d}'

    def bind(self, runtime):
        config = runtime.config.to_dict()
        config['dispatch'] = dict(worker='batch', batch_size=self.batch_size, share_workdir=False, retain_info=False)
        self.runtimes[payload_digest(config)] = runtime
        return config

    def request(self, runtime, atoms, directory, seed=None):
        key = self.key(directory)
        config = self.bind(runtime)
        # Stable seeds are explicit task inputs, never regenerated on restore.
        if seed is None:
            seed = config['executor'].get('parameters', {}).get('random_seed')
        if seed is None:
            seed = int(payload_digest(dict(key=key, chain_seed=self.chain_seed))[:8], 16)
        runtime_digest = payload_digest(config)
        signature = payload_digest(dict(runtime=config, structure=structure_digest([atoms]), seed=seed))
        existing = self.state['requests'].get(key)
        if existing is not None:
            if existing['signature'] != signature:
                raise ValueError(f'Sampling evaluation request changed: {key}. Start a new run.')
            return
        runtimes = self.state.setdefault('runtimes', {})
        runtime_id = next((key for key, digest in runtimes.items() if digest == runtime_digest), None)
        if runtime_id is None:
            # Reuse runtime files belonging to runs created before numeric names.
            runtime_id = runtime_digest if (self.directory / 'runtimes' / f'{runtime_digest}.json').exists() else self.next_id('runtimes')
            runtimes[runtime_id] = runtime_digest
        identifier = self.next_id('requests')
        path = self.directory / 'requests' / identifier
        path.mkdir(parents=True, exist_ok=True)
        save_structure(path / 'input.json', atoms)
        runtime_path = self.directory / 'runtimes' / f'{runtime_id}.json'
        runtime_path.parent.mkdir(exist_ok=True)
        if not runtime_path.exists():
            save_data(runtime_path, config)
        self.state['requests'][key] = dict(id=identifier, signature=signature, runtime=runtime_id,
                                           seed=seed, chain=self.key(self.chain_directory), batch=None)
        self.save()

    def ready(self, row):
        return (self.directory / 'requests' / row['id'] / 'result.json').is_file()

    def result(self, directory):
        row = self.state['requests'].get(self.key(directory))
        if row is None or not self.ready(row):
            return None
        return load_structure(self.directory / 'requests' / row['id'] / 'result.json')

    def dispatch(self):
        """Batch compatible outstanding requests, then poll each batch independently."""
        from gdpx.execution.factory import create_worker
        grouped = {}
        for key, row in self.state['requests'].items():
            if row['batch'] is None:
                grouped.setdefault(row['runtime'], []).append(key)
        # Persist every assignment before preparing/submitting any batch. On
        # restart the same batch, job UUIDs, inputs, and seeds are reused.
        for keys in grouped.values():
            for offset in range(0, len(keys), self.batch_size):
                members = keys[offset:offset + self.batch_size]
                identifier = self.next_id('batches')
                self.state['batches'][identifier] = dict(members=members)
                for key in members:
                    self.state['requests'][key]['batch'] = identifier
        self.save()
        changed = False
        for identifier, batch in self.state['batches'].items():
            rows = [self.state['requests'][key] for key in batch['members']]
            if all(self.ready(row) for row in rows):
                continue
            root = self.directory / 'batches' / identifier
            runtime_path = self.directory / 'runtimes' / f"{rows[0]['runtime']}.json"
            config = load_data(runtime_path)
            runtime = self.runtimes.get(payload_digest(config))
            if runtime is not None:
                from gdpx.providers.configuration import RuntimeConfig
                runtime = replace(runtime, config=RuntimeConfig.from_mapping(config))
            worker = create_worker(runtime or config, directory=root)
            worker.task_directories = [os.path.relpath(self.root / key, root)
                                       for key in batch['members']]

            relative = lambda path: os.path.relpath(path, root / '_meta')
            worker.evaluation_sources = dict(
                runtime=dict(file=relative(runtime_path), digest=payload_digest(config)),
                structures=[relative(self.directory / 'requests' / row['id'] / 'input.json') for row in rows],
                results={name: relative(self.directory / 'requests' / row['id'] / 'result.json')
                         for name, row in zip(worker.task_directories, rows)},
            )
            frames = [load_structure(self.directory / 'requests' / row['id'] / 'input.json') for row in rows]
            for key, row in zip(batch['members'], rows):
                if self.ready(row):
                    continue
                task_directory = self.root / key
                task_directory.mkdir(parents=True, exist_ok=True)
                (task_directory / '.sampling-request').touch()
            # Batch diagnostics remain out of the controller's gdp.out.
            with candidate_logging(root):
                worker.run(frames, rng_states=[row['seed'] for row in rows])
                worker.inspect(resubmit=True)
                # A finished task can release its chain before another job in
                # the batch finishes. Queued jobs are never resubmitted here.
                for key, row in zip(batch['members'], rows):
                    if self.ready(row):
                        continue
                    source = self.root / key
                    worker.driver.directory = source
                    if not worker.driver.read_convergence():
                        continue
                    trajectory = worker.driver.read_trajectory()
                    if not trajectory:
                        raise RuntimeError(f'Completed evaluation has no trajectory: {key}')
                    endpoint = trajectory[-1]
                    save_structure(self.directory / 'requests' / row['id'] / 'result.json',
                                   endpoint, copy.deepcopy(endpoint.calc.results))
                    save_data(source / 'evaluation.json', dict(request=key, batch=identifier))
                    log_path = source / 'mc.out'
                    if log_path.exists():
                        chain_log = self.root / row['chain'] / 'mc.out'
                        with chain_log.open('a') as log:
                            log.write(f'\nEvaluation: {key}\n')
                            log.write(log_path.read_text())
                    changed = True
        return changed

    def advance(self, chains):
        token = _ACTIVE.set(self)
        try:
            while True:
                states = []
                for engine, atoms, directory, seed in chains:
                    engine.directory = directory
                    engine.set_rng(seed=seed)
                    self.chain_seed = seed
                    self.chain_directory = engine.directory
                    states.append(engine.run(atoms))
                if all(state is RunState.FINISHED for state in states):
                    return RunState.FINISHED
                if not self.dispatch():
                    return RunState.PENDING
        finally:
            _ACTIVE.reset(token)


@contextmanager
def coordinator(root, batch_size=1):
    directory = Path(root) / '_requests'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another sampling coordinator is advancing this worker.') from error
        try:
            yield EvaluationBroker(root, batch_size)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def run_standalone(engine, atoms):
    """Keep direct executor calls usable through the same single-chain coordinator."""
    if _ACTIVE.get() is not None:
        return None
    with coordinator(engine.directory) as broker:
        return broker.advance([(engine, atoms, engine.directory, engine.random_seed)])


def evaluation_workers(directory):
    """Find the basic batch workers referenced by this chain's request ledger."""
    from gdpx.execution.factory import create_worker
    directory = Path(directory).resolve()
    for root in (directory, *directory.parents):
        ledger = root / '_requests/index.json'
        if not ledger.exists():
            ledger = root / '_requests/requests.json'
        if not ledger.exists():
            continue
        state = load_data(ledger)
        groups = {}
        for key, row in state['requests'].items():
            if (root / key).is_relative_to(directory) and row['batch'] is not None:
                groups[row['batch']] = load_data(root / '_requests/runtimes' / f"{row['runtime']}.json")
        return [create_worker(runtime, directory=root / '_requests/batches' / batch)
                for batch, runtime in groups.items()]
    return None
