"""Sampling schedules basic evaluations across independent chains."""
import copy
import json

import numpy as np
import pytest
from ase.io import read

from gdpx.execution.checkpoint import load_data
from gdpx.execution.factory import create_worker
from gdpx.execution.lifecycle import prepare_compute, inspect_compute, resubmit_compute, collect_compute
from gdpx.execution.lifecycle.service import orchestrate_compute
from gdpx.execution.schedulers.direct import DirectScheduler
from gdpx.execution.workers.monte_carlo import MonteCarloWorker
from test_sampling_config import sampling_config
from test_sampling_lifecycle import copper


@pytest.mark.parametrize('method,basic', [('monte_carlo', 'spc'), ('biased_monte_carlo', 'min'),
                                         ('hybrid_monte_carlo', 'md')])
def test_batches_basic_calculations_instead_of_chains(tmp_path, method, basic):
    config = sampling_config(method)
    config['dispatch'] = {'batch_size': 2}
    config['executor']['parameters']['stop']['steps'] = 1
    assert isinstance(create_worker(config), MonteCarloWorker)
    plan = prepare_compute(config, [copper(), copper(), copper()], tmp_path)
    assert len(plan.workers[0].batches) == 1  # One coordinator, not one scheduler job.
    assert orchestrate_compute(plan).state == 'finished'
    ledger = load_data(tmp_path / '_requests/requests.json')
    assert all((tmp_path / '_requests/requests' / row['id'] / 'result.json').is_file()
               for row in ledger['requests'].values())
    methods = {load_data(tmp_path / '_requests/runtimes' / f"{row['runtime']}.json")['executor']['method']
               for row in ledger['requests'].values()}
    assert basic in methods
    assert not methods.intersection({'monte_carlo', 'biased_monte_carlo', 'hybrid_monte_carlo'})
    assert {len(batch['members']) for batch in ledger['batches'].values()} == {1, 2}
    for batch in ledger['batches'].values():
        requests = [ledger['requests'][key] for key in batch['members']]
        assert all(row['runtime'] == requests[0]['runtime'] for row in requests)
    for i in range(3):
        assert (tmp_path / f'cand{i}/mc.out').is_file()
        assert (tmp_path / f'cand{i}/steps/step.0000/traj.xyz').is_file()
    assert not (tmp_path / '_sampling').exists()
    assert not list((tmp_path / '_meta').rglob('traj.xyz'))
    assert not list((tmp_path / '_requests').rglob('traj.xyz'))
    assert not list((tmp_path / '_requests/batches').glob('*/cand*'))
    assert len(list((tmp_path / '_requests/runtimes').glob('*.json'))) == len(
        {row['runtime'] for row in ledger['requests'].values()})
    assert all('ready' not in row for row in ledger['requests'].values())
    assert all('complete' not in batch for batch in ledger['batches'].values())
    assert not list((tmp_path / '_requests').rglob('completed-evaluations.json'))
    assert not list((tmp_path / '_requests/batches').rglob('structures-*.json'))
    for path in (tmp_path / '_requests/batches').glob('*/_meta/inputs.json'):
        catalog = json.loads(path.read_text())
        for job in catalog['jobs'].values():
            assert job['input']['runtime'] == {'runtime_ref': catalog['evaluation_sources']['runtime']['digest']}
        for record in catalog['workers'].values():
            for batch in record['calculation_set']['batches']:
                assert set(batch['runtime']) == {'runtime_ref'}


def test_ready_chains_advance_while_other_batches_wait_and_resume(tmp_path, monkeypatch):
    config = sampling_config()
    config['dispatch'] = {'batch_size': 2}
    config['executor']['parameters']['stop']['steps'] = 2
    baseline = prepare_compute(config, [copper()] * 3, tmp_path / 'baseline')
    assert orchestrate_compute(baseline).state == 'finished'
    root = tmp_path / 'deferred'
    plan = prepare_compute(config, [copper()] * 3, root)
    submit, finished = DirectScheduler.submit, DirectScheduler.is_finished
    callbacks = {}
    delayed = set()
    submissions = []

    def deferred_submit(scheduler, func_to_execute=None):
        path = str(scheduler.script)
        if str(root / '_requests/batches') in path:
            submissions.append(path)
            # Hold the first basic batch; let the third chain keep advancing.
            if not delayed:
                delayed.add(path)
                callbacks[path] = func_to_execute
                return 'deferred'
        return submit(scheduler, func_to_execute)

    monkeypatch.setattr(DirectScheduler, 'submit', deferred_submit)
    monkeypatch.setattr(DirectScheduler, 'is_finished', lambda scheduler:
                        str(scheduler.script) not in callbacks and finished(scheduler))
    assert orchestrate_compute(plan).state == 'running'
    assert not (root / 'cand0/current.json').exists()
    assert len(read(root / 'cand2/mc.xyz', ':')) == 3
    before = copy.deepcopy(load_data(root / '_requests/requests.json'))
    resubmit_compute(plan)
    assert load_data(root / '_requests/requests.json') == before
    assert len(submissions) == len(set(submissions))
    pending = list(callbacks.values())
    callbacks.clear()
    for callback in pending:
        callback()
    resubmit_compute(plan)
    assert inspect_compute(plan).state == 'finished'
    collect_compute(plan)
    assert len(submissions) == len(set(submissions))
    for i in range(3):
        actual = read(root / f'cand{i}/mc.xyz', ':')
        expected = read(tmp_path / f'baseline/cand{i}/mc.xyz', ':')
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a.positions, b.positions)
        assert len(actual) == len(expected)
        assert load_data(root / f'cand{i}/checkpoint.2/state.json')['rng'] == load_data(
            tmp_path / f'baseline/cand{i}/checkpoint.2/state.json')['rng']


def test_parent_scheduler_is_inherited_by_basic_requests(tmp_path):
    config = sampling_config('hybrid_monte_carlo')
    config['scheduler'] = {'provider': 'slurm', 'parameters': {'is_dry_run': True, 'concurrent_tasks': 2}}
    config['dispatch'] = {'batch_size': 2}
    plan = prepare_compute(config, [copper(), copper()], tmp_path)
    assert orchestrate_compute(plan).state == 'running'
    ledger = load_data(tmp_path / '_requests/requests.json')
    assert len(ledger['batches']) == 1
    row = next(iter(ledger['requests'].values()))
    runtime = load_data(tmp_path / '_requests/runtimes' / f"{row['runtime']}.json")
    assert runtime['executor']['method'] == 'spc'
    assert runtime['scheduler']['provider'] == 'slurm'
    scripts = list((tmp_path / '_requests/batches').glob('*/_meta/jobscripts/*.script'))
    assert len(scripts) == 1
    assert 'concurrent_tasks=2' in scripts[0].read_text()


def test_assignment_survives_interruption_before_submission(tmp_path, monkeypatch):
    from gdpx.execution.workers.drive import DriverBasedWorker
    from gdpx.sampling.acceptance import AcceptanceRule

    config = sampling_config()
    config['dispatch'] = {'batch_size': 2}
    config['executor']['parameters']['stop']['steps'] = 1
    plan = prepare_compute(config, [copper(), copper()], tmp_path)
    original = DriverBasedWorker.run
    seen = []

    def interrupt_once(worker, *args, **kwargs):
        if '_requests' in worker.directory.parts and not seen:
            seen.append(str(worker.directory))
            raise KeyboardInterrupt()
        return original(worker, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(DriverBasedWorker, 'run', interrupt_once)
        with pytest.raises(KeyboardInterrupt):
            orchestrate_compute(plan)
    before = load_data(tmp_path / '_requests/requests.json')
    assert len(before['batches']) == 1
    assert all(row['batch'] is not None for row in before['requests'].values())
    decisions = []
    accept = AcceptanceRule.accept

    def record_accept(self, *args, **kwargs):
        decisions.append(1)
        return accept(self, *args, **kwargs)

    monkeypatch.setattr(AcceptanceRule, 'accept', record_accept)
    assert orchestrate_compute(plan).state == 'finished'
    after = load_data(tmp_path / '_requests/requests.json')
    assert set(before['batches']) <= set(after['batches'])
    assert len(decisions) == 2
    assert orchestrate_compute(plan).state == 'finished'
    assert len(decisions) == 2


def test_compatibility_includes_surface_and_execution_settings(tmp_path):
    from gdpx.execution import resolve_runtime
    from gdpx.execution.sampling_requests import coordinator

    configurations = []
    base = {'potential': {'provider': 'emt'}, 'executor': {'provider': 'ase', 'method': 'spc'}}
    configurations.append(base)
    modified = copy.deepcopy(base)
    modified['executor']['parameters'] = {'setup': {'constraint': '`index 0`'}}
    configurations.append(modified)
    modified = copy.deepcopy(base)
    modified['modifiers'] = [{'provider': 'builtin', 'method': 'distance_harmonic',
                             'parameters': {'group': [0, 1], 'center': 2., 'kspring': .1}}]
    configurations.append(modified)
    with coordinator(tmp_path, batch_size=3) as broker:
        for i, config in enumerate(configurations):
            broker.request(resolve_runtime(config), copper(), tmp_path / f'steps/{i}', seed=17)
        assert broker.dispatch()
        assert len(broker.state['batches']) == 3
        assert all(len(batch['members']) == 1 for batch in broker.state['batches'].values())


def test_interruption_after_decision_replays_output_once(tmp_path, monkeypatch):
    from gdpx.execution.monte_carlo.monte_carlo import MonteCarlo
    from gdpx.sampling.acceptance import AcceptanceRule

    config = sampling_config()
    baseline = prepare_compute(config, [copper()], tmp_path / 'baseline')
    assert orchestrate_compute(baseline).state == 'finished'
    plan = prepare_compute(config, [copper()], tmp_path / 'interrupted')
    save_checkpoint = MonteCarlo._save_checkpoint
    accept = AcceptanceRule.accept
    decisions = []

    def record_accept(self, *args, **kwargs):
        decisions.append(1)
        return accept(self, *args, **kwargs)

    def interrupt(self, step, force=False):
        if step == 1:
            raise KeyboardInterrupt()
        return save_checkpoint(self, step, force)

    monkeypatch.setattr(AcceptanceRule, 'accept', record_accept)
    with monkeypatch.context() as patch:
        patch.setattr(MonteCarlo, '_save_checkpoint', interrupt)
        with pytest.raises(KeyboardInterrupt):
            orchestrate_compute(plan)
    assert len(decisions) == 1
    assert orchestrate_compute(plan).state == 'finished'
    assert len(decisions) == 2
    for name in ('mc.xyz', 'opstat.txt', 'mc_attempts.xyz'):
        assert (tmp_path / 'interrupted' / name).read_bytes() == (tmp_path / 'baseline' / name).read_bytes()
    assert load_data(tmp_path / 'interrupted/checkpoint.2/state.json')['rng'] == load_data(
        tmp_path / 'baseline/checkpoint.2/state.json')['rng']


def test_compute_run_reports_pending_basic_jobs(tmp_path):
    from gdpx.execution.lifecycle.service import run_compute_batch

    config = sampling_config()
    config['scheduler'] = {'provider': 'slurm', 'parameters': {'is_dry_run': True}}
    plan = prepare_compute(config, [copper()], tmp_path)
    assert not run_compute_batch(plan, 0).finished


def test_saved_batch_uses_step_directories_and_skips_pruned_results(tmp_path):
    import shutil
    from gdpx.execution import resolve_runtime
    from gdpx.execution.sampling_requests import coordinator, evaluation_workers

    config = {'potential': {'provider': 'emt'},
              'executor': {'provider': 'ase', 'method': 'spc'},
              'scheduler': {'provider': 'slurm', 'parameters': {'is_dry_run': True}}}
    with coordinator(tmp_path, batch_size=2) as broker:
        for index in range(2):
            broker.request(resolve_runtime(config), copper(),
                           tmp_path / f'cand{index}/steps/step.0000', seed=17)
        assert not broker.dispatch()
        worker = evaluation_workers(tmp_path)[0]
        job = worker.job_store.get_queued()[0]
        worker.run_saved_job(job.uid, task=0)
        assert broker.dispatch()
        first = tmp_path / 'cand0/steps/step.0000'
        assert (first / 'traj.xyz').exists()
        shutil.rmtree(first)
        worker.run_saved_job(job.uid)
        assert broker.dispatch()
        assert not first.exists()
        assert (tmp_path / 'cand1/steps/step.0000/traj.xyz').exists()
        assert all(broker.ready(row) for row in broker.state['requests'].values())
        assert not list((tmp_path / '_meta').rglob('traj.xyz'))
        assert not list((tmp_path / '_requests').rglob('traj.xyz'))


@pytest.mark.parametrize('source', ['runtime', 'structure'])
def test_saved_evaluation_rejects_modified_shared_sources(tmp_path, source):
    from gdpx.execution import resolve_runtime
    from gdpx.execution.accepted_state import load_structure, save_structure
    from gdpx.execution.checkpoint import save_data
    from gdpx.execution.sampling_requests import coordinator, evaluation_workers

    config = {'potential': {'provider': 'emt'}, 'executor': {'provider': 'ase', 'method': 'spc'},
              'scheduler': {'provider': 'slurm', 'parameters': {'is_dry_run': True}}}
    with coordinator(tmp_path) as broker:
        broker.request(resolve_runtime(config), copper(), tmp_path / 'steps/step.0000', seed=17)
        assert not broker.dispatch()
        worker = evaluation_workers(tmp_path)[0]
        job = worker.job_store.get_queued()[0]
        row = next(iter(broker.state['requests'].values()))
        if source == 'runtime':
            path = tmp_path / '_requests/runtimes' / f"{row['runtime']}.json"
            changed = load_data(path)
            changed['potential']['provider'] = 'modified'
            save_data(path, changed)
        else:
            path = tmp_path / '_requests/requests' / row['id'] / 'input.json'
            atoms = load_structure(path)
            atoms.positions[0, 0] += .1
            save_structure(path, atoms)
        with pytest.raises(ValueError, match='fingerprint mismatch'):
            worker.run_saved_job(job.uid)
        assert not (tmp_path / 'steps/step.0000/traj.xyz').exists()
