"""Bounded reporting borrows cached scientific results without evaluating atoms."""
from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms

from gdpx.core.output import Box
from gdpx.execution.output import WorkerReporter


def reporter(tmp_path, method='min', count=2):
    job = SimpleNamespace(gdir='job', wdir_names=[str(i) for i in range(count)])
    worker = SimpleNamespace(directory=tmp_path,
        job_store=SimpleNamespace(get_queued=lambda: [job], get_finished=lambda: [job]),
        _drivers=[SimpleNamespace(setting=SimpleNamespace(task=method, steps=100))],
        runtime=SimpleNamespace(config=SimpleNamespace(to_dict=lambda: {
            'potential': {'provider': 'emt'}, 'executor': {'provider': 'ase'}})))
    output = WorkerReporter(worker)
    output.configure([(None, job.wdir_names)])
    return output


def frame(index, step):
    atoms = Atoms('Cu2', positions=[[0, 0, 0], [2, 0, 0]])
    atoms.info.update(wdir=str(index), step=step)
    atoms.calc = SinglePointCalculator(atoms, energy=-float(index), forces=np.ones((2, 3)))
    return atoms


def test_steps_are_metadata_and_results_are_borrowed(tmp_path, monkeypatch):
    output = reporter(tmp_path)
    frames = [frame(0, 10), frame(1, 90)]
    frames[0].set_constraint(FixAtoms(indices=[0, 1]))
    monkeypatch.setattr(Atoms, 'copy', lambda *args: (_ for _ in ()).throw(AssertionError('copy')))
    monkeypatch.setattr(Atoms, 'get_forces', lambda *args: (_ for _ in ()).throw(AssertionError('evaluate')))
    lines = []
    parent = Box('test', emit=lines.append)
    with parent.as_parent():
        output.collect([[atoms] for atoms in frames])
        first = len(lines)
        output.collect([[atoms] for atoms in frames])
    assert len(lines) == first
    assert 'min 10.0   mean 50.0   max 90.0' in '\n'.join(lines)
    assert output.values['0'][2] == 0
    assert np.all(frames[0].calc.results['forces'] == 1)
    assert 'converged:' not in '\n'.join(lines)
    assert 'step limit:' not in '\n'.join(lines)
    energy = next(line for line in lines if 'energy [eV]:' in line)
    force = next(line for line in lines if 'maxfrc [eV/Å]:' in line)
    for label in ('min ', 'mean ', 'max '):
        assert energy.index(label) == force.index(label)
    assert all(len(line) == parent.width for line in lines)


def test_large_batch_and_spc_have_bounded_output(tmp_path):
    output = reporter(tmp_path, method='spc', count=1000)
    lines = []
    with Box('test', emit=lines.append, unicode=False).as_parent():
        output.collect([[frame(i, 0)] for i in range(1000)])
    assert len(lines) < 20
    assert 'steps: -' in '\n'.join(lines)
    assert all(line.isascii() for line in lines)


def test_selection_excludes_history_and_counts_failures(tmp_path):
    output = reporter(tmp_path)
    output.configure([(None, ['1', '2'])])
    assert output.counts() == (1, 1, 0)
    output.task_finished('2', False)
    assert output.counts() == (1, 0, 1)
    output.configure([(None, ['3'])])
    assert output.counts() == (0, 1, 0)
    assert not output.values


def test_waiting_poll_and_local_progress_are_throttled(tmp_path, monkeypatch):
    import gdpx.execution.output as module
    now = [0.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    output = reporter(tmp_path)
    output.worker.job_store.get_finished = lambda: []
    lines = []
    with Box('test', emit=lines.append).as_parent():
        output.progress()
        first = len(lines)
        output.progress()
        output.task_finished('0', True)
        assert len(lines) == first
        now[0] = 31.0
        output.task_finished('1', True)
    assert 'worker progress: 2 finished | 0 pending | 0 failed' in '\n'.join(lines)
