"""BH evaluation is batch-oriented and independent of driver implementations."""
import copy
import json

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read

from gdpx.exploration.basin_hopping import chain
from gdpx.exploration.generation import EvaluationStatus
from gdpx.exploration.sampling import parse_operators


def atoms():
    frame = Atoms("Cu2", positions=[[5., 5., 5.], [7.4, 5., 5.]], tags=[1, 2], cell=[20]*3)
    frame.info.update(confid=42, data={"parents": [42]})
    frame.calc = SinglePointCalculator(frame, energy=0., forces=np.zeros((2, 3)))
    return frame


def operators():
    return parse_operators([dict(method="move", particles=["Cu"], max_disp=.05, skip_distance_check=True)])


class Worker:
    """Only the public worker lifecycle; deliberately no driver attribute."""
    def __init__(self, energy=0., pending=False, fail_at=None, result_mode=None):
        self.energy, self.pending, self.fail_at = energy, pending, fail_at
        self.result_mode = result_mode
        self.batches = []
        self.identities = []

    def run(self, frames):
        self.batches.append([a.info["confid"] for a in frames])
        self.identities.append([id(a) for a in frames])
        if len(self.batches) == self.fail_at:
            raise RuntimeError("interrupted submission")
        self.results = []
        for frame in frames:
            result = Atoms(frame.numbers, positions=frame.positions, cell=frame.cell, pbc=frame.pbc,
                           tags=frame.get_tags())
            result.info = copy.deepcopy(frame.info)
            result.calc = SinglePointCalculator(result, energy=self.energy, forces=np.zeros((len(result), 3)))
            self.results.append([result])

    def inspect(self, resubmit=False):
        pass

    def get_number_of_running_jobs(self):
        return int(self.pending)

    def retrieve(self, include_retrieved=False, use_archive=False):
        assert include_retrieved
        results = list(reversed(self.results))
        if self.result_mode == "missing":
            return results[:-1]
        if self.result_mode == "duplicate":
            return results + results[:1]
        if self.result_mode == "unexpected":
            results[0][-1].info["confid"] = 99
        return results


def run(path, worker, rng=None, starts=None, steps=3, ops=None, on_progress=None):
    ops, probs = operators() if ops is None else (ops, [1.])
    return chain.run_hopping_rounds(starts if starts is not None else [atoms(), atoms()], worker,
                                    ops, probs, steps, rng if rng is not None else np.random.default_rng(9), path,
                                    on_progress=on_progress)


@pytest.mark.parametrize("energy,decision", [(0., 0), (100., 1)])
def test_batched_acceptance_and_borrowing(tmp_path, monkeypatch, energy, decision):
    starts = [atoms(), atoms()]
    originals = [a.positions.copy() for a in starts]
    worker = Worker(energy)
    # ASE calculators/I/O may copy; BH itself must submit borrowed structures.
    import inspect
    original_copy = Atoms.copy
    def checked_copy(frame):
        caller = inspect.currentframe().f_back.f_globals.get("__name__", "")
        assert not caller.startswith("gdpx.exploration.basin_hopping")
        return original_copy(frame)
    monkeypatch.setattr(Atoms, "copy", checked_copy)
    outcome = run(tmp_path / "rounds", worker, starts=starts)
    assert outcome.status is EvaluationStatus.FINISHED
    assert worker.batches == [[0, 1]] * 3
    assert worker.identities[0] == [id(a) for a in starts]
    for frame, original in zip(starts, originals):
        np.testing.assert_array_equal(frame.positions, original)
        assert frame.info["confid"] == 42
    events = [json.loads(line) for line in (tmp_path / "rounds/events.jsonl").read_text().splitlines()]
    assert [event["decisions"] for event in events[1:]] == [[decision, decision]] * 3
    if decision == 1:
        assert all(a is b for a, b in zip(outcome.endpoints, starts))


def test_pending_round_reuses_inputs_and_rng(tmp_path, monkeypatch):
    baseline_rng = np.random.default_rng(9)
    expected = run(tmp_path / "baseline/rounds", Worker(), rng=baseline_rng)
    worker = Worker(pending=True)
    starts = [atoms(), atoms()]
    outcome = run(tmp_path / "restart/rounds", worker, starts=starts)
    assert outcome.status is EvaluationStatus.PENDING
    for a in starts:
        assert a.info["confid"] == 42
        assert a.get_potential_energy() == 0.
    op = operators()[0][0]
    original = op.propose
    calls = []
    def propose(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(op, "propose", propose)
    rng = np.random.default_rng(123)
    resumed = run(tmp_path / "restart/rounds", Worker(), rng=rng, ops=[op])
    assert len(calls) == 4  # Only rounds 2 and 3, not the persisted first trial.
    for actual, wanted in zip(resumed.endpoints, expected.endpoints):
        np.testing.assert_array_equal(actual.positions, wanted.positions)
    assert rng.bit_generator.state == baseline_rng.bit_generator.state
    from gdpx.exploration.basin_hopping import export_trajectories
    assert not (tmp_path / "restart/mctrajs").exists()
    paths = export_trajectories(tmp_path / "restart/rounds", tmp_path / "export")
    assert len(read(paths[0], ":")) == 4


@pytest.mark.parametrize("boundary", ["submission", "before_commit", "after_commit"])
def test_interrupted_round_matches_uninterrupted(tmp_path, monkeypatch, boundary):
    rng = np.random.default_rng(9)
    expected = run(tmp_path / "baseline/rounds", Worker(), rng=rng)
    starts = [atoms(), atoms()]
    worker = Worker(fail_at=2 if boundary == "submission" else None)
    original = chain._commit
    def interrupt(directory, step, *args):
        if step == 2 and boundary == "before_commit":
            raise RuntimeError("interrupted commit")
        original(directory, step, *args)
        if step == 2 and boundary == "after_commit":
            raise RuntimeError("interrupted commit")
    monkeypatch.setattr(chain, "_commit", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        run(tmp_path / "restart/rounds", worker, starts=starts)
    for frame in starts:
        assert frame.info["confid"] == 42
    monkeypatch.setattr(chain, "_commit", original)
    resumed_rng = np.random.default_rng(123)
    resumed_worker = Worker()
    actual = run(tmp_path / "restart/rounds", resumed_worker, rng=resumed_rng)
    assert len(resumed_worker.batches) == (1 if boundary == "after_commit" else 2)
    for a, b in zip(actual.endpoints, expected.endpoints):
        np.testing.assert_array_equal(a.positions, b.positions)
    assert resumed_rng.bit_generator.state == rng.bit_generator.state


@pytest.mark.parametrize("mode", ["missing", "duplicate", "unexpected"])
def test_result_ids_are_validated(tmp_path, mode):
    if mode == "missing":
        assert run(tmp_path / "rounds", Worker(result_mode=mode)).status is EvaluationStatus.PENDING
        assert not (tmp_path / "rounds/round-000001").exists()
    else:
        with pytest.raises(RuntimeError, match="unexpected or duplicate"):
            run(tmp_path / "rounds", Worker(result_mode=mode))


@pytest.mark.parametrize("steps", [0, 2])
def test_invalid_and_zero_hops_skip_worker(tmp_path, monkeypatch, steps):
    op = operators()[0][0]
    from gdpx.exploration.sampling.proposal import MoveProposal
    def invalid(frame, rng):
        proposal = MoveProposal(frame)
        proposal.valid = False
        proposal.rollback()
        return proposal
    monkeypatch.setattr(op, "propose", invalid)
    worker = Worker()
    starts = [atoms(), atoms()]
    outcome = run(tmp_path / "rounds", worker, starts=starts, steps=steps, ops=[op])
    assert not worker.batches
    assert outcome.endpoints == starts
    assert outcome.status is EvaluationStatus.FINISHED


def test_mixed_validity_and_submission_exception_roll_back(tmp_path, monkeypatch):
    op = operators()[0][0]
    original = op.propose
    starts = [atoms(), atoms()]
    starts[0].info["skip"] = True
    from gdpx.exploration.sampling.proposal import MoveProposal
    def propose(frame, rng):
        if frame.info.get("skip"):
            proposal = MoveProposal(frame)
            proposal.valid = False
            proposal.rollback()
            return proposal
        return original(frame, rng)
    monkeypatch.setattr(op, "propose", propose)
    worker = Worker(fail_at=1)
    originals = [a.positions.copy() for a in starts]
    with pytest.raises(RuntimeError, match="submission"):
        run(tmp_path / "rounds", worker, starts=starts, steps=1, ops=[op])
    assert worker.batches == [[1]]
    for frame, positions in zip(starts, originals):
        np.testing.assert_array_equal(frame.positions, positions)
        assert frame.info["confid"] == 42
        assert frame.get_potential_energy() == 0.
    resumed = run(tmp_path / "rounds", Worker(), starts=starts, steps=1, ops=[op])
    np.testing.assert_array_equal(resumed.endpoints[0].positions, originals[0])
    assert resumed.status is EvaluationStatus.FINISHED


def test_failure_before_submission_replays_round(tmp_path, monkeypatch):
    baseline_rng = np.random.default_rng(9)
    expected = run(tmp_path / "baseline/rounds", Worker(), rng=baseline_rng)
    save = chain._save
    def interrupt(path, value):
        save(path, value)
        if path.name == "proposal.json":
            raise RuntimeError("interrupted proposal persistence")
    monkeypatch.setattr(chain, "_save", interrupt)
    starts = [atoms(), atoms()]
    worker = Worker()
    with pytest.raises(RuntimeError, match="persistence"):
        run(tmp_path / "restart/rounds", worker, starts=starts)
    assert not worker.batches
    assert all(a.info["confid"] == 42 for a in starts)
    monkeypatch.setattr(chain, "_save", save)
    rng = np.random.default_rng(123)
    actual = run(tmp_path / "restart/rounds", Worker(), rng=rng)
    for a, b in zip(actual.endpoints, expected.endpoints):
        np.testing.assert_array_equal(a.positions, b.positions)
    assert rng.bit_generator.state == baseline_rng.bit_generator.state


def test_failed_retrieval_does_not_repropose(tmp_path, monkeypatch):
    worker = Worker()
    def fail(**kwargs):
        raise RuntimeError("interrupted retrieval")
    monkeypatch.setattr(worker, "retrieve", fail)
    with pytest.raises(RuntimeError, match="retrieval"):
        run(tmp_path / "rounds", worker, steps=1)
    op = operators()[0][0]
    monkeypatch.setattr(op, "propose", lambda *a: pytest.fail("reproposed persisted trial"))
    assert run(tmp_path / "rounds", Worker(), steps=1, ops=[op]).status is EvaluationStatus.FINISHED


def test_bh_retention_and_recovery_from_damaged_latest_snapshot(tmp_path):
    baseline_rng = np.random.default_rng(9)
    expected = run(tmp_path / 'baseline/rounds', Worker(), rng=baseline_rng, steps=8)
    path = tmp_path / 'restart/rounds'
    actual_rng = np.random.default_rng(9)
    run(path, Worker(), rng=actual_rng, steps=8)
    assert len(list(path.glob('round-*'))) == 2
    assert not list(path.glob('pending-*'))
    assert len((path / 'events.jsonl').read_text().splitlines()) == 9
    (path / 'round-000008/state.json').write_text('{broken')
    # Uncommitted journal data is discarded when the recovered round commits.
    with (path / 'events.jsonl').open('ab') as stream:
        stream.write(b'partial journal tail')
    worker = Worker()
    resumed_rng = np.random.default_rng(999)
    actual = run(path, worker, rng=resumed_rng, steps=8)
    assert len(worker.batches) == 1
    for a, b in zip(actual.endpoints, expected.endpoints):
        np.testing.assert_array_equal(a.positions, b.positions)
    assert resumed_rng.bit_generator.state == baseline_rng.bit_generator.state
    assert len(list(path.glob('round-*'))) == 2
    assert len((path / 'events.jsonl').read_text().splitlines()) == 9
    from gdpx.exploration.basin_hopping import export_trajectories
    assert not (path.parent / 'mctrajs').exists()
    assert len(read(export_trajectories(path, tmp_path / 'export')[0], ':')) == 9


@pytest.mark.parametrize('boundary', ['manifest', 'prune'])
def test_bh_interrupted_publication_and_pruning(tmp_path, monkeypatch, boundary):
    import gdpx.exploration.checkpoint as checkpoint
    baseline_rng = np.random.default_rng(9)
    baseline = run(tmp_path / 'baseline/rounds', Worker(), rng=baseline_rng)
    path = tmp_path / 'restart/rounds'
    with monkeypatch.context() as patch:
        if boundary == 'manifest':
            write = checkpoint.write_json
            def fail(target, data):
                if target.name == 'current.json' and data['snapshots'][0] == 'round-000002':
                    raise RuntimeError('interrupted manifest')
                return write(target, data)
            patch.setattr(checkpoint, 'write_json', fail)
        else:
            remove = checkpoint.shutil.rmtree
            def fail(target, *args, **kwargs):
                if target.name == 'round-000000':
                    raise RuntimeError('interrupted prune')
                return remove(target, *args, **kwargs)
            patch.setattr(checkpoint.shutil, 'rmtree', fail)
        with pytest.raises(RuntimeError, match='interrupted'):
            run(path, Worker())
    rng = np.random.default_rng(123)
    actual = run(path, Worker(), rng=rng)
    for a, b in zip(actual.endpoints, baseline.endpoints):
        np.testing.assert_array_equal(a.positions, b.positions)
    assert rng.bit_generator.state == baseline_rng.bit_generator.state
    assert len(list(path.glob('round-*'))) == 2
    assert not list(path.glob('pending-*'))


@pytest.mark.parametrize('energy,expected_count', [(-1., 4), (1.e6, 1)])
def test_export_committed_history_after_finalization(tmp_path, energy, expected_count):
    from gdpx.exploration.basin_hopping import export_trajectories
    directory = tmp_path / 'rounds'
    run(directory, Worker(energy=energy))
    assert not (tmp_path / 'mctrajs').exists()
    paths = export_trajectories(directory, tmp_path / 'export')
    before = [path.read_bytes() for path in paths]
    assert all(len(read(path, ':')) == expected_count for path in paths)
    chain.finalize_checkpoints(directory)
    assert not list(directory.glob('round-*'))
    with (directory / 'events.jsonl').open('ab') as stream:
        stream.write(b'uncommitted tail')
    assert export_trajectories(directory, tmp_path / 'export') == paths
    assert [path.read_bytes() for path in paths] == before


def test_bh_rattle_batches_and_restart(tmp_path):
    def rattle():
        return parse_operators([dict(method='rattle', particles=['Cu'], rattle_prop=1.,
                                     rattle_strength=.05, skip_distance_check=True)])[0]
    baseline_rng = np.random.default_rng(9)
    expected = run(tmp_path / 'baseline/rounds', Worker(), rng=baseline_rng, ops=rattle())
    assert run(tmp_path / 'restart/rounds', Worker(pending=True), ops=rattle()).status is EvaluationStatus.PENDING
    resumed_rng = np.random.default_rng(100)
    result = run(tmp_path / 'restart/rounds', Worker(), rng=resumed_rng, ops=rattle())
    for actual, wanted in zip(result.endpoints, expected.endpoints):
        np.testing.assert_array_equal(actual.positions, wanted.positions)
    assert resumed_rng.bit_generator.state == baseline_rng.bit_generator.state


def test_progress_observes_commits_and_checkpoint_fallback(tmp_path, monkeypatch):
    expected_rng = np.random.default_rng(9)
    expected = run(tmp_path / 'baseline', Worker(), rng=expected_rng)
    path = tmp_path / 'observed'
    calls = []
    def observe(step, offset, resumed=False):
        manifest = json.loads((path / 'current.json').read_text())
        assert manifest['snapshots'][0] == f'round-{step:06d}'
        committed = (path / 'events.jsonl').read_bytes()[:offset]
        assert json.loads(committed.splitlines()[-1])['step'] == step
        calls.append((step, resumed))
    actual_rng = np.random.default_rng(9)
    actual = run(path, Worker(), rng=actual_rng, on_progress=observe)
    assert calls == [(0, False), (1, False), (2, False), (3, False)]
    assert actual_rng.bit_generator.state == expected_rng.bit_generator.state
    for a, b in zip(actual.endpoints, expected.endpoints):
        np.testing.assert_array_equal(a.positions, b.positions)
    # Recovery announces the last valid commit, not the damaged latest round.
    (path / 'round-000003/state.json').write_text('{broken')
    calls.clear()
    run(path, Worker(), on_progress=observe)
    assert calls == [(2, True), (3, False)]


def test_progress_does_not_report_pending_or_failed_commit(tmp_path, monkeypatch):
    calls = []
    observe = lambda step, offset, resumed=False: calls.append((step, resumed))
    path = tmp_path / 'rounds'
    run(path, Worker(pending=True), on_progress=observe)
    assert calls == [(0, False)]
    commit = chain._commit
    def fail(directory, step, *args, **kwargs):
        if step == 1:
            raise RuntimeError('commit interrupted')
        return commit(directory, step, *args, **kwargs)
    monkeypatch.setattr(chain, '_commit', fail)
    calls.clear()
    with pytest.raises(RuntimeError, match='commit interrupted'):
        run(path, Worker(), on_progress=observe)
    assert calls == [(0, True)]


def test_move_logging_preserves_results_rng_and_resume(tmp_path, monkeypatch):
    from gdpx.exploration.sampling.logging import MoveLog
    rngs = [np.random.default_rng(9), np.random.default_rng(9)]
    ops, probs = operators()
    baseline = chain.run_hopping_rounds([atoms(), atoms()], Worker(), ops, probs, 1, rngs[0], tmp_path / 'baseline')
    worker = Worker(pending=True)
    path = tmp_path / 'logged'
    logfile = tmp_path / 'gen0001.log'
    with MoveLog(logfile, 1, ops, probs) as log:
        pending = chain.run_hopping_rounds([atoms(), atoms()], worker, ops, probs, 1, rngs[1], path, move_logger=log)
    assert pending.status is EvaluationStatus.PENDING
    assert 'committed: accepted' not in logfile.read_text()
    monkeypatch.setattr(ops[0], 'propose', lambda *args: pytest.fail('regenerated pending proposal'))
    worker.pending = False
    with MoveLog(logfile, 1, ops, probs) as log:
        resumed = chain.run_hopping_rounds([atoms(), atoms()], worker, ops, probs, 1, rngs[1], path, move_logger=log)
    assert rngs[0].bit_generator.state == rngs[1].bit_generator.state
    for expected, actual in zip(baseline.endpoints, resumed.endpoints):
        np.testing.assert_array_equal(expected.positions, actual.positions)
    assert (tmp_path / 'baseline/events.jsonl').read_bytes() == (path / 'events.jsonl').read_bytes()
    text = logfile.read_text()
    assert text.count('proposal begins') == 2
    assert text.count('committed: accepted') == 2
    assert 'reusing pending proposals' in text and 'resumed after committed round 0' in text


@pytest.mark.parametrize('valid,energy,word', [(False, 0., 'invalid proposal'), (True, 1e6, 'rejected')])
def test_move_log_invalid_and_rejected(tmp_path, monkeypatch, valid, energy, word):
    from gdpx.exploration.sampling.logging import MoveLog
    from gdpx.exploration.sampling.proposal import MoveProposal
    ops, probs = operators()
    if not valid:
        def invalid(frame, rng):
            proposal = MoveProposal(frame)
            proposal.valid = False
            proposal.rollback()
            return proposal
        monkeypatch.setattr(ops[0], 'propose', invalid)
    worker = Worker(energy=energy)
    logfile = tmp_path / 'moves.log'
    with MoveLog(logfile, 1, ops, probs) as log:
        chain.run_hopping_rounds([atoms()], worker, ops, probs, 1, np.random.default_rng(9),
                                 tmp_path / 'rounds', move_logger=log)
    assert 'committed: ' + word in logfile.read_text()
    if not valid:
        assert not worker.batches
