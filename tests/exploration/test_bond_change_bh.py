"""Relaxed identities, history, acceptance and recovery for changing bonds."""

import copy

import ase.db
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms

from gdpx.exploration.basin_hopping.chain import run_hopping_rounds
from gdpx.exploration.basin_hopping.engine import canonical_candidates_from_worker_results
from gdpx.exploration.generation import EvaluationStatus
from gdpx.exploration.sampling import parse_operators
from gdpx.utils.atoms_tags import get_tags_per_species


def water():
    frame = Atoms("OOHH", positions=[[5, 5, 3], [5, 5, 5], [6, 5, 5], [5, 6, 5]],
                  tags=[0, 1, 1, 1], cell=[20]*3, pbc=True)
    frame.set_constraint(FixAtoms(indices=[0, 1]))
    frame.info["confid"] = 42
    frame.calc = SinglePointCalculator(frame, energy=0., forces=np.zeros((4, 3)))
    return frame


def operators(**settings):
    params = dict(method="bond_change", pairs=[["O", "H"]], bond_cutoffs={"O-H": 1.25},
                  break_fragments=["H2O", "H"], form_fragments=[["OH", "H"], ["substrate", "H"]],
                  allow_isolated=True)
    params.update(settings)
    ops, _ = parse_operators([params])
    for op in ops:
        op._print = op._debug = lambda *args: None
    return ops


class Worker:
    def __init__(self, energy=0., pending=False, relax=None):
        self.energy, self.pending, self.relax = energy, pending, relax

    def run(self, frames):
        self.results = []
        for frame in frames:
            result = frame.copy()
            if self.relax == "recombine" and "bond_change" in frame.info:
                moved = frame.info["bond_change"]["pair"][1]
                result.positions[moved] = water().positions[moved]
            elif self.relax == "split":
                result.positions[2] = [7.5, 5, 5]
            # Deliberately lose tags, constraints and provenance on evaluation.
            result.set_tags(0)
            result.set_constraint()
            result.info = {"confid": frame.info["confid"]}
            result.calc = SinglePointCalculator(result, energy=self.energy, forces=np.zeros((len(result), 3)))
            self.results.append([result])

    def inspect(self, resubmit=False):
        pass

    def get_number_of_running_jobs(self):
        return int(self.pending)

    def retrieve(self, **kwargs):
        return self.results


def run(path, worker, rng=None, ops=None, steps=1):
    ops = operators() if ops is None else ops
    return run_hopping_rounds([water()], worker, ops, [1.]*len(ops), steps,
                              np.random.default_rng(9) if rng is None else rng, path)


@pytest.mark.parametrize("energy,accepted", [(0., True), (100., False)])
def test_acceptance_records_final_tags_without_changing_rejected_parent(tmp_path, energy, accepted):
    result = run(tmp_path, Worker(energy), ops=operators(mode="break"))
    assert result.status is EvaluationStatus.FINISHED
    endpoint = result.endpoints[0]
    assert ("HO" in get_tags_per_species(endpoint)) == accepted
    if not accepted:
        np.testing.assert_array_equal(endpoint.positions, water().positions)
        np.testing.assert_array_equal(endpoint.get_tags(), water().get_tags())
    assert len(endpoint.constraints[0].get_indices()) == 2
    rows = list(ase.db.connect(tmp_path / "history.db").select())
    trial = rows[-1]
    assert trial.data["bond_change"]["operation"] == "break"
    assert "HO" in get_tags_per_species(trial.toatoms())
    assert trial.data["fragment_changes"]["bonds"]


def test_relaxation_recombines_proposed_break_and_preserves_history(tmp_path):
    result = run(tmp_path, Worker(relax="recombine"), ops=operators(mode="break"))
    np.testing.assert_array_equal(result.endpoints[0].get_tags(), [0, 1, 1, 1])
    trial = list(ase.db.connect(tmp_path / "history.db").select())[-1]
    assert trial.data["bond_change"]["operation"] == "break"
    assert trial.data["fragment_changes"]["indices"]
    assert "H2O" in get_tags_per_species(trial.toatoms())
    candidate = result.endpoints[0]
    candidate.info["bond_change"] = copy.deepcopy(trial.data["bond_change"])
    canonical_candidates_from_worker_results([candidate], gen_num=1, use_tags=True)
    assert candidate.info["data"]["bond_change"]["operation"] == "break"
    assert candidate.info["identity_stats"]["H2O"] == 1


def test_rotation_trial_also_reconciles_fragments(tmp_path):
    rotate = parse_operators([dict(method="rotate", particles=["H2O"], center="O",
                                   max_angle=45, axis=[0, 0, 1], skip_distance_check=True)])[0][0]
    bond = operators()[0]
    result = run_hopping_rounds([water()], Worker(relax="split"), [rotate, bond], [1., 0.], 1,
                                np.random.default_rng(9), tmp_path)
    assert "HO" in get_tags_per_species(result.endpoints[0])
    assert "H2O" not in get_tags_per_species(result.endpoints[0])
    assert "bond_change" not in result.endpoints[0].info
    proposal = rotate.propose(result.endpoints[0], np.random.default_rng(3))
    assert not proposal.valid  # The broken water is no longer a rotation target.


def test_pending_resume_matches_positions_tags_history_and_rng(tmp_path):
    expected_rng = np.random.default_rng(9)
    expected = run(tmp_path / "baseline", Worker(), rng=expected_rng, steps=3)
    assert run(tmp_path / "resume", Worker(pending=True), steps=3).status is EvaluationStatus.PENDING
    actual_rng = np.random.default_rng(100)
    actual = run(tmp_path / "resume", Worker(), rng=actual_rng, steps=3)
    np.testing.assert_array_equal(actual.endpoints[0].positions, expected.endpoints[0].positions)
    np.testing.assert_array_equal(actual.endpoints[0].get_tags(), expected.endpoints[0].get_tags())
    assert actual_rng.bit_generator.state == expected_rng.bit_generator.state
    records = []
    for directory in ("baseline", "resume"):
        records.append([row.data for row in ase.db.connect(tmp_path / directory / "history.db").select()])
    assert records[0] == records[1]


def test_resume_rejects_changed_or_removed_bond_policy(tmp_path):
    run(tmp_path, Worker(pending=True))
    with pytest.raises(ValueError, match="Incompatible bond_change"):
        run(tmp_path, Worker(), ops=operators(bond_cutoffs={"O-H": 1.3}))
    move = parse_operators([dict(method="move", particles=["H2O"], skip_distance_check=True)])[0]
    with pytest.raises(ValueError, match="Incompatible bond_change"):
        run(tmp_path, Worker(), ops=move)


@pytest.mark.parametrize("boundary", ["submission", "commit"])
def test_interruption_with_split_and_merged_tags_replays_exactly(tmp_path, monkeypatch, boundary):
    from gdpx.exploration.basin_hopping import chain
    baseline_rng = np.random.default_rng(9)
    expected = run(tmp_path / "baseline", Worker(), rng=baseline_rng, steps=3)
    worker = Worker()
    submit, commit = worker.run, chain._commit
    calls = []
    def fail_submission(frames):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("interrupted submission")
        return submit(frames)
    def fail_commit(directory, step, *args, **kwargs):
        if step == 2:
            raise RuntimeError("interrupted commit")
        return commit(directory, step, *args, **kwargs)
    with monkeypatch.context() as patch:
        if boundary == "submission":
            patch.setattr(worker, "run", fail_submission)
        else:
            patch.setattr(chain, "_commit", fail_commit)
        with pytest.raises(RuntimeError, match="interrupted"):
            run(tmp_path / "resume", worker, steps=3)
    rng = np.random.default_rng(100)
    actual = run(tmp_path / "resume", Worker(), rng=rng, steps=3)
    np.testing.assert_array_equal(actual.endpoints[0].positions, expected.endpoints[0].positions)
    np.testing.assert_array_equal(actual.endpoints[0].get_tags(), expected.endpoints[0].get_tags())
    assert rng.bit_generator.state == baseline_rng.bit_generator.state
