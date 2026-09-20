"""State ownership, allocation, and evaluation regressions for reusable moves."""

import copy

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms

from gdpx.sampling import MoveProposal, parse_operators
from gdpx.sampling.acceptance import AcceptanceRule, ExchangeAcceptance, ReactionAcceptance, SemiGrandAcceptance
from gdpx.structures.geometry.spatial import get_bond_distance_dict
from gdpx.exploration.move_step import run_worker_move
from gdpx.exploration.basin_hopping.engine import run_hopping_steps
from gdpx.exploration.monte_carlo.monte_carlo import MonteCarlo, MCStepState
from gdpx.exploration.monte_carlo.hybrid_monte_carlo import HybridMonteCarlo
from gdpx.execution.workers.single import SingleWorker


REGION = {"method": "lattice", "origin": [0, 0, 0], "cell": [20, 0, 0, 0, 20, 0, 0, 0, 20]}


def structure():
    atoms = Atoms("HHeH", positions=[[5, 5, 5], [6, 5, 5], [7, 5, 5]], cell=[20] * 3)
    atoms.set_tags([1, 2, 3])
    atoms.new_array("custom", np.arange(6).reshape(3, 2))
    atoms.calc = SinglePointCalculator(atoms, energy=0.0)
    return atoms


def operator(method="move", **kwargs):
    params = dict(method=method, particles=["H"], region=REGION, skip_distance_check=True)
    params.update(kwargs)
    op = parse_operators([params])[0][0]
    op._print = op._debug = lambda *args: None
    op.bond_distance_dict = get_bond_distance_dict([1, 2], ratio=1.0)
    op.blmin = get_bond_distance_dict([1, 2], ratio=0.8)
    op.custom_pair_distance_dict = None
    return op


def assert_arrays(atoms, original):
    assert atoms.arrays.keys() == original.keys()
    for name, values in original.items():
        np.testing.assert_array_equal(atoms.arrays[name], values)


@pytest.mark.parametrize("method,params", [
    ("move", {}), ("swap", {"particles": ["H", "He"]}),
    ("swap_type", {"particles": ["H", "He"], "chempots": [0.0, 0.0]}),
    ("bounce", {"direction": "+x"}),
    ("exchange", {"chempots": [0.0]}),
    ("biased_volume_exchange", {"chempots": [0.0]}),
    ("cavity_exchange", {"chempots": [0.0], "num_trials": 3}),
])
def test_moves_borrow_and_restore_exactly(method, params):
    op, atoms = operator(method, **params), structure()
    original = {name: a.copy() for name, a in atoms.arrays.items()}
    calc = atoms.calc
    for seed in range(6):
        proposal = op.propose(atoms, np.random.default_rng(seed))
        if proposal.valid:
            assert proposal.atoms is atoms
            proposal.rollback()
        assert proposal.closed and proposal.atoms is None
        assert_arrays(atoms, original)
        assert atoms.calc is calc
        assert atoms.get_potential_energy() == 0.0
        assert op._atoms is None and op._state == {}


def test_topology_journal_restores_custom_arrays_constraints_and_order():
    atoms = structure()
    atoms.set_constraint(FixAtoms(indices=[1]))
    original = {name: a.copy() for name, a in atoms.arrays.items()}
    with MoveProposal(atoms) as trial:
        trial.delete([0, 2])
        particle = Atoms("Li", positions=[[1, 1, 1]])
        particle.set_momenta([[1, 2, 3]])
        trial.append(particle)
        trial.set_info("step", 7)
        assert len(atoms) == 2
    assert_arrays(atoms, original)
    np.testing.assert_array_equal(atoms.constraints[0].get_indices(), [1])
    assert "step" not in atoms.info


def test_transaction_exclusion_commit_and_exception_cleanup():
    atoms = structure()
    proposal = MoveProposal(atoms)
    with pytest.raises(RuntimeError, match="active move"):
        MoveProposal(atoms)
    proposal.watch([0])
    atoms.positions[0, 0] = 9.0
    assert proposal.commit() is atoms
    assert atoms.positions[0, 0] == 9.0
    with pytest.raises(RuntimeError, match="closed"):
        proposal.rollback()
    with pytest.raises(ValueError, match="evaluation"):
        with MoveProposal(atoms) as trial:
            trial.watch([0])
            atoms.positions[0, 0] = 2.0
            raise ValueError("evaluation")
    assert atoms.positions[0, 0] == 9.0


def test_rollback_does_not_reattach_or_copy_calculator_state():
    class Calculator:
        calls = 0

        def set_atoms(self, atoms):
            self.calls += 1
            assert self.calls == 1, "rollback reattached the calculator"

    atoms = structure()
    calculator = Calculator()
    atoms.calc = calculator
    with operator().propose(atoms, np.random.default_rng(0)):
        assert atoms.calc is None
    assert atoms.calc is calculator
    assert calculator.calls == 1


def test_failed_proposal_and_exception_after_edit_restore(monkeypatch):
    from gdpx.sampling.moves import MoveOperator
    atoms, op = structure(), operator(max_random_attempts=0)
    original = atoms.positions.copy()
    assert not op.propose(atoms, np.random.default_rng(0)).valid
    np.testing.assert_array_equal(atoms.positions, original)

    def fail(self, atoms, rng):
        self._transaction.watch([0])
        atoms.positions[0] += 1
        raise RuntimeError("after edit")

    monkeypatch.setattr(MoveOperator, "_propose", fail)
    with pytest.raises(RuntimeError, match="after edit"):
        op.propose(atoms, np.random.default_rng(0))
    np.testing.assert_array_equal(atoms.positions, original)


@pytest.mark.parametrize("method,params", [("move", {}), ("swap", {"particles": ["H", "He"]})])
def test_local_move_does_not_copy_system_and_undo_is_constant_size(monkeypatch, method, params):
    original_copy = Atoms.copy
    system_ids = set()

    def guarded_copy(atoms):
        assert id(atoms) not in system_ids, "whole-system copy in local proposal"
        return original_copy(atoms)

    monkeypatch.setattr(Atoms, "copy", guarded_copy)
    sizes = []
    op = operator(method, **params)
    for count in [10, 1000]:
        atoms = Atoms("H" * (count - 1) + "He", positions=np.full((count, 3), 5.0), cell=[20] * 3)
        atoms.set_tags(np.arange(count))
        system_ids.add(id(atoms))
        # Reject deepcopy as well; particle-fragment copies remain permitted.
        atoms.__deepcopy__ = lambda memo: pytest.fail("whole-system deepcopy")
        with op.propose(atoms, np.random.default_rng(0)) as trial:
            assert trial.atoms is atoms
            sizes.append(trial.undo_nbytes)
        system_ids.remove(id(atoms))
    assert sizes[0] == sizes[1]
    assert 0 < sizes[0] <= 64


def test_acceptance_rules_preserve_formulas_and_do_not_mutate_metadata():
    temperature = 1 / units.kB
    assert AcceptanceRule(temperature).probability({}, 0, 1) == pytest.approx(np.exp(-1))
    rule = ExchangeAcceptance(temperature, 0.3, 2.0)
    metadata = dict(operation="insert", num_particles=2, volume=4.0, symm_factor=2.0)
    original = copy.deepcopy(metadata)
    assert rule.probability(metadata, 0, 1) == pytest.approx(4 / 3 / 2 / 2 * np.exp(-0.7))
    assert metadata == original
    metadata["operation"] = "remove"
    assert rule.probability(metadata, 0, 1) == pytest.approx(2 * 2 * 2 / 4 * np.exp(-1.3))
    rule = SemiGrandAcceptance(temperature, ("H", "He"), (0.1, 0.4))
    assert rule.probability(dict(first_ptype="H", second_ptype="He"), 0, 1) == pytest.approx(np.exp(-0.7))
    rule = ReactionAcceptance(temperature, (-1, 1), (0.0, 0.0))
    assert rule.probability({"direction": 1, "particle_numbers": [1, 4], "volume": 2}, 0, 1) == pytest.approx(0.2 * np.exp(-1))


class Worker:
    def __init__(self, energy, waiting=False):
        self.energy, self.waiting, self.calls = energy, waiting, 0

    def run(self, frames, **kwargs):
        self.calls += 1
        self.result = frames[0].copy()  # The execution ownership boundary.
        self.result.positions += 0.25  # Mimic relaxation beyond the moved fragment.
        self.result.calc = SinglePointCalculator(self.result, energy=self.energy)

    def inspect(self, **kwargs):
        pass

    def get_number_of_running_jobs(self):
        return int(self.waiting)

    def retrieve(self):
        return [[self.result]]


@pytest.mark.parametrize("energy,accepted", [(-1.0, True), (100.0, False)])
def test_worker_result_has_separate_ownership(tmp_path, energy, accepted):
    atoms, op = structure(), operator()
    original = atoms.positions.copy()
    result = run_worker_move(atoms, 0, [op], [1.0], np.random.default_rng(2), Worker(energy), tmp_path / "pending")
    assert result.accepted is accepted
    np.testing.assert_array_equal(atoms.positions, original)
    assert (result.atoms is atoms) is (not accepted)
    assert atoms.get_potential_energy() == 0


def test_worker_exception_restores_trial(tmp_path):
    class FailingWorker(Worker):
        def run(self, frames, **kwargs):
            raise RuntimeError("worker failed")
    atoms = structure()
    original = atoms.positions.copy()
    with pytest.raises(RuntimeError, match="worker failed"):
        run_worker_move(atoms, 0, [operator()], [1.0], np.random.default_rng(2), FailingWorker(0), tmp_path / "pending")
    np.testing.assert_array_equal(atoms.positions, original)
    assert atoms.get_potential_energy() == 0


def test_pending_worker_resumes_without_redrawing(tmp_path, monkeypatch):
    atoms, op, rng = structure(), operator(), np.random.default_rng(2)
    worker = Worker(-1, waiting=True)
    path = tmp_path / "pending"
    waiting = run_worker_move(atoms, 0, [op], [1.0], rng, worker, path)
    assert waiting.accepted is None
    expected_rng = copy.deepcopy(rng)
    expected_rng.uniform()  # Only acceptance remains.
    monkeypatch.setattr(op, "propose", lambda *args: pytest.fail("redrew pending move"))
    worker.waiting = False
    restarted_rng = np.random.default_rng(999)
    result = run_worker_move(structure(), 0, [op], [1.0], restarted_rng, worker, path)
    assert result.accepted and worker.calls == 1
    assert restarted_rng.bit_generator.state == expected_rng.bit_generator.state
    assert not path.exists()


def test_bh_rejection_does_not_undo_relaxation_on_the_wrong_object(tmp_path):
    class Driver:
        directory = tmp_path / "driver"

        def run(self, atoms, **kwargs):
            self.result = atoms.copy()
            self.result.positions += 3
            self.result.calc = SinglePointCalculator(self.result, energy=100)

        def read_trajectory(self):
            return [self.result]

    atoms = structure()
    original = atoms.positions.copy()
    result, states = run_hopping_steps(atoms, 0, Driver(), [operator()], [1.0], 3, np.random.default_rng(2))
    assert states == [1, 1, 1]
    assert result is atoms
    np.testing.assert_array_equal(atoms.positions, original)
    assert atoms.get_potential_energy() == 0


def test_bh_evaluation_exception_restores_structure_and_metadata(tmp_path):
    class Driver:
        directory = tmp_path / "failed-driver"

        def run(self, atoms, **kwargs):
            raise RuntimeError("failed evaluation")

    atoms = structure()
    atoms.info["mcstep"] = 99
    original = atoms.positions.copy()
    with pytest.raises(RuntimeError, match="failed evaluation"):
        run_hopping_steps(atoms, 0, Driver(), [operator()], [1.0], 1, np.random.default_rng(0))
    np.testing.assert_array_equal(atoms.positions, original)
    assert atoms.info["mcstep"] == 99
    assert atoms.get_potential_energy() == 0


def test_reaction_undo_and_configuration_roundtrip():
    params = dict(method="react", reaction=dict(particles=["H", "H2"], chempot_0=[0, 0], coefficients=[-2, 1]),
                  region=REGION, temperature=300, use_bias=False, skip_distance_check=True)
    op = parse_operators([params])[0][0]
    assert params["method"] == "react"
    assert parse_operators([op.as_dict()])[0][0].as_dict() == op.as_dict()
    for atoms in [structure(), Atoms("H2", positions=[[5, 5, 5], [5, 5, 6]], tags=[1, 1], cell=[20] * 3)]:
        original = {name: a.copy() for name, a in atoms.arrays.items()}
        with op.propose(atoms, np.random.default_rng(3)) as trial:
            assert trial.valid
        assert_arrays(atoms, original)


def test_adsorbate_helper_exception_rolls_back_appended_atoms(monkeypatch):
    from gdpx.sampling.moves.exchange import adsorb

    def fail(atoms, particle, **kwargs):
        atoms.extend(particle)
        raise RuntimeError("adsorption failed")

    monkeypatch.setattr(adsorb, "insert_one_particle_on_site", fail)
    atoms = structure()
    atoms.numbers[:] = 2  # No H: force an insertion.
    original = {name: a.copy() for name, a in atoms.arrays.items()}
    op = operator("adsorbate_exchange", chempots=[0], anchors={"group": "symbol He"})
    assert parse_operators([op.as_dict()])[0][0].as_dict() == op.as_dict()
    with pytest.raises(RuntimeError, match="adsorption failed"):
        op.propose(atoms, np.random.default_rng(0))
    assert_arrays(atoms, original)


class SingleTestWorker(Worker, SingleWorker):
    def rewind_to_step(self, step):
        self.rewound = step


def mc_engine(directory, cls=MonteCarlo):
    from types import SimpleNamespace
    from ase.io import write
    engine = object.__new__(cls)
    engine.directory = directory
    directory.mkdir(parents=True, exist_ok=True)
    engine.rng = np.random.default_rng(9)
    engine.random_seed = 9
    engine._print = engine._debug = lambda *args: None
    engine.atoms = structure()
    engine.builder = SimpleNamespace()
    engine.operators = [operator()]
    engine.op_probs = [1.0]
    engine.energy_stored = 0.0
    engine.ckpt_period = 1
    engine.should_retry = False
    engine.convergence = {"steps": 4}
    engine.worker = SingleTestWorker(-1)
    write(directory / "mc.xyz", engine.atoms)
    write(directory / "mc_attempts.xyz", engine.atoms)
    (directory / "opstat.txt").write_text("#Step\n")
    return engine


def test_mc_restart_matches_uninterrupted_run(tmp_path):
    engine = mc_engine(tmp_path / "mc")
    assert engine._irun(1) == MCStepState.FINISHED
    engine._save_checkpoint(1)
    assert engine._irun(2) == MCStepState.FINISHED
    expected = {name: a.copy() for name, a in engine.atoms.arrays.items()}
    rng_state = copy.deepcopy(engine.rng.bit_generator.state)
    engine._load_checkpoint()
    assert engine.start_step == 1 and engine.worker.rewound == 1
    assert engine._irun(2) == MCStepState.FINISHED
    assert_arrays(engine.atoms, expected)
    assert engine.rng.bit_generator.state == rng_state


def test_mc_legacy_checkpoint_rejected_before_unpickling(tmp_path):
    engine = mc_engine(tmp_path / "legacy")
    checkpoint = engine.directory / "checkpoint.1"
    checkpoint.mkdir()
    (checkpoint / "op-0.ckpt").write_bytes(b"invalid legacy pickle")
    with pytest.raises(ValueError, match="Legacy MC operator checkpoint"):
        engine._load_checkpoint()


def test_hybrid_pending_move_resumes_at_its_original_index(tmp_path, monkeypatch):
    engine = mc_engine(tmp_path / "hybrid", HybridMonteCarlo)
    engine.num_mcmoves = 2
    engine.worker.waiting = True
    original = engine.atoms.positions.copy()
    assert engine._irun_metropolis(1, "mc", engine.worker) == MCStepState.UNFINISHED
    np.testing.assert_array_equal(engine.atoms.positions, original)
    from gdpx.exploration.move_step import read_pending
    engine.atoms, pending = read_pending(engine.directory / "pending-hybrid", engine.rng)
    engine._resume_context = pending["context"]
    engine.worker.waiting = False
    assert engine._irun_metropolis(1, "mc", engine.worker) == MCStepState.FINISHED
    assert engine.worker.calls == 2
    assert engine._resume_context is None


def test_all_move_settings_roundtrip_without_mutating_input():
    configurations = [
        dict(method="move", max_disp=0.2),
        dict(method="swap", particles=["H", "He"], swap_mode="cop_z", check_used_pairs=True),
        dict(method="bounce", direction="+z", bias_ratio=0.5, repulsion_strength=0.2),
        dict(method="exchange", chempots=[-1.0]),
        dict(method="biased_volume_exchange", chempots=[-1.0]),
        dict(method="cavity_exchange", chempots=[-1.0], num_trials=4, cavity_distance=[0.8, None]),
        dict(method="swap_type", particles=["H", "He"], chempots=[0.1, 0.2]),
    ]
    for config in configurations:
        params = dict(particles=["H"], region=REGION, max_random_attempts=13)
        params.update(config)
        original = copy.deepcopy(params)
        op = parse_operators([params])[0][0]
        assert params == original
        assert parse_operators([op.as_dict()])[0][0].as_dict() == op.as_dict()


def test_hopping_with_actual_emt_driver(tmp_path):
    from ase.calculators.emt import EMT
    from gdpx.execution.factory import create_worker

    worker = create_worker({
        "potential": {"provider": "emt", "parameters": {}},
        "executor": {"provider": "ase", "method": "spc", "parameters": {}},
        "options": {"worker": "single"},
    })
    driver = worker.driver
    driver.directory = tmp_path / "evaluation"
    atoms = Atoms("Cu2", positions=[[5, 5, 5], [7.4, 5, 5]], tags=[1, 2], cell=[20] * 3)
    atoms.calc = EMT()
    atoms.get_potential_energy()
    op = operator(particles=["Cu"], max_disp=0.05)
    result, states = run_hopping_steps(atoms, 0, driver, [op], [1.0], 2, np.random.default_rng(3))
    assert len(states) == 2 and set(states) <= {0, 1}
    assert np.isfinite(result.get_potential_energy())
    np.testing.assert_array_equal(result.get_tags(), [1, 2])


def test_promoted_bh_runs_a_population_generation_with_emt(tmp_path):
    from ase.io import write
    from gdpx.exploration.factory import create_expedition

    atoms = Atoms("Cu2", positions=[[5, 5, 5], [7.4, 5, 5]], tags=[1, 2], cell=[20] * 3)
    source = tmp_path / "seed.xyz"
    write(source, atoms)
    runtime = {
        "potential": {"provider": "emt", "parameters": {}},
        "executor": {"provider": "ase", "method": "spc", "parameters": {}},
        "options": {"worker": "single"},
    }
    engine = create_expedition({"method": "basin_hopping", "recipe": {
        "population": {"initial_size": 1, "generation_size": 1,
                       "random_offspring_generator": {"method": "read_stru", "fname": str(source)}},
        "operators": [{"method": "move", "particles": ["Cu"], "max_disp": 0.05,
                       "skip_distance_check": True}],
        "num_mcmoves": 2, "mcworker": runtime, "convergence": {"generation": 1},
        "random_seed": 7, "use_archive": False,
    }})
    engine.directory = tmp_path / "bh"
    engine.register_worker(runtime)
    engine.run()
    assert engine.read_convergence()
    assert (engine.directory / "results" / "all_candidates.xyz").exists()
    serialized = engine.as_dict()
    assert serialized["method"] == "basin_hopping"
    assert serialized["runtime"]["executor"]["method"] == "spc"
    assert len(engine.get_workers()) == 2
