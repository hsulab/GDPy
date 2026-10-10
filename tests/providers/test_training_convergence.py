"""Convergence checks must be safe before the first training invocation."""
import pytest

from gdpx.providers.deepmd.training.deepmd_jax import DeepmdJaxTrainer
from gdpx.providers.gp.trainer import GaussianProcessTrainer
from gdpx.providers.mace.trainer import MaceTrainer
from gdpx.providers.nequip.trainer import NequipTrainer
from gdpx.providers.nnp.trainer import NnpTrainer, COMPLETION_NAME


@pytest.mark.parametrize("trainer_type", [
    DeepmdJaxTrainer, GaussianProcessTrainer, MaceTrainer, NequipTrainer, NnpTrainer,
])
def test_fresh_training_is_not_complete(tmp_path, trainer_type):
    trainer = object.__new__(trainer_type)
    trainer.directory = tmp_path
    assert not trainer.read_convergence()


@pytest.mark.parametrize("trainer_type", [DeepmdJaxTrainer, GaussianProcessTrainer])
def test_exported_internal_training_is_complete(tmp_path, trainer_type):
    trainer = object.__new__(trainer_type)
    trainer.directory = tmp_path
    (tmp_path / trainer.frozen_name).touch()
    assert trainer.read_convergence()


def test_nnp_intermediate_weights_do_not_mark_training_complete(tmp_path):
    trainer = object.__new__(NnpTrainer)
    trainer.directory = tmp_path
    weights = tmp_path / trainer.frozen_name
    weights.touch()
    assert not trainer.read_convergence()
    (tmp_path / COMPLETION_NAME).touch()
    assert trainer.read_convergence()
    weights.unlink()
    assert not trainer.read_convergence()


def test_nnp_marks_completion_only_after_training_returns(tmp_path):
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    frames = []
    for distance in (2.0, 2.5):
        atoms = Atoms("Cu2", positions=[[0, 0, 0], [distance, 0, 0]])
        atoms.calc = SinglePointCalculator(atoms, energy=1.0 / distance)
        frames.append(atoms)
    trainer = NnpTrainer(
        config={"n_epochs": 1, "verbose": 0, "loss": {
            "start_pref_e": 1.0, "limit_pref_e": 1.0,
            "start_pref_f": 0.0, "limit_pref_f": 0.0,
        }}, directory=tmp_path, random_seed=42,
        calculator_params={"elements": ["Cu"], "g2_params": [(0.1, 0.0)],
                           "g4_params": [], "r_cut": 6.0, "hidden_sizes": [4]},
    )
    assert not trainer.read_convergence()
    trainer.train(frames)
    assert trainer.read_convergence()
    # A failed new attempt must not leave the old completion marker in place.
    with pytest.raises(ValueError, match="non-empty dataset"):
        trainer.train([])
    assert not trainer.read_convergence()


@pytest.mark.parametrize("trainer_type", [MaceTrainer, NequipTrainer])
def test_log_training_checks_empty_and_completed_logs(tmp_path, trainer_type):
    trainer = object.__new__(trainer_type)
    trainer.directory = tmp_path
    if trainer_type is MaceTrainer:
        log = tmp_path / "logs/model.log"
        final_line = "Done\n"
    else:
        log = tmp_path / trainer.RUN_NAME / "log"
        final_line = "Cumulative wall time: 10 s\n"
    log.parent.mkdir(parents=True)
    log.touch()
    assert not trainer.read_convergence()
    log.write_text("Still training\n")
    assert not trainer.read_convergence()
    log.write_text(final_line)
    assert trainer.read_convergence()
