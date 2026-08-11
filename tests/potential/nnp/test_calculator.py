import tempfile
import pathlib
import json
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from gdpx.potential.nnp.calculator import ACSFNN
from gdpx.trainer.nnp_trainer import NnpTrainer


def _train_minimal_model(extra_params=None):
    np.random.seed(42)
    ds = []
    for _ in range(4):
        n = 2
        pos = np.random.randn(n, 3) * 2.0
        has_au = extra_params and "Au" in extra_params.get("elements", [])
        species = "CuAu" if has_au else "Cu" * n
        a = Atoms(species, positions=pos, pbc=False)
        ref = 1.0 / max(np.linalg.norm(pos[0] - pos[1]), 0.5)
        a.calc = SinglePointCalculator(a, energy=ref)
        ds.append(a)

    params = dict(
        elements=["Cu"],
        g2_params=[(0.1, 0.0)],
        g4_params=[],
        r_cut=6.0,
        hidden_sizes=[16, 16],
    )
    if extra_params:
        params.update(extra_params)

    tmp = tempfile.mkdtemp()
    t = NnpTrainer(
        config=dict(
            n_epochs=5,
            learning_rate=dict(start=0.01, stop=0.01),
            loss=dict(
                start_pref_e=1.0,
                limit_pref_e=1.0,
                start_pref_f=0.0,
                limit_pref_f=0.0,
            ),
            verbose=0,
        ),
        directory=tmp,
        calculator_params=params,
    )
    t.train(ds)
    return ACSFNN(model_file=pathlib.Path(tmp) / "nn_weights.npz"), tmp


class TestCalculator:
    def test_load_and_run(self):
        calc, tmp = _train_minimal_model()
        atoms = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]], calculator=calc)
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        assert isinstance(e, float)
        assert f.shape == (2, 3)

    def test_type_map_subset(self):
        calc, tmp = _train_minimal_model(extra_params=dict(
            elements=["Cu", "Au"],
            g2_params=[(0.1, 0.0)],
        ))
        calc2 = ACSFNN(model_file=str(pathlib.Path(tmp) / "nn_weights.npz"), type_map=["Cu"])
        atoms = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]], calculator=calc2)
        atoms.get_potential_energy()
        atoms.get_forces()

    def test_type_map_validation(self):
        calc, tmp = _train_minimal_model()
        import traceback
        try:
            ACSFNN(model_file=str(pathlib.Path(tmp) / "nn_weights.npz"), type_map=["Au"])
            assert False, "should have raised"
        except ValueError:
            pass

    def test_unknown_atom(self):
        calc, tmp = _train_minimal_model()
        atoms = Atoms("Au2", positions=[[0, 0, 0], [2.5, 0, 0]], calculator=calc)
        try:
            atoms.get_potential_energy()
            assert False, "should have raised"
        except ValueError:
            pass


class TestModelFile:
    def test_metadata_in_file(self):
        calc, tmp = _train_minimal_model()
        model_path = pathlib.Path(tmp) / "nn_weights.npz"
        loaded = np.load(model_path)
        for key in ["format_version", "elements", "g2_eta", "g2_Rs", "g4_eta", "g4_zeta",
                     "g4_lambda_", "r_cut", "hidden_sizes", "feature_mean",
                     "feature_scale", "atomic_offsets", "W_0_0"]:
            assert key in loaded, f"Missing key: {key}"
        assert int(loaded["format_version"]) == 2

    def test_legacy_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "legacy.npz"
            np.savez(path, elements=np.array(["Cu"]))
            try:
                ACSFNN(path)
                assert False, "legacy models must be rejected"
            except ValueError as error:
                assert "Retrain" in str(error)

    def test_g2_only_model(self):
        calc, tmp = _train_minimal_model(extra_params=dict(
            g4_params=[],
        ))
        atoms = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]], calculator=calc)
        atoms.get_potential_energy()

    def test_g4_model(self):
        calc, tmp = _train_minimal_model(extra_params=dict(
            g4_params=[(0.01, 1.0, 1.0)],
            hidden_sizes=[32, 16],
        ))
        atoms = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]], calculator=calc)
        atoms.get_potential_energy()

    def test_linear_nn(self):
        calc, tmp = _train_minimal_model(extra_params=dict(
            hidden_sizes=[],
        ))
        atoms = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]], calculator=calc)
        atoms.get_potential_energy()

    def test_validation_and_history_artifacts(self):
        np.random.seed(12)
        dataset = []
        for _ in range(6):
            positions = np.random.randn(2, 3)
            atoms = Atoms("Cu2", positions=positions, pbc=False)
            atoms.calc = SinglePointCalculator(
                atoms, energy=float(np.sum(positions**2))
            )
            dataset.append(atoms)
        with tempfile.TemporaryDirectory() as tmp:
            trainer = NnpTrainer(
                config=dict(
                    n_epochs=3,
                    learning_rate=dict(start=0.003, stop=0.001),
                    loss=dict(
                        start_pref_e=1.0,
                        limit_pref_e=1.0,
                        start_pref_f=0.0,
                        limit_pref_f=0.0,
                    ),
                    validation_fraction=1.0 / 3.0,
                    early_stopping_patience=2,
                    batch_size=2,
                    verbose=0,
                ),
                calculator_params=dict(
                    elements=["Cu"],
                    g2_params=[(0.1, 0.0)],
                    g4_params=[],
                    r_cut=6.0,
                    hidden_sizes=[8],
                ),
                directory=tmp,
                random_seed=10,
            )
            trainer.train(dataset)
            config = json.loads((pathlib.Path(tmp) / "train_config.json").read_text())
            history = json.loads((pathlib.Path(tmp) / "training_history.json").read_text())
            assert config["dataset"]["n_training"] == 4
            assert config["dataset"]["n_validation"] == 2
            assert 1 <= len(history) <= 3
            assert history[0]["validation_energy_rmse"] is not None
            for key in [
                "energy_prefactor",
                "force_prefactor",
                "effective_energy_gradient_norm",
                "effective_force_gradient_norm",
                "gradient_cosine",
                "train_energy_loss",
                "train_force_loss",
                "selection_score",
            ]:
                assert key in history[0]
            assert np.isclose(
                history[0]["effective_energy_gradient_norm"],
                history[0]["energy_prefactor"]
                * history[0]["energy_gradient_norm"],
            )

    def test_legacy_loss_options_are_rejected(self):
        atoms = Atoms("Cu2", positions=[[0, 0, 0], [2.0, 0, 0]], pbc=False)
        atoms.calc = SinglePointCalculator(atoms, energy=0.5)
        with tempfile.TemporaryDirectory() as tmp:
            trainer = NnpTrainer(
                config=dict(n_epochs=1, learning_rate=0.003, force_weight=0.0),
                calculator_params=dict(
                    elements=["Cu"],
                    g2_params=[(0.1, 0.0)],
                    g4_params=[],
                    r_cut=6.0,
                    hidden_sizes=[4],
                ),
                directory=tmp,
            )
            try:
                trainer.train([atoms])
                assert False, "legacy loss options must be rejected"
            except ValueError as error:
                assert "Legacy NNP loss options" in str(error)
