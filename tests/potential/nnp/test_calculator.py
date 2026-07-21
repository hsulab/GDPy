import tempfile
import pathlib
import numpy as np
from ase import Atoms
from gdpx.potential.nnp.calculator import ACSFNN
from gdpx.trainer.nnp_trainer import NnpTrainer


def _train_minimal_model(extra_params=None):
    np.random.seed(42)
    ds = []
    for _ in range(4):
        n = 2
        pos = np.random.randn(n, 3) * 2.0
        a = Atoms("Cu" * n, positions=pos, pbc=False)
        ref = 1.0 / max(np.linalg.norm(pos[0] - pos[1]), 0.5)
        a.info["energy"] = ref
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
    t = NnpTrainer(config=dict(n_epochs=5, learning_rate=0.01, verbose=0), directory=tmp)
    t.train(ds, calculator_params=params)
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
        for key in ["elements", "g2_eta", "g2_Rs", "g4_eta", "g4_zeta",
                     "g4_lambda_", "r_cut", "hidden_sizes"]:
            assert key in loaded, f"Missing key: {key}"

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
