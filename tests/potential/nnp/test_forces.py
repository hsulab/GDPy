import tempfile
import numpy as np
from ase import Atoms

from gdpx.potential.nnp.calculator import ACSFNN
from gdpx.trainer.nnp_trainer import NnpTrainer


def _make_model(g2_params, g4_params, hidden_sizes=(16, 16)):
    np.random.seed(42)
    ds = []
    for _ in range(4):
        n = 2 + np.random.randint(0, 2)
        pos = np.random.randn(n, 3) * 2.0
        a = Atoms("Cu" * n, positions=pos, pbc=False)
        ref = sum(
            1.0 / max(np.linalg.norm(pos[i] - pos[j]), 0.5)
            for i in range(n) for j in range(i + 1, n)
        )
        a.info["energy"] = ref
        ds.append(a)

    with tempfile.TemporaryDirectory() as tmp:
        t = NnpTrainer(
            config=dict(n_epochs=5, learning_rate=0.01, verbose=0), directory=tmp
        )
        t.train(
            ds,
            calculator_params=dict(
                elements=["Cu"],
                g2_params=g2_params,
                g4_params=g4_params,
                r_cut=6.0,
                hidden_sizes=hidden_sizes,
            ),
        )
        from pathlib import Path
        return ACSFNN(model_file=Path(tmp) / "nn_weights.npz")


def fd_force(atoms, i, d, h=1e-6):
    pos = atoms.positions.copy()
    atoms.positions[i, d] = pos[i, d] + h
    ep = atoms.get_potential_energy()
    atoms.positions[i, d] = pos[i, d] - h
    em = atoms.get_potential_energy()
    atoms.positions[i, d] = pos[i, d]
    return -(ep - em) / (2.0 * h)


def check_forces(atoms, atol=1e-6):
    f = atoms.get_forces()
    assert np.allclose(np.sum(f, axis=0), 0.0, atol=1e-12), \
        f"Forces not conserved: sum={np.sum(f, axis=0)}"
    max_diff = 0.0
    for i in range(len(atoms)):
        for d in range(3):
            fd = fd_force(atoms, i, d)
            diff = abs(fd - f[i, d])
            max_diff = max(max_diff, diff)
            assert diff < atol, \
                f"Force mismatch atom {i} dir {d}: analytical={f[i,d]:.8f} FD={fd:.8f} diff={diff:.2e}"
    return max_diff


class TestG2Forces:
    def test_dimer(self):
        calc = _make_model([(0.1, 0.0)], [])
        atoms = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]], calculator=calc)
        check_forces(atoms)

    def test_trimer(self):
        calc = _make_model([(0.1, 0.0)], [])
        atoms = Atoms(
            "Cu3", positions=[[0, 0, 0], [2.5, 0, 0], [1.0, 2.0, 0.0]], calculator=calc
        )
        check_forces(atoms)

    def test_four_atoms_nonplanar(self):
        calc = _make_model([(0.1, 0.0), (0.2, 0.0)], [])
        atoms = Atoms(
            "Cu4",
            positions=[[0, 0, 0], [2.5, 0, 0], [1.0, 2.0, 0.0], [0.5, 0.5, 2.0]],
            calculator=calc,
        )
        check_forces(atoms)

    def test_single_atom(self):
        calc = _make_model([(0.1, 0.0)], [])
        atoms = Atoms("Cu", positions=[[0, 0, 0]], calculator=calc)
        f = atoms.get_forces()
        assert np.all(f == 0.0)

    def test_multi_g2(self):
        calc = _make_model([(0.05, 0.0), (0.1, 0.0), (0.2, 0.5)], [])
        atoms = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]], calculator=calc)
        check_forces(atoms)

    def test_wide_separation(self):
        calc = _make_model([(0.1, 0.0)], [])
        atoms = Atoms("Cu2", positions=[[0, 0, 0], [10.0, 0, 0]], calculator=calc)
        f = atoms.get_forces()
        assert np.allclose(f, 0.0, atol=1e-12)


class TestG4Forces:
    def test_dimer_g4(self):
        calc = _make_model([(0.1, 0.0)], [(0.01, 1.0, 1.0)])
        atoms = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]], calculator=calc)
        check_forces(atoms)

    def test_trimer_g4(self):
        calc = _make_model(
            [(0.1, 0.0)], [(0.01, 1.0, 1.0), (0.02, 2.0, -1.0)]
        )
        atoms = Atoms(
            "Cu3", positions=[[0, 0, 0], [2.5, 0, 0], [1.0, 2.0, 0.0]], calculator=calc
        )
        check_forces(atoms)

    def test_four_atoms_g4(self):
        calc = _make_model([(0.1, 0.0)], [(0.01, 1.0, 1.0)])
        atoms = Atoms(
            "Cu4",
            positions=[[0, 0, 0], [2.5, 0, 0], [1.0, 2.0, 0.0], [0.5, 0.5, 2.0]],
            calculator=calc,
        )
        check_forces(atoms)


class TestRandomStructures:
    def test_many_sizes(self):
        calc = _make_model(
            [(0.1, 0.0), (0.2, 0.5)], [(0.01, 1.0, 1.0), (0.02, 2.0, -1.0)]
        )
        np.random.seed(123)
        for n in [2, 3, 4, 5]:
            pos = np.random.randn(n, 3) * 3.0
            atoms = Atoms("Cu" * n, positions=pos, calculator=calc)
            max_diff = 0.0
            h = 1e-6
            for i in range(n):
                for d in range(3):
                    p = atoms.positions.copy()
                    atoms.positions[i, d] = p[i, d] + h; ep = atoms.get_potential_energy()
                    atoms.positions[i, d] = p[i, d] - h; em = atoms.get_potential_energy()
                    atoms.positions[i, d] = p[i, d]
                    fd = -(ep - em) / (2.0 * h)
                    max_diff = max(max_diff, abs(fd - atoms.get_forces()[i, d]))
            assert max_diff < 1e-6, f"n={n}: max FD diff = {max_diff:.2e}"
