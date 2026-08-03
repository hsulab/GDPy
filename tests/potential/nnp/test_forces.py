import pathlib
import tempfile
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

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
        a.calc = SinglePointCalculator(a, energy=ref)
        ds.append(a)

    with tempfile.TemporaryDirectory() as tmp:
        t = NnpTrainer(
            config=dict(n_epochs=5, learning_rate=0.01, verbose=0, force_weight=0.0),
            directory=tmp,
            calculator_params=dict(
                elements=["Cu"],
                g2_params=g2_params,
                g4_params=g4_params,
                r_cut=6.0,
                hidden_sizes=hidden_sizes,
            ),
        )
        t.train(ds)
        return ACSFNN(model_file=pathlib.Path(tmp) / "nn_weights.npz")


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


class TestForceGradient:
    def test_force_loss_gradient_matches_fd(self):
        from gdpx.potential.nnp.descriptor import (
            G2Param,
            G4Param,
            compute_forces,
            compute_force_gradient_weights,
            compute_n_features,
            compute_symmetry_functions,
        )
        from gdpx.potential.nnp.nn import SimpleNN

        np.random.seed(3)
        elements = ["Cu", "Au"]
        g2 = [G2Param(0.05, 0.0), G2Param(0.1, 0.0), G2Param(0.2, 0.5)]
        g4 = [G4Param(0.01, 1.0, 1.0), G4Param(0.02, 2.0, -1.0)]
        r_cut = 6.0
        nfeat = compute_n_features(elements, g2, g4)
        nn = SimpleNN(nfeat, hidden_sizes=(8, 8), seed=7)
        atoms = Atoms("Cu2Au", positions=[[0, 0, 0], [2.5, 0, 0], [1.0, 2.0, 1.5]], pbc=False)
        Fref = np.random.randn(3, 3) * 0.3
        fw = 10.0
        n_atoms = 3

        def forces_pred():
            G = compute_symmetry_functions(atoms, elements, g2, g4, r_cut)
            _, dEdG = nn.energy_and_gradient(G)
            return compute_forces(atoms, elements, g2, g4, r_cut, dEdG)

        def loss():
            return fw * np.mean((forces_pred() - Fref) ** 2)

        G = compute_symmetry_functions(atoms, elements, g2, g4, r_cut)
        _, dEdG = nn.energy_and_gradient(G)
        dF = compute_forces(atoms, elements, g2, g4, r_cut, dEdG) - Fref
        B = compute_force_gradient_weights(atoms, elements, g2, g4, r_cut, dF)
        analytic = nn.double_backward(B)
        scale = -2.0 * fw / (3.0 * n_atoms)
        analytic = {
            "weights": [w * scale for w in analytic["weights"]],
            "biases": [b * scale for b in analytic["biases"]],
        }

        eps = 1e-6
        max_err = 0.0
        for wi, W in enumerate(nn.weights):
            for p in np.ndindex(W.shape):
                old = W[p]
                W[p] = old + eps
                Lp = loss()
                W[p] = old - eps
                Lm = loss()
                W[p] = old
                fd = (Lp - Lm) / (2.0 * eps)
                max_err = max(max_err, abs(fd - analytic["weights"][wi][p]))
        for bi, Bp in enumerate(nn.biases):
            for p in np.ndindex(Bp.shape):
                old = Bp[p]
                Bp[p] = old + eps
                Lp = loss()
                Bp[p] = old - eps
                Lm = loss()
                Bp[p] = old
                fd = (Lp - Lm) / (2.0 * eps)
                max_err = max(max_err, abs(fd - analytic["biases"][bi][p]))
        assert max_err < 1e-4, f"force-gradient FD error = {max_err:.2e}"


class TestForceTraining:
    def test_force_training_reduces_force_loss(self):
        np.random.seed(1)
        ds = []
        for _ in range(3):
            pos = np.random.randn(3, 3) * 1.5
            ref = sum(
                1.0 / max(np.linalg.norm(pos[i] - pos[j]), 0.5)
                for i in range(3) for j in range(i + 1, 3)
            )
            forces = np.zeros((3, 3))
            for i in range(3):
                for j in range(i + 1, 3):
                    d = max(np.linalg.norm(pos[i] - pos[j]), 0.5)
                    rhat = (pos[i] - pos[j]) / np.linalg.norm(pos[i] - pos[j])
                    forces[i] += (1.0 / d**2) * rhat
                    forces[j] -= (1.0 / d**2) * rhat
            a = Atoms("Cu3", positions=pos, pbc=False)
            a.calc = SinglePointCalculator(a, energy=ref, forces=forces)
            ds.append(a)

        from gdpx.potential.nnp.calculator import ACSFNN
        from gdpx.potential.nnp.descriptor import compute_forces

        def eval_force_loss(model_path):
            calc = ACSFNN(model_file=model_path)
            fl = 0.0
            for a in ds:
                G = calc._compute_descriptor(a)
                _, dEdG = calc.nn.energy_and_gradient(G)
                Fp = compute_forces(
                    a, calc.model_elements, calc.g2_params, calc.g4_params, calc.r_cut, dEdG
                )
                fl += np.mean((Fp - a.calc.results["forces"]) ** 2)
            return fl / len(ds)

        with tempfile.TemporaryDirectory() as tmp:
            t = NnpTrainer(
                config=dict(
                    n_epochs=40, learning_rate=0.05, force_weight=10.0, verbose=0
                ),
                directory=tmp,
                calculator_params=dict(
                    elements=["Cu"],
                    g2_params=[(0.1, 0.0), (0.2, 0.0)],
                    g4_params=[(0.01, 1.0, 1.0)],
                    r_cut=6.0,
                    hidden_sizes=(16, 16),
                ),
            )
            t.train(ds)
            final = eval_force_loss(pathlib.Path(tmp) / "nn_weights.npz")
            assert final < 0.05, f"force loss not reduced: {final:.6f}"


def _pair_dataset(n_structures=3, offset=0.0, seed=11):
    np.random.seed(seed)
    ds = []
    for _ in range(n_structures):
        pos = np.random.randn(3, 3) * 1.5
        ref = offset + sum(
            1.0 / max(np.linalg.norm(pos[i] - pos[j]), 0.5)
            for i in range(3) for j in range(i + 1, 3)
        )
        forces = np.zeros((3, 3))
        for i in range(3):
            for j in range(i + 1, 3):
                d = max(np.linalg.norm(pos[i] - pos[j]), 0.5)
                rhat = (pos[i] - pos[j]) / np.linalg.norm(pos[i] - pos[j])
                forces[i] += (1.0 / d**2) * rhat
                forces[j] -= (1.0 / d**2) * rhat
        a = Atoms("Cu3", positions=pos, pbc=False)
        a.calc = SinglePointCalculator(a, energy=ref, forces=forces)
        ds.append(a)
    return ds


def _model_losses(model_path, ds):
    from gdpx.potential.nnp.calculator import ACSFNN
    from gdpx.potential.nnp.descriptor import compute_forces

    calc = ACSFNN(model_file=model_path)
    el = 0.0
    fl = 0.0
    for a in ds:
        G = calc._compute_descriptor(a)
        E = float(np.sum(calc.nn.forward(G))) + calc.energy_shift
        el += (E - a.calc.results["energy"]) ** 2 / len(a)
        _, dEdG = calc.nn.energy_and_gradient(G)
        Fp = compute_forces(
            a, calc.model_elements, calc.g2_params, calc.g4_params, calc.r_cut, dEdG
        )
        fl += np.mean((Fp - a.calc.results["forces"]) ** 2)
    return el / len(ds), fl / len(ds)


class TestCombinedTraining:
    def test_energy_shift_roundtrip(self):
        ds = _pair_dataset(offset=10.0, seed=5)
        with tempfile.TemporaryDirectory() as tmp:
            t = NnpTrainer(
                config=dict(n_epochs=40, learning_rate=0.003, force_weight=10.0, verbose=0),
                directory=tmp,
                calculator_params=dict(
                    elements=["Cu"],
                    g2_params=[(0.1, 0.0), (0.2, 0.0)],
                    g4_params=[(0.01, 1.0, 1.0)],
                    r_cut=6.0,
                    hidden_sizes=(16, 16),
                ),
            )
            t.train(ds)
            model = pathlib.Path(tmp) / "nn_weights.npz"
            loaded = np.load(model)
            assert "energy_shift" in loaded
            ref_mean = np.mean([a.calc.results["energy"] for a in ds])
            assert abs(float(loaded["energy_shift"]) - ref_mean) < 1e-9

            from gdpx.potential.nnp.calculator import ACSFNN

            calc = ACSFNN(model_file=model)
            assert abs(calc.energy_shift - ref_mean) < 1e-9
            # predictions include the shift: reproduce the *uncentered* energies
            el, _ = _model_losses(model, ds)
            assert el < 1.0, f"shifted energies not recovered: energy loss {el:.4f}"

    def test_combined_reduces_energy_and_force_loss(self):
        ds = _pair_dataset(offset=3.0, seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            t = NnpTrainer(
                config=dict(n_epochs=60, learning_rate=0.003, force_weight=10.0, verbose=0),
                directory=tmp,
                calculator_params=dict(
                    elements=["Cu"],
                    g2_params=[(0.1, 0.0), (0.2, 0.0)],
                    g4_params=[(0.01, 1.0, 1.0)],
                    r_cut=6.0,
                    hidden_sizes=(16, 16),
                ),
            )
            t.train(ds)
            el, fl = _model_losses(pathlib.Path(tmp) / "nn_weights.npz", ds)
            assert el < 0.5, f"energy loss not reduced: {el:.4f}"
            assert fl < 0.25, f"force loss not reduced: {fl:.4f}"
