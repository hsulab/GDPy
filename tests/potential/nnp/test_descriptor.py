import numpy as np
from ase import Atoms

from gdpx.potential.nnp.descriptor import (
    compute_force_gradient_weights,
    compute_forces,
    compute_symmetry_functions,
    compute_symmetry_functions_and_derivatives,
)


class TestDescriptorValues:
    def test_g2_g4_values(self):
        atoms = Atoms(
            "Cu2Au", positions=[[0, 0, 0], [2.5, 0, 0], [1.0, 2.0, 1.5]], pbc=False
        )
        desc = compute_symmetry_functions(
            atoms,
            ["Cu", "Au"],
            [(0.1, 0.0), (0.2, 0.5)],
            [(0.01, 1.0, 1.0), (0.02, 2.0, -1.0)],
            6.0,
        )
        expected = np.array(
            [
                [0.3368986402, 0.2828119288, 0.2809734651, 0.2218001421,
                 0.0000000000, 0.0000000000, 0.2098214024, 0.0242591426,
                 0.0000000000, 0.0000000000],
                [0.3368986402, 0.2828119288, 0.2231623640, 0.1625514855,
                 0.0000000000, 0.0000000000, 0.2317163364, 0.0144710443,
                 0.0000000000, 0.0000000000],
                [0.5041358291, 0.3843516276, 0.0000000000, 0.0000000000,
                 0.2455761352, 0.0095746935, 0.0000000000, 0.0000000000,
                 0.0000000000, 0.0000000000],
            ]
        )
        assert desc.shape == (3, 10)
        assert np.allclose(desc, expected, atol=1e-9), f"descriptor mismatch:\n{desc}"

    def test_sparse_jacobian_matches_direct_contractions(self):
        atoms = Atoms(
            "Cu2Au",
            positions=[[0, 0, 0], [2.5, 0, 0], [1.0, 2.0, 1.5]],
            pbc=False,
        )
        elements = ["Cu", "Au"]
        g2 = [(0.1, 0.0), (0.2, 0.5)]
        g4 = [(0.01, 1.0, 1.0), (0.02, 2.0, -1.0)]
        desc, jacobian = compute_symmetry_functions_and_derivatives(
            atoms, elements, g2, g4, 6.0
        )
        rng = np.random.default_rng(8)
        dE_dG = rng.normal(size=desc.shape)
        force_residual = rng.normal(size=(len(atoms), 3))
        assert np.allclose(
            jacobian.forces(dE_dG),
            compute_forces(atoms, elements, g2, g4, 6.0, dE_dG),
            atol=1e-12,
        )
        assert np.allclose(
            jacobian.adjoint(force_residual),
            compute_force_gradient_weights(
                atoms, elements, g2, g4, 6.0, force_residual
            ),
            atol=1e-12,
        )

    def test_sparse_jacobian_matches_descriptor_finite_difference(self):
        atoms = Atoms(
            "Cu2Au",
            positions=[[0, 0, 0], [2.5, 0, 0], [1.0, 2.0, 1.5]],
            pbc=False,
        )
        args = (["Cu", "Au"], [(0.1, 0.0)], [(0.01, 1.0, 1.0)], 6.0)
        desc, jacobian = compute_symmetry_functions_and_derivatives(atoms, *args)
        rng = np.random.default_rng(9)
        seed = rng.normal(size=desc.shape)
        analytical = -jacobian.forces(seed)
        finite_difference = np.zeros((len(atoms), 3))
        step = 1.0e-6
        for atom_index in range(len(atoms)):
            for direction in range(3):
                original = atoms.positions[atom_index, direction]
                atoms.positions[atom_index, direction] = original + step
                plus = np.sum(seed * compute_symmetry_functions(atoms, *args))
                atoms.positions[atom_index, direction] = original - step
                minus = np.sum(seed * compute_symmetry_functions(atoms, *args))
                atoms.positions[atom_index, direction] = original
                finite_difference[atom_index, direction] = (
                    plus - minus
                ) / (2.0 * step)
        assert np.allclose(analytical, finite_difference, atol=1e-7)
