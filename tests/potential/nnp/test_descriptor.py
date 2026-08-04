import numpy as np
from ase import Atoms

from gdpx.potential.nnp.descriptor import compute_symmetry_functions


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
