import numpy as np
from ase import Atoms

from gdpx.exploration.genetic_algorithm.mutation.exchange import ExchangeMutation


def _exchange_mutation(bounds, seed=1):
    return ExchangeMutation(
        species="Cu",
        bond_distance_dict={(29, 29): 2.0},
        num_min_max=list(bounds),
        region={
            "method": "lattice",
            "origin": [0.0, 0.0, 0.0],
            "cell": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        },
        rng=np.random.default_rng(seed),
    )


def test_exchange_removes_at_inclusive_upper_bound():
    atoms = Atoms(
        "Cu4",
        positions=[[2.0, 2.0, 2.0], [4.0, 2.0, 2.0], [2.0, 4.0, 2.0], [4.0, 4.0, 2.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
        tags=[1, 2, 3, 4],
    )

    mutant, description = _exchange_mutation((1.0, 4.0)).mutate(atoms)

    assert len(mutant) == 3
    assert description.startswith("remove_Cu_")
