import numpy as np
import pytest
from ase import Atoms
from omegaconf import OmegaConf

from gdpx.builder.random_structure import RandomStructureImprovedModifier
from gdpx.describer.connectivity import ConnectivityDescriber
from gdpx.expedition.persist.thanos import dispatch_thanos
from gdpx.geometry.insert import insert_fragments_by_step
from gdpx.geometry.restraints import evaluate_restraints, parse_restraints
from gdpx.geometry.spatial import get_bond_distance_dict


def make_atoms(symbols, positions, tags=None, pbc=False):
    atoms = Atoms(symbols, positions=positions, cell=[10.0, 10.0, 10.0], pbc=pbc)
    if tags is not None:
        atoms.set_tags(tags)
    return atoms


def test_contact_count_uses_custom_half_open_distance_window():
    config = [
        {
            "type": "contact_count",
            "pair": ["C", "O"],
            "min": 1,
            "max": 1,
            "distance": {"min": 1.0, "max": 2.0},
        }
    ]
    restraints = parse_restraints(config)

    assert evaluate_restraints(make_atoms("CO", [[0, 0, 0], [1.0, 0, 0]], [1, 2]), restraints)
    assert evaluate_restraints(make_atoms("CO", [[0, 0, 0], [1.5, 0, 0]], [1, 2]), restraints)
    assert not evaluate_restraints(make_atoms("CO", [[0, 0, 0], [2.0, 0, 0]], [1, 2]), restraints)
    assert not evaluate_restraints(make_atoms("CO", [[0, 0, 0], [0.9, 0, 0]], [1, 2]), restraints)


def test_distance_bounds_fall_back_to_covalent_ratio_independently():
    restraints = parse_restraints(
        [
            {
                "type": "contact_count",
                "pair": ["C", "O"],
                "min": 1,
                "distance": {"max": 2.0},
            }
        ],
        covalent_ratio=[0.5, 2.0],
    )
    # ASE C/O radii sum to 1.42 A, so the derived minimum is 0.71 A.
    assert restraints[0].distance.minimum == pytest.approx(0.71)
    assert restraints[0].distance.maximum == pytest.approx(2.0)


def test_restraints_accept_omegaconf_sequences():
    config = OmegaConf.create(
        [{"type": "contact_count", "pair": ["C", "O"], "max": 0}]
    )
    assert len(parse_restraints(config)) == 1


def test_inter_particle_is_default_and_requires_explicit_tags():
    restraints = parse_restraints(
        [{"type": "contact_count", "pair": ["C", "O"], "max": 0, "distance": {"max": 2.0}}]
    )
    with pytest.raises(ValueError, match="tags"):
        evaluate_restraints(make_atoms("CO", [[0, 0, 0], [1.5, 0, 0]]), restraints)

    same_particle = make_atoms("CO", [[0, 0, 0], [1.5, 0, 0]], [4, 4])
    assert evaluate_restraints(same_particle, restraints)

    different_particles = make_atoms("CO", [[0, 0, 0], [1.5, 0, 0]], [0, 4])
    assert not evaluate_restraints(different_particles, restraints)


def test_all_atoms_scope_ignores_tags():
    restraints = parse_restraints(
        [
            {
                "type": "contact_count",
                "pair": ["C", "O"],
                "scope": "all_atoms",
                "max": 0,
                "distance": {"max": 2.0},
            }
        ]
    )
    atoms = make_atoms("CO", [[0, 0, 0], [1.5, 0, 0]])
    assert not evaluate_restraints(atoms, restraints)


def test_coordination_supports_all_centers_and_matching_center_count():
    atoms = make_atoms("CCO", [[0, 0, 0], [4, 0, 0], [1.2, 0, 0]], [1, 2, 3])
    common = {
        "type": "coordination",
        "center": "C",
        "neighbor": "O",
        "coordination": {"min": 1, "max": 1},
        "distance": {"min": 0.8, "max": 1.8},
    }

    all_centers = parse_restraints([{**common, "matching_centers": "all"}])
    assert not evaluate_restraints(atoms, all_centers)

    one_center = parse_restraints(
        [{**common, "matching_centers": {"min": 1, "max": 1}}]
    )
    assert evaluate_restraints(atoms, one_center)


def test_contact_distance_uses_minimum_image_convention():
    atoms = make_atoms("CO", [[0.1, 0, 0], [9.1, 0, 0]], [1, 2], pbc=True)
    restraints = parse_restraints(
        [
            {
                "type": "contact_count",
                "pair": ["C", "O"],
                "min": 1,
                "distance": {"min": 0.8, "max": 1.2},
            }
        ]
    )
    assert evaluate_restraints(atoms, restraints)


def test_thanos_and_connectivity_use_shared_restraints():
    config = [
        {
            "type": "contact_count",
            "pair": ["C", "O"],
            "max": 0,
            "distance": {"min": 1.0, "max": 2.0},
        }
    ]
    atoms = make_atoms("CO", [[0, 0, 0], [1.5, 0, 0]], [1, 2])

    extinct = dispatch_thanos(name="restraints", restraints=config)
    assert extinct(atoms) == 1
    assert ConnectivityDescriber(restraints=config).run([atoms]).tolist() == [0]


class SequenceRegion:
    def __init__(self, positions):
        self.positions = iter(positions)

    def get_random_positions(self, size, rng):
        assert size == 1
        return np.asarray([next(self.positions)], dtype=float)


def test_fragment_insertion_rejects_forbidden_contact():
    config = [
        {
            "type": "contact_count",
            "pair": ["C", "O"],
            "max": 0,
            "distance": {"min": 1.0, "max": 2.0},
        }
    ]
    restraints = parse_restraints(config)
    numbers = [6, 8]
    result = insert_fragments_by_step(
        substrate=Atoms("", cell=[10, 10, 10], pbc=False),
        fragments=[Atoms("C"), Atoms("O")],
        region=SequenceRegion([[0, 0, 0], [1.5, 0, 0]]),
        molecular_distances=[-np.inf, np.inf],
        covalent_ratio=[0.8, 2.0],
        bond_distance_dict=get_bond_distance_dict(numbers),
        random_state=7,
        restraints=restraints,
        max_attempts=1,
    )
    assert result is None


def test_random_structure_improved_returns_only_complete_valid_candidates():
    builder = RandomStructureImprovedModifier(
        composition={"C": 1, "O": 1},
        box=[6.0, 6.0, 6.0],
        random_seed=7,
        max_times_size=2,
        restraints=[
            {
                "type": "contact_count",
                "pair": ["C", "O"],
                "min": 1,
                "max": 1,
                "distance": {"min": 0.5, "max": 10.0},
            }
        ],
    )
    frames = builder.run(size=1)

    assert len(frames) == 1
    assert frames[0].has("tags")
    assert evaluate_restraints(frames[0], builder.restraints)


@pytest.mark.parametrize(
    "config, message",
    [
        ([{"type": "contact_count", "pair": ["C", "Xx"], "max": 0}], "chemical symbol"),
        (
            [
                {
                    "type": "coordination",
                    "center": "C",
                    "neighbor": "O",
                    "coordination": {"min": 2, "max": 1},
                    "matching_centers": "all",
                }
            ],
            "cannot be greater",
        ),
    ],
)
def test_invalid_restraints_fail_clearly(config, message):
    with pytest.raises(ValueError, match=message):
        parse_restraints(config)
