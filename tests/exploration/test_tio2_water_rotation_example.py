"""The two-water example must contain distinct seeds and usable tagged rotations."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from ase.io import read

from gdpx.exploration.factory import create_exploration
from gdpx.sampling.geometry import prepare_operators
from gdpx.structures.groups import evaluate_constraint_expression

EXAMPLES = Path(__file__).resolve().parents[2] / "examples/global_optimisation"


def geometry_report(atoms):
    """Measure intact adsorption, dipole alignment, and donor-H-acceptor geometry."""
    waters = [list(range(index, index+3)) for index in range(192, len(atoms), 3)]
    ti = np.flatnonzero(atoms.numbers[:192] == 22)
    rows, bisectors = [], []
    for oxygen, *hydrogens in waters:
        oh = atoms.get_distances(oxygen, hydrogens, mic=True)
        vectors = atoms.get_distances(oxygen, hydrogens, mic=True, vector=True)
        bisector = vectors.sum(axis=0)
        bisectors.append(bisector / np.linalg.norm(bisector))
        distances = atoms.get_distances(oxygen, ti, mic=True)
        nearest = int(np.argmin(distances))
        rows.append(dict(oh_angstrom=oh.tolist(), nearest_ti=int(ti[nearest]),
                         ti_o_angstrom=float(distances[nearest]),
                         intact=bool(np.all((oh > 0.7) & (oh < 1.25))),
                         adsorbed=bool(distances[nearest] < 2.8)))
    oo_distances = [float(atoms.get_distance(waters[i][0], waters[j][0], mic=True))
                    for i in range(len(waters)) for j in range(i+1, len(waters))]
    hydrogen_bonds = []
    contacts = []
    for donor, acceptor in [(i, j) for i in range(len(waters)) for j in range(len(waters)) if i != j]:
        oxygen = waters[donor][0]
        other = waters[acceptor][0]
        oo = float(atoms.get_distance(oxygen, other, mic=True))
        for hydrogen in waters[donor][1:]:
            hd = atoms.get_distance(hydrogen, oxygen, mic=True, vector=True)
            ha = atoms.get_distance(hydrogen, other, mic=True, vector=True)
            distance = float(np.linalg.norm(ha))
            angle = float(np.rad2deg(np.arccos(np.clip(np.dot(hd, ha) / (np.linalg.norm(hd)*distance), -1, 1))))
            contact = dict(donor=oxygen, hydrogen=hydrogen, acceptor=other,
                           h_o_angstrom=distance, o_h_o_degrees=angle)
            contacts.append(contact)
            if oo <= 3.5 and distance <= 2.5 and angle >= 150:
                hydrogen_bonds.append(contact)
    alignment = float(min(np.clip(np.dot(bisectors[i], bisectors[j]), -1, 1)
                          for i in range(len(waters)) for j in range(i+1, len(waters))))
    return dict(waters=rows, o_o_angstrom=min(oo_distances), o_o_distances_angstrom=oo_distances,
                dipole_cosine=alignment,
                parallel=bool(alignment > np.cos(np.deg2rad(20))),
                hydrogen_bonds=hydrogen_bonds, water_contacts=contacts,
                molecular_adsorption=all(row["intact"] and row["adsorbed"] for row in rows))


@pytest.mark.parametrize("waters", [2, 3])
def test_seeds_have_neighboring_parallel_and_hydrogen_bonded_waters(waters):
    seeds = EXAMPLES / f"assets/tio2_water{waters}_seeds.xyz"
    parallel, bonded = read(seeds, ":")
    for atoms in [parallel, bonded]:
        assert len(atoms) == 192+3*waters
        assert atoms.get_chemical_formula() == f"H{2*waters}O{128+waters}Ti64"
        np.testing.assert_array_equal(atoms.get_tags(), [0]*192+[tag for tag in range(1,waters+1) for _ in range(3)])
        assert len(atoms.constraints[0].index) == 48
    np.testing.assert_array_equal(parallel.positions[:192], bonded.positions[:192])
    parallel_report = geometry_report(parallel)
    bonded_report = geometry_report(bonded)
    assert parallel_report["parallel"] and not parallel_report["hydrogen_bonds"]
    assert bonded_report["hydrogen_bonds"] and not bonded_report["parallel"]
    assert len(bonded_report["hydrogen_bonds"]) == waters-1
    assert parallel_report["molecular_adsorption"] and bonded_report["molecular_adsorption"]


@pytest.mark.parametrize("waters", [2, 3])
def test_example_population_accepts_seeds_and_rotation_preserves_anchors(waters, monkeypatch):
    monkeypatch.chdir(EXAMPLES / "basin_hopping")
    seeds = EXAMPLES / f"assets/tio2_water{waters}_seeds.xyz"
    recipe = yaml.safe_load((EXAMPLES / f"basin_hopping/tio2_water{waters}.yaml").read_text())
    search = create_exploration(recipe)
    runtime = yaml.safe_load((EXAMPLES / "runtimes/xreac_tioh.yaml").read_text())
    constraint = runtime["executor"]["parameters"]["setup"]["constraint"]
    for seed in read(seeds, ":"):
        search.population_config.validate_candidate(seed, "test seeds")
        _, fixed = evaluate_constraint_expression(seed, constraint)
        assert sorted(fixed) == sorted(seed.constraints[0].index)
        original = seed.positions.copy()
        op = search.operators[0]
        op._print = op._debug = lambda *args: None
        prepare_operators([op], sorted(set(seed.numbers)))
        proposal = op.propose(seed, np.random.default_rng(19))
        assert proposal.valid
        oxygen_indices = list(range(192,len(seed),3))
        np.testing.assert_array_equal(seed.positions[oxygen_indices], original[oxygen_indices])
        np.testing.assert_array_equal(seed.positions[:192], original[:192])
        proposal.rollback()
        np.testing.assert_array_equal(seed.positions, original)


@pytest.mark.parametrize("waters", [2, 3])
def test_latest_xreac_has_finite_energy_and_forces_for_both_seeds(waters):
    xreac = pytest.importorskip("xreac")
    from xreac.ase import ReaxFFCalculator
    ff = xreac.ForceField.bundled(yaml.safe_load((EXAMPLES / "runtimes/xreac_tioh.yaml").read_text())["potential"]["parameters"]["model"].removeprefix("bundled:"))
    for seed in read(EXAMPLES / f"assets/tio2_water{waters}_seeds.xyz", ":"):
        seed.calc = ReaxFFCalculator(ff)
        assert np.isfinite(seed.get_potential_energy())
        assert np.all(np.isfinite(seed.get_forces()))
