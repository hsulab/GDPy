"""TiO2 fixtures exercise geometric water splitting and joining with xreac."""

from pathlib import Path
import runpy

import numpy as np
import pytest
import yaml
from ase.io import read

from gdpx.exploration.factory import create_exploration
from gdpx.exploration.sampling import parse_operators
from gdpx.exploration.sampling.geometry import prepare_operators

EXAMPLES = Path(__file__).resolve().parents[2] / "examples/global_optimisation"
DEMO = runpy.run_path(str(EXAMPLES / "tio2_water_bond_change.py"))


@pytest.mark.parametrize("waters", [2, 3])
def test_seeds_and_bh_recipe_preserve_slab_and_describe_dissociation(waters):
    seeds = read(EXAMPLES / f"assets/tio2_water{waters}_bond_seeds.xyz", ":")
    rebuilt = DEMO["build_seeds"](waters)
    recipe = yaml.safe_load((EXAMPLES / f"explorations/basin_hopping/tio2_water{waters}_bond_change.yaml").read_text())
    recipe["system"]["builders"]["seeds"]["frames"] = str(EXAMPLES / f"assets/tio2_water{waters}_bond_seeds.xyz")
    search = create_exploration(recipe)
    for stored, generated in zip(seeds, rebuilt):
        search.population_config.validate_candidate(stored, "archived water bond-change seed")
        search.population_config.validate_candidate(generated, "regenerated water bond-change seed")
        assert generated.get_chemical_formula() == stored.get_chemical_formula()
        np.testing.assert_array_equal(generated.get_tags()[:192], 0)
        np.testing.assert_array_equal(generated.positions[generated.constraints[0].get_indices()],
                                      stored.positions[stored.constraints[0].get_indices()])
        assert len(stored) == 192+3*waters
        assert stored.pbc.all()
        assert len(stored.constraints[0].get_indices()) == 48
        assert (stored.get_tags()[:192] == 0).all()
    np.testing.assert_array_equal(seeds[0].positions[:192], seeds[1].positions[:192])
    reference = read(EXAMPLES / f"assets/tio2_water{waters}_seeds.xyz")
    fixed = reference.constraints[0].get_indices()
    np.testing.assert_array_equal(seeds[0].positions[fixed], reference.positions[fixed])
    for pair in (seeds, rebuilt):
        molecular, dissociated = [DEMO["adsorption_report"](atoms) for atoms in pair]
        assert molecular["molecular_waters"] == waters and molecular["hydroxyls"] == 0
        assert dissociated["molecular_waters"] == waters-1
        assert dissociated["hydroxyls"] == dissociated["surface_hydrogens"] == 1
        assert dissociated["free_hydrogens"] == 0
    events = rebuilt[1].info["seed_bond_changes"]
    assert [event["operation"] for event in events] == ["break", "form"]
    assert events[0]["bond_distance"] == pytest.approx(1.455)
    assert events[1]["bond_distance"] == pytest.approx(0.97)
    assert seeds[1].info["bond_change"]["operation"] == "break"
    assert seeds[1].info["bond_change"]["bond_distance"] == pytest.approx(1.5)


@pytest.mark.parametrize("waters", [2, 3])
def test_form_operator_recombines_oh_and_free_h_on_tio2(waters):
    atoms = read(EXAMPLES / f"assets/tio2_water{waters}_bond_seeds.xyz")
    atoms.positions[193] = atoms.positions[192]+[0., 0., 2.5]
    DEMO["POLICY"].reconcile(atoms)
    report = DEMO["adsorption_report"](atoms)
    assert report["free_hydrogens"] == report["hydroxyls"] == 1
    original = atoms.positions.copy()
    op = parse_operators([dict(method="bond_form", particles=["H"], bond=["O", "H"],
                               cutoffs=[1.2, 3.5])])[0][0]
    op._print = op._debug = lambda *args: None
    prepare_operators([op], sorted(set(atoms.numbers)))
    proposal = op.propose(atoms, np.random.default_rng(12))
    assert proposal.valid
    assert proposal.metadata["bond_change"]["operation"] == "form"
    assert DEMO["adsorption_report"](atoms)["molecular_waters"] == waters
    np.testing.assert_array_equal(atoms.positions[:192], original[:192])
    proposal.rollback()
    np.testing.assert_array_equal(atoms.positions, original)


@pytest.mark.parametrize("waters", [2, 3])
def test_xreac_seed_energies_and_forces_are_finite(waters):
    xreac = pytest.importorskip("xreac")
    from xreac.ase import ReaxFFCalculator
    force_field = xreac.ForceField.bundled(DEMO["MODEL"])
    for atoms in read(EXAMPLES / f"assets/tio2_water{waters}_bond_seeds.xyz", ":"):
        atoms.calc = ReaxFFCalculator(force_field)
        assert np.isfinite(atoms.get_potential_energy())
        assert np.isfinite(atoms.get_forces()).all()
