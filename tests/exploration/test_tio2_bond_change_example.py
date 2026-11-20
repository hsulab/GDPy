"""TiO2 fixtures exercise geometric water splitting and joining with xreac."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from ase.io import read

from gdpx.structures.geometry.fragments import FragmentPolicy
from gdpx.utils.atoms_tags import get_tags_per_species

from gdpx.exploration.factory import create_exploration
from gdpx.sampling import parse_operators
from gdpx.sampling.geometry import prepare_operators

EXAMPLES = Path(__file__).resolve().parents[2] / "examples/global_optimisation"


OH_CUTOFF = FragmentPolicy().cutoff(("O", "H"))


def adsorption_report(atoms):
    """Classify each mobile H by its nearest bonded oxygen, without imposing bonds."""
    tags = atoms.get_tags()
    oxygens = np.flatnonzero(atoms.numbers == 8)
    hydrogen_sites = []
    counts = {int(i): 0 for i in oxygens if tags[i] > 0}
    for hydrogen in np.flatnonzero((atoms.numbers == 1) & (tags > 0)):
        distances = atoms.get_distances(hydrogen, oxygens, mic=True)
        nearest = int(np.argmin(distances))
        oxygen, distance = int(oxygens[nearest]), float(distances[nearest])
        kind = "free" if distance > OH_CUTOFF else ("surface" if tags[oxygen] == 0 else "adsorbate")
        if kind == "adsorbate":
            counts[oxygen] += 1
        hydrogen_sites.append(dict(hydrogen=int(hydrogen), oxygen=oxygen, distance_angstrom=distance, kind=kind))
    ti = np.flatnonzero(atoms.numbers == 22)
    ti_o = {str(oxygen): float(atoms.get_distances(oxygen, ti, mic=True).min()) for oxygen in counts}
    return dict(molecular_waters=sum(count == 2 for count in counts.values()),
                hydroxyls=sum(count == 1 for count in counts.values()),
                surface_hydrogens=sum(site["kind"] == "surface" for site in hydrogen_sites),
                free_hydrogens=sum(site["kind"] == "free" for site in hydrogen_sites),
                hydrogens_per_adsorbate_oxygen=counts, hydrogen_sites=hydrogen_sites,
                ti_o_angstrom=ti_o, all_oxygens_adsorbed=all(distance < 2.8 for distance in ti_o.values()),
                fragments={species: len(groups) for species, groups in get_tags_per_species(atoms).items()
                           if any(tag > 0 for tag, _ in groups)})


@pytest.mark.parametrize("waters", [2, 3])
def test_seeds_and_bh_recipe_preserve_slab_and_describe_dissociation(waters, monkeypatch):
    monkeypatch.chdir(EXAMPLES / "basin_hopping")
    seeds = read(EXAMPLES / f"assets/tio2_water{waters}_bond_seeds.xyz", ":")
    assert len(seeds) == 2
    recipe = yaml.safe_load((EXAMPLES / f"basin_hopping/tio2_water{waters}_bond_change.yaml").read_text())
    search = create_exploration(recipe)
    for stored in seeds:
        search.population_config.validate_candidate(stored, "archived water bond-change seed")
        assert stored.get_chemical_formula() == f"H{2*waters}O{128+waters}Ti64"
        assert len(stored) == 192+3*waters
        assert stored.pbc.all()
        assert len(stored.constraints[0].get_indices()) == 48
        assert (stored.get_tags()[:192] == 0).all()
    np.testing.assert_array_equal(seeds[0].positions[:192], seeds[1].positions[:192])
    reference = read(EXAMPLES / f"assets/tio2_water{waters}_seeds.xyz")
    fixed = reference.constraints[0].get_indices()
    np.testing.assert_array_equal(seeds[0].positions[fixed], reference.positions[fixed])
    molecular, dissociated = [adsorption_report(atoms) for atoms in seeds]
    assert molecular["molecular_waters"] == waters and molecular["hydroxyls"] == 0
    assert dissociated["molecular_waters"] == waters-1
    assert dissociated["hydroxyls"] == dissociated["surface_hydrogens"] == 1
    assert dissociated["free_hydrogens"] == 0
    assert seeds[1].info["bond_change"]["operation"] == "break"
    assert seeds[1].info["bond_change"]["bond_distance"] == pytest.approx(1.5)


@pytest.mark.parametrize("waters", [2, 3])
def test_form_operator_recombines_oh_and_free_h_on_tio2(waters):
    atoms = read(EXAMPLES / f"assets/tio2_water{waters}_bond_seeds.xyz")
    atoms.positions[193] = atoms.positions[192]+[0., 0., 2.5]
    FragmentPolicy().reconcile(atoms)
    report = adsorption_report(atoms)
    assert report["free_hydrogens"] == report["hydroxyls"] == 1
    original = atoms.positions.copy()
    op = parse_operators([dict(method="bond_form", particles=["H"], bond=["O", "H"],
                               cutoffs=[1.2, 3.5])])[0][0]
    op._print = op._debug = lambda *args: None
    prepare_operators([op], sorted(set(atoms.numbers)))
    proposal = op.propose(atoms, np.random.default_rng(12))
    assert proposal.valid
    assert proposal.metadata["bond_change"]["operation"] == "form"
    assert adsorption_report(atoms)["molecular_waters"] == waters
    np.testing.assert_array_equal(atoms.positions[:192], original[:192])
    proposal.rollback()
    np.testing.assert_array_equal(atoms.positions, original)


@pytest.mark.parametrize("waters", [2, 3])
def test_xreac_seed_energies_and_forces_are_finite(waters):
    xreac = pytest.importorskip("xreac")
    from xreac.ase import ReaxFFCalculator
    force_field = xreac.ForceField.bundled(yaml.safe_load((EXAMPLES / "runtimes/xreac_tioh.yaml").read_text())["potential"]["parameters"]["model"].removeprefix("bundled:"))
    for atoms in read(EXAMPLES / f"assets/tio2_water{waters}_bond_seeds.xyz", ":"):
        atoms.calc = ReaxFFCalculator(force_field)
        assert np.isfinite(atoms.get_potential_energy())
        assert np.isfinite(atoms.get_forces()).all()
