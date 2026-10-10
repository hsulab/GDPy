"""The two-water example must contain distinct seeds and usable tagged rotations."""

from pathlib import Path
import runpy

import numpy as np
import pytest
import yaml
from ase.io import read

from gdpx.exploration.factory import create_exploration
from gdpx.exploration.sampling.geometry import prepare_operators
from gdpx.structures.groups import evaluate_constraint_expression

EXAMPLES = Path(__file__).resolve().parents[2] / "examples/global_optimisation"
DEMO = runpy.run_path(str(EXAMPLES / "tio2_water_rotation.py"))


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
    parallel_report = DEMO["geometry_report"](parallel)
    bonded_report = DEMO["geometry_report"](bonded)
    assert parallel_report["parallel"] and not parallel_report["hydrogen_bonds"]
    assert bonded_report["hydrogen_bonds"] and not bonded_report["parallel"]
    assert len(bonded_report["hydrogen_bonds"]) == waters-1
    assert parallel_report["molecular_adsorption"] and bonded_report["molecular_adsorption"]


@pytest.mark.parametrize("waters", [2, 3])
def test_example_population_accepts_seeds_and_rotation_preserves_anchors(waters):
    seeds = EXAMPLES / f"assets/tio2_water{waters}_seeds.xyz"
    recipe = yaml.safe_load((EXAMPLES / f"explorations/basin_hopping/tio2_water{waters}.yaml").read_text())
    recipe["system"]["builders"]["seeds"]["frames"] = str(seeds)
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
    ff = xreac.ForceField.bundled(DEMO["MODEL"])
    for seed in read(EXAMPLES / f"assets/tio2_water{waters}_seeds.xyz", ":"):
        seed.calc = ReaxFFCalculator(ff)
        assert np.isfinite(seed.get_potential_energy())
        assert np.all(np.isfinite(seed.get_forces()))
