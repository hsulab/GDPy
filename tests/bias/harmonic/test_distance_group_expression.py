import numpy as np
import pytest
from ase import Atoms

from gdpx.modifiers.bias.harmonic.distance import DistanceHarmonicCalculator


def test_distance_harmonic_resolves_group_expression(tmp_path):
    atoms = Atoms("H3", positions=[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    atoms.calc = DistanceHarmonicCalculator(
        group="`index 0 2`",
        center=1.5,
        kspring=2.0,
        directory=tmp_path,
    )

    assert atoms.get_potential_energy() == pytest.approx(0.25)
    np.testing.assert_allclose(atoms.get_forces(), [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])


def test_distance_harmonic_requires_expression_to_select_two_atoms(tmp_path):
    atoms = Atoms("H3", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    atoms.calc = DistanceHarmonicCalculator(
        group="`symbol H`",
        center=1.5,
        kspring=2.0,
        directory=tmp_path,
    )

    with pytest.raises(ValueError, match="must select exactly two atoms; selected 3"):
        atoms.get_potential_energy()
