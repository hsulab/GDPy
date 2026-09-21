"""Opt in with GDPX_TEST_TACE=1; requires the extra and downloads model weights."""
import os

import numpy as np
import pytest
from ase import Atoms
from ase.optimize import BFGS

pytestmark = pytest.mark.skipif(
    os.environ.get('GDPX_TEST_TACE') != '1', reason='Set GDPX_TEST_TACE=1 for real TACE inference')


def test_foundation_checkpoint_and_relaxation(tmp_path):
    from tace.foundations import tace_foundations
    from tace.interface.ase import TACEAseCalc
    from gdpx.providers.tace.manager import TaceManager

    atoms = Atoms('Cu4O4', positions=[[9,9,10], [11,9,10], [11,11,10], [9,11,10],
                  [10,8,10], [12,10,10], [10,12,10], [8,10,10]], cell=[20]*3, pbc=True)
    manager = TaceManager()
    manager.register_calculator(dict(model='TACE-OAM-7M', device='cpu', fidelity_idx=0))
    atoms.calc = manager.calc
    energy, forces = atoms.get_potential_energy(), atoms.get_forces()
    assert np.isfinite(energy) and np.isfinite(forces).all()
    checkpoint = str(tace_foundations['TACE-OAM-7M'])
    reference = atoms.copy()
    reference.calc = TACEAseCalc(model=checkpoint, device='cpu', dtype='float32', fidelity_idx=0)
    np.testing.assert_allclose(energy, reference.get_potential_energy(), atol=1e-5)
    np.testing.assert_allclose(forces, reference.get_forces(), atol=1e-5)
    local = TaceManager()
    local.register_calculator(dict(model=checkpoint, device='cpu', fidelity_idx=0))
    reference.calc = local.calc
    np.testing.assert_allclose(energy, reference.get_potential_energy(), atol=1e-5)
    assert BFGS(atoms, logfile=str(tmp_path / 'relax.log')).run(fmax=0.05, steps=300)
    assert np.linalg.norm(atoms.get_forces(), axis=1).max() <= 0.05
