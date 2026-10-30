"""Opt-in real MatterSim GCMC lifecycle regression (uses pretrained weights)."""
import os

import numpy as np
import pytest
from ase.build import bulk
from ase.io import read

from gdpx.execution.lifecycle import load_compute_plan, prepare_compute
from gdpx.execution.lifecycle.service import orchestrate_compute
from test_sampling_config import sampling_config


@pytest.mark.skipif(os.environ.get("GDPX_TEST_MATTERSIM") != "1",
                    reason="Set GDPX_TEST_MATTERSIM=1 for real MatterSim inference")
def test_single_biased_gcmc_mattersim(tmp_path):
    config = sampling_config("biased_monte_carlo")
    config["potential"] = {
        "provider": "mattersim", "backend": "ase",
        "parameters": {"model": "MatterSim-v1.0.0-1M", "device": "cpu"},
    }
    config["dispatch"] = {"worker": "single"}
    params = config["executor"]["parameters"]
    params["ensemble"] = {
        "method": "uvt", "temperature": 333.0, "chemical_potentials": {"Cu": -4.0},
    }
    params["setup"]["operators"] = [{
        "method": "biased_volume_exchange", "particles": ["Cu"],
        "probability": 1.0, "max_random_attempts": 1, "skip_distance_check": True,
        "region": {"method": "lattice", "origin": [0., 0., 0.],
                   "cell": [[8., 0., 0.], [0., 8., 0.], [0., 0., 8.]]},
    }]
    params["setup"]["evaluation"] = {
        "provider": "ase", "steps": 2, "fmax": 0.05, "ignore_convergence": True,
    }
    params["stop"]["steps"] = 1
    atoms = bulk("Cu", cubic=True)
    atoms.set_cell([8.] * 3)
    atoms.center()
    plan = prepare_compute(config, [atoms], tmp_path)
    assert orchestrate_compute(plan).state == "finished"
    trial = read(tmp_path / "steps/step.0001/traj.xyz", -1)
    assert abs(len(trial) - len(atoms)) == 1
    assert np.isfinite(trial.get_potential_energy())
    assert np.isfinite(trial.get_forces()).all()
    trajectory = tmp_path / "mc.xyz"
    assert len(read(trajectory, ":")) == 2
    before = trajectory.read_bytes()
    assert orchestrate_compute(load_compute_plan(tmp_path)).state == "finished"
    assert trajectory.read_bytes() == before
