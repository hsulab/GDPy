from ase import Atoms
from ase.calculators.emt import EMT
from ase.io import read, write

from gdpx.cli.compute import run_computation


def test_one_shot_compute_runs_neb_reactor(tmp_path):
    initial = Atoms("Cu", positions=[[1.0, 1.0, 1.0]], cell=[4.0, 4.0, 4.0], pbc=True)
    final = initial.copy()
    final.positions[0, 0] += 0.2
    for atoms in (initial, final):
        atoms.calc = EMT()
        atoms.get_potential_energy()
    structures = tmp_path / "endpoints.xyz"
    write(structures, [initial, final])

    config = {
        "potential": {"provider": "emt"},
        "executor": {
            "provider": "ase",
            "method": "neb",
            "parameters": {
                "setup": {
                    "nimages": 3,
                    "interpolation": {"mic": False},
                    "optimizer": {"name": "bfgs", "parameters": {"maxstep": 0.1}},
                },
                "output": {"trajectory": {"period": 1}},
                "stop": {"fmax": 0.5, "steps": 1},
            },
        },
    }

    result = run_computation([str(structures)], config, directory=tmp_path / "run")

    assert result.number_of_trajectories == 1
    assert len(read(result.end_frames, ":")) == 3
