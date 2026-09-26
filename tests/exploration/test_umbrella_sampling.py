import json

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from gdpx.analysis.selectors.clustering import group_structures_by_info
from gdpx.data.array import AtomsNDArray
from gdpx.execution.factory import create_worker
from gdpx.exploration.factory import create_exploration
from gdpx.workflow.factory import create_selector


def _config(frames, **strategy):
    settings = {
        "centers": [1.9, 2.3],
        "kspring": 2.0,
        "replicas": 2,
        "equilibration_steps": 1,
    }
    settings.update(strategy)
    return {
        "method": "umbrella_sampling",
        "random_seed": 17,
        "system": {
            "builder": {"method": "direct", "frames": frames},
            "collective_variable": {"method": "distance", "group": "`index 0 1`"},
        },
        "strategy": settings,
    }


def _runtime():
    return {
        "potential": {"provider": "emt"},
        "executor": {
            "provider": "ase",
            "method": "md",
            "parameters": {
                "ensemble": "nvt",
                "temp": 300.0,
                "timestep": 0.1,
                "steps": 2,
                "dump_period": 1,
                "remove_rotation": False,
                "controller": {"name": "berendsen", "params": {"Tdamp": 10.0}},
            },
        },
        "dispatch": {"worker": "single"},
    }


def _seeds():
    return [
        Atoms("Cu2", positions=[[0, 0, 0], [1.8, 0, 0]], cell=[8, 8, 8], pbc=True),
        Atoms("Cu2", positions=[[0, 0, 0], [2.4, 0, 0]], cell=[8, 8, 8], pbc=True),
    ]


@pytest.mark.parametrize(
    ("strategy", "message"),
    [
        ({"centers": []}, "nonempty"),
        ({"centers": [1.0, 1.0]}, "unique"),
        ({"kspring": 0.0}, "positive"),
        ({"replicas": 0}, "positive integer"),
        ({"equilibration_steps": 0}, "positive integer"),
    ],
)
def test_umbrella_configuration_validation(strategy, message):
    with pytest.raises(ValueError, match=message):
        create_exploration(_config(_seeds(), **strategy))


def test_umbrella_sampling_runs_window_replica_grid_and_resumes(tmp_path):
    exploration = create_exploration(_config(_seeds()))
    exploration.directory = tmp_path
    exploration.register_worker(create_worker(_runtime(), print_func=lambda _: None))

    exploration.run()

    assert exploration.read_convergence()
    manifest = json.loads((tmp_path / "windows.json").read_text())
    assert [(item["window"], item["replica"]) for item in manifest["records"]] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert [item["seed_index"] for item in manifest["records"]] == [0, 0, 1, 1]
    random_seeds = {
        (item["equilibration_velocity_seed"], item["equilibration_random_seed"])
        for item in manifest["records"]
    }
    assert len(random_seeds) == 4
    workers = exploration.get_workers()
    assert len(workers) == 4
    assert [worker.runtime.config.modifiers[-1].parameters["center"] for worker in workers] == [
        1.9,
        1.9,
        2.3,
        2.3,
    ]
    results = [worker.retrieve(include_retrieved=True) for worker in workers]
    first_frames = [trajectories[0][0] for trajectories in results]
    assert [frame.info["umbrella_center"] for frame in first_frames] == [1.9, 1.9, 2.3, 2.3]

    archive = tmp_path / "production.h5"
    AtomsNDArray(first_frames).save_file(archive)
    restored = AtomsNDArray.from_file(archive)
    assert [restored[index].info["umbrella_center"] for index in range(4)] == [
        1.9,
        1.9,
        2.3,
        2.3,
    ]
    assert [restored[index].info["umbrella_replica"] for index in range(4)] == [0, 1, 0, 1]
    assert (tmp_path / "samples.csv").read_text().count("\n") == 1 + 4 * 3

    before = (tmp_path / "windows.json").read_text()
    exploration.run()
    assert (tmp_path / "windows.json").read_text() == before

    changed = create_exploration(_config(_seeds(), centers=[1.9, 2.4]))
    changed.directory = tmp_path
    changed.register_worker(create_worker(_runtime(), print_func=lambda _: None))
    with pytest.raises(ValueError, match="configuration changed"):
        changed.run()


def test_group_structures_by_umbrella_center():
    frames = []
    for center in (1.0, 1.0, 1.5, 1.5):
        atoms = Atoms("H")
        atoms.info["umbrella_center"] = center
        frames.append(atoms)
    data = AtomsNDArray([frames[:2], frames[2:]])

    groups = group_structures_by_info(data, "umbrella_center")

    assert groups == {1.0: [[0, 0], [0, 1]], 1.5: [[1, 0], [1, 1]]}


def test_property_selection_is_applied_per_umbrella_center(tmp_path):
    frames = []
    for center, deviations in ((1.0, (0.1, 0.2)), (1.5, (0.3, 0.4))):
        for deviation in deviations:
            atoms = Atoms("H")
            atoms.info.update(umbrella_center=center, max_devi_f=deviation)
            atoms.calc = SinglePointCalculator(atoms, energy=0.0, forces=np.zeros((1, 3)))
            frames.append(atoms)
    selector = create_selector(
        {
            "method": "property",
            "group_by": "info umbrella_center",
            "name": "max_devi_f",
            "sparsify": {"method": "hist", "range": [0.0, 1.0], "nbins": 4},
            "number": [1, 1.0],
            "random_seed": 7,
            "directory": tmp_path,
        }
    )

    selected = selector.select(AtomsNDArray(frames))

    assert len(selected) == 2
    assert {atoms.info["umbrella_center"] for atoms in selected} == {1.0, 1.5}


def test_umbrella_runtime_must_be_unbiased_md():
    exploration = create_exploration(_config(_seeds()))
    runtime = _runtime()
    runtime["modifiers"] = [
        {
            "provider": "builtin",
            "method": "distance_harmonic",
            "parameters": {"group": [0, 1], "center": 2.0, "kspring": 1.0},
        }
    ]

    with pytest.raises(ValueError, match="unbiased"):
        exploration.register_worker(create_worker(runtime, print_func=lambda _: None))
