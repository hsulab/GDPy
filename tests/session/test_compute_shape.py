from types import SimpleNamespace

import numpy as np
from ase import Atoms

from gdpx.analysis.selectors.clustering import group_structures
from gdpx.data.array import AtomsNDArray
from gdpx.workflow.nodes.driver import (
    _is_single_point_driver,
    convert_results_to_structures,
)


def test_single_point_driver_detection_supports_current_and_legacy_tasks():
    spc = SimpleNamespace(setting=SimpleNamespace(task="spc", steps=-1))
    legacy_spc = SimpleNamespace(setting=SimpleNamespace(task="min", steps=0))
    dynamics = SimpleNamespace(setting=SimpleNamespace(task="md", steps=100))

    assert _is_single_point_driver(spc)
    assert _is_single_point_driver(legacy_spc)
    assert not _is_single_point_driver(dynamics)


def test_single_point_results_preserve_input_shape_and_axis_groups():
    trajectories = []
    for temperature, nframes in zip((400, 500, 600, 700), (3, 2, 1, 2)):
        trajectory = [
            Atoms("H", info={"temperature": temperature, "frame": frame})
            for frame in range(nframes)
        ]
        trajectories.append([trajectory])
    inputs = AtomsNDArray(trajectories)
    input_markers = np.argwhere(inputs.markers)

    flat_results = [frame.copy() for frame in inputs.get_marked_structures()]
    worker_results = AtomsNDArray([[[frame] for frame in flat_results]])

    converted = convert_results_to_structures(
        worker_results,
        inputs.shape,
        input_markers,
    )

    assert converted.shape == (4, 1, 3)
    groups = group_structures(converted, group_by="axis 0")
    assert list(groups) == [0, 1, 2, 3]
    assert [len(markers) for markers in groups.values()] == [3, 2, 1, 2]
