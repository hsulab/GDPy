from types import SimpleNamespace

import numpy as np
from ase import Atoms

from gdpx.analysis.selectors.clustering import group_structures
from gdpx.data.array import AtomsNDArray
from gdpx.workflow.nodes.driver import (
    _is_single_point_driver,
    compute,
    convert_results_to_structures,
)
from gdpx.workflow.nodes.runtime import ExecutorVariable, PotentialVariable, RuntimeVariable
from gdpx.workflow.session.variable import Variable


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


def test_compute_preserves_runtime_dispatch_batch_size(tmp_path, monkeypatch):
    class PendingWorker:
        def __init__(self):
            self.driver = SimpleNamespace(
                setting=SimpleNamespace(task="spc", steps=-1)
            )
            self.batchsize = 24
            self.directory = tmp_path
            self._share_wdir = False
            self._retain_info = False

        def run(self, frames):
            self.frames = frames

        def inspect(self, resubmit=False):
            self.resubmit = resubmit

        def get_number_of_running_jobs(self):
            return 1

    runtime_variable = RuntimeVariable(
        PotentialVariable("emt"),
        ExecutorVariable("ase", "spc"),
        dispatch={"batch_size": 24},
    )
    structures = AtomsNDArray([Atoms("H"), Atoms("H")])
    worker = PendingWorker()
    monkeypatch.setattr(
        "gdpx.workflow.nodes.driver.create_worker",
        lambda runtime: worker,
    )
    operation = compute(
        runtime_variable,
        structures=Variable(structures),
        directory=tmp_path,
    )

    operation.forward(structures, runtime_variable.value)

    assert worker.batchsize == 24
    assert len(worker.frames) == 2
