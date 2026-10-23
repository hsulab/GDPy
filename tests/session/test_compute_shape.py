from types import SimpleNamespace

import numpy as np
import pytest
from ase.io.jsonio import decode
from ase import Atoms

from gdpx.analysis.selectors.clustering import group_structures
from gdpx.data.array import AtomsNDArray
from gdpx.workflow.nodes.driver import (
    _is_single_point_driver,
    compute,
    convert_results_to_structures,
    extract,
    get_input_layout,
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
    assert not (tmp_path / "_shape").exists()
    assert worker.input_layout["shape"] == [2]
    assert worker.input_layout["selected_indices"] == [[0], [1]]


def test_single_point_layout_is_frozen_and_used_by_extract(tmp_path):
    from gdpx.execution.factory import create_worker

    frames = [Atoms("Cu", positions=[[x, 0, 0]], cell=[4, 4, 4], pbc=True)
              for x in (0.1, 0.2, 0.3)]
    structures = AtomsNDArray([[[frames[0], frames[1]]], [[None, frames[2]]]],
                             dims=("temperature", "structure", "frame"),
                             coords={"temperature": [400, 500]})
    structures.markers = [[0, 0, 0], [1, 0, 1]]
    runtime = RuntimeVariable(PotentialVariable("emt"), ExecutorVariable("ase", "spc"))
    operation = compute(runtime, structures=Variable(structures), directory=tmp_path / "compute")
    result = operation.forward(structures, runtime.value)

    inputs_path = operation.directory / "_meta" / "manifest.json"
    before = inputs_path.read_bytes()
    record = decode(before.decode())["workers"]["."]
    assert record["layout"] == {
        "version": 1, "shape": [2, 1, 2], "dims": ["temperature", "structure", "frame"],
        "coords": {"temperature": [400, 500]}, "selected_indices": [[0, 0, 0], [1, 0, 1]],
    }
    assert not (operation.directory / "_shape").exists()
    assert result.shape == structures.shape
    assert result.dims == structures.dims
    np.testing.assert_array_equal(result.coords["temperature"], [400, 500])
    np.testing.assert_array_equal(result.markers, structures.markers)

    # A separate extraction operation needs only the saved manifest and results.
    worker = create_worker(runtime.value, directory=operation.directory)
    extraction = extract(operation, directory=tmp_path / "extract")
    restored = extraction.forward([worker])
    assert restored.shape == result.shape
    assert restored.dims == result.dims
    np.testing.assert_array_equal(restored.markers, result.markers)
    np.testing.assert_array_equal(restored.coords["temperature"], [400, 500])
    # Resume with completed calculations also reads the frozen layout.
    resumed = operation.forward(structures, runtime.value)
    assert resumed.shape == result.shape
    assert resumed.dims == result.dims
    assert inputs_path.read_bytes() == before


def test_extract_layout_falls_back_to_legacy_shape_files(tmp_path):
    legacy = tmp_path / "_shape"
    legacy.mkdir()
    np.savetxt(legacy / "shape.dat", [4, 1, 3], fmt="%d")
    np.savetxt(legacy / "markers.dat", [[0, 0, 1], [3, 0, 2]], fmt="%d")
    shape, markers, layout = get_input_layout(tmp_path)
    np.testing.assert_array_equal(shape, [4, 1, 3])
    np.testing.assert_array_equal(markers, [[0, 0, 1], [3, 0, 2]])
    assert layout is None


def test_changed_layout_does_not_modify_frozen_inputs(tmp_path):
    from gdpx.execution.factory import create_worker

    runtime = RuntimeVariable(
        PotentialVariable("emt"), ExecutorVariable("ase", "spc"),
        scheduler={"provider": "slurm", "parameters": {"is_dry_run": True}},
    )
    frame = Atoms("Cu", cell=[4, 4, 4], pbc=True)
    worker = create_worker(runtime.value, directory=tmp_path)
    worker.input_layout = {"version": 1, "shape": [2], "dims": ["frame"],
                           "coords": {}, "selected_indices": [[0]]}
    worker.prepare_batches([frame])
    before = worker.metadata.inputs.path.read_bytes()
    worker.input_layout = {**worker.input_layout, "selected_indices": [[1]]}
    with pytest.raises(ValueError, match="layout changed"):
        worker.prepare_batches([frame])
    assert worker.metadata.inputs.path.read_bytes() == before

    worker.input_layout = {**worker.input_layout, "selected_indices": [[3]]}
    with pytest.raises(ValueError, match="Invalid input layout"):
        worker.prepare_batches([frame])
    assert worker.metadata.inputs.path.read_bytes() == before
