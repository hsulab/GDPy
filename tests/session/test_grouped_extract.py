import pytest
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from gdpx.data.array import AtomsNDArray
from gdpx.analysis.selectors.locate import LocateSelector
from gdpx.execution.lifecycle import create_runtime_workers
from gdpx.workflow.nodes.driver import (
    compute, extract_results_from_workers, extract_results_from_workers_compact,
    convert_results_to_structures,
)
from gdpx.workflow.session.variable import Variable


def config(batch_size=4):
    return {
        "potential": {"provider": "emt"},
        "executor": {"provider": "ase", "method": "md",
                     "parameters": {"ensemble": "nvt", "steps": 2,
                                    "dump_period": 1, "random_seed": 17},
                     "broadcast": {"temp": [400, 500, 600, 700]}},
        "dispatch": {"worker": "batch", "batch_size": batch_size},
    }


def image(**info):
    atoms = Atoms("H", info=info)
    atoms.calc = SinglePointCalculator(atoms, energy=0., forces=np.zeros((1, 3)))
    return atoms


@pytest.mark.parametrize("extractor", [extract_results_from_workers, extract_results_from_workers_compact])
def test_extraction_restores_variant_axis_from_physical_cache(tmp_path, extractor):
    worker = create_runtime_workers(config())[0]
    worker.directory = tmp_path / "simulation"
    raw = [[image(confid=4 * si + ti, step=fi)
            for fi in range(2 + si)]
           for si in range(2) for ti in range(4)]
    worker.retrieve = lambda **kwargs: raw
    for cached in [False, True]:
        if cached:
            worker.retrieve = lambda **kwargs: pytest.fail("Cache should be reused")
        status, extracted = extractor(tmp_path / "extracted", [worker], safe_inspect=False)
        result = convert_results_to_structures(extracted, None, None, workers=[worker])
        assert status == "finished"
        assert result.shape == (4, 2, 3)
        assert result.dims == ("temperature", "structure", "frame")
        assert result.coords["temperature"].tolist() == [400, 500, 600, 700]
        assert {a.info["confid"] % 4 for row in result[1] for a in row if a is not None} == {1}
        selected = result.sel(temperature=500)
        assert {a.info["confid"] // 4 for a in selected.get_marked_structures()} == {0, 1}
        assert len(selected.get_marked_structures()) == 5
        selector = LocateSelector(group_by=0, indices="1")
        selector._mark_structures(result)
        assert {a.info["confid"] % 4 for a in result.get_marked_structures()} == {1}
        assert len(result.get_marked_structures()) == 5


def test_grouped_compute_and_single_point_preserve_temperature_indices(tmp_path):
    worker = create_runtime_workers(config())[0]
    frames = [Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[8, 8, 8], pbc=True),
              Atoms("Cu2", positions=[[0, 0, 0], [2.6, 0, 0]], cell=[8, 8, 8], pbc=True)]
    runtimes = worker.runtimes
    node = compute(Variable(runtimes), structures=Variable(frames),
                   use_archive=False, directory=tmp_path / "md")
    result = node.forward(frames, runtimes)
    assert result.shape == (4, 2, 3)
    assert result.coords["temperature"].tolist() == [400, 500, 600, 700]
    assert all(frame.get_temperature() == pytest.approx(500, rel=1e-5)
               for frame in [row[0] for row in result[1]])
    spc = create_runtime_workers({"potential": {"provider": "emt"},
                                 "executor": {"provider": "ase", "method": "spc"}})[0].runtime
    spc_node = compute(Variable(spc), structures=Variable(result),
                       use_archive=False, directory=tmp_path / "spc")
    evaluated = spc_node.forward(result, spc)
    assert evaluated.shape == result.shape
    assert all(frame.get_potential_energy() is not None for row in evaluated[1] for frame in row)


def test_single_variant_reduces_but_merge_is_explicit(tmp_path):
    worker = create_runtime_workers(config())[0]
    raw = [[image()] for _ in range(8)]
    worker.directory = tmp_path / "md"
    worker.retrieve = lambda **kwargs: raw
    _, data = extract_results_from_workers(tmp_path / "extracted", [worker], safe_inspect=False)
    merged = convert_results_to_structures(data, None, None, merge_workers=True, workers=[worker])
    assert merged.shape == (8, 1)
    single = convert_results_to_structures(AtomsNDArray([raw]), None, None)
    assert single.shape == (8, 1)
