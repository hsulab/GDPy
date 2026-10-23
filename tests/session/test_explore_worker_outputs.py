"""Typed exploration outputs keep MD and minimisation selection independent."""
from types import SimpleNamespace
from ase import Atoms

import pytest

from gdpx.workflow.nodes.exploration import explore
from gdpx.workflow.session.variable import Variable


@pytest.mark.parametrize("pending", [False, True])
def test_explore_separates_worker_methods(tmp_path, monkeypatch, pending):
    def worker(method):
        return SimpleNamespace(runtime=SimpleNamespace(config=SimpleNamespace(
            executor=SimpleNamespace(method=method))))
    md = worker("md")
    initial, minimum = worker("min"), worker("min")
    other = worker("spc")
    workers = [initial, md, minimum, other]
    class Manager:
        def __init__(self, *args):
            pass
        def run(self):
            pass
        def inspect(self, **kwargs):
            pass
        def get_number_of_running_jobs(self):
            return int(pending)
        def retrieve(self, **kwargs):
            return workers
    monkeypatch.setattr('gdpx.workflow.nodes.exploration.ExplorationBasedWorker', Manager)
    node = explore(Variable(None), directory=tmp_path)
    candidates = [Atoms("Cu"), Atoms("Cu")]
    outputs = node.forward(SimpleNamespace(get_minimised_candidates=lambda: candidates), None, None)
    if pending:
        assert outputs['minimised_candidates'] == outputs['dynamics'] == outputs['minimisation'] == []
    else:
        assert outputs['minimised_candidates'] == candidates
        assert node.default_output == 'minimised_candidates'
        assert outputs['dynamics'] == [md]
        assert outputs['minimisation'] == [initial, minimum]
        assert set(outputs) == set(node.output_names)


def test_example_uses_separate_selection_and_dataset_branches():
    from pathlib import Path
    from gdpx.workflow.configuration import load_workflow, OutputReference
    from gdpx.workflow.compiler import validate_workflow
    from gdpx.workflow.nodes import data, dataset, driver, selector, trainer  # Register example nodes.
    source = Path(__file__).resolve().parents[2] / 'examples/workflows/active-learning-dpa4c/sa.yaml'
    spec = load_workflow(source)
    validate_workflow(spec)
    assert spec.steps['md_trajectories'].inputs['compute'] == OutputReference('sample', 'dynamics')
    assert spec.steps['min_trajectories'].inputs['compute'] == OutputReference('sample', 'minimisation')
    for suffix in ('md', 'min'):
        assert spec.steps[f'select_{suffix}'].inputs['structures'] == f'score_{suffix}'
        assert spec.steps[f'label_{suffix}'].inputs['structures'] == f'select_{suffix}'
    assert spec.steps['collect_min'].inputs['dataset'] == OutputReference('collect_md', 'dataset')
    assert spec.state['training_data'].update == OutputReference('collect_min', 'dataset')


def test_sa_branches_extract_select_and_label_independently(tmp_path):
    from pathlib import Path
    import copy
    from ase import Atoms
    from ase.io import write
    from gdpx.workflow.configuration import load_workflow
    from gdpx.workflow.nodes.exploration import ExplorationVariable
    from gdpx.workflow.nodes.runtime import RuntimeVariable
    from gdpx.workflow.nodes.driver import extract, compute
    from gdpx.workflow.nodes.selector import SelectorVariable, select
    from gdpx.workflow.state import OutputSelector, TargetBarrier
    from gdpx.workflow.session.once import OnceSession

    source = Path(__file__).resolve().parents[2] / 'examples/workflows/active-learning-dpa4c/sa.yaml'
    spec = load_workflow(source)
    seed = tmp_path / 'seed.xyz'
    write(seed, Atoms('Cu2', positions=[[4, 5, 5], [6.4, 5, 5]], cell=[10]*3, pbc=True))
    search = copy.deepcopy(dict(spec.resources['search'].options))
    search['system']['builders']['init']['fname'] = str(seed)
    search['strategy']['minimisation']['runtime']['executor']['parameters']['stop']['steps'] = 20
    md = copy.deepcopy(dict(spec.resources['md'].options))
    md['parameters']['stop']['steps'] = 2
    md['parameters']['output']['trajectory']['period'] = 1
    sampling = RuntimeVariable(potential={'provider': 'emt'}, executor=md)
    labeling = RuntimeVariable(potential={'provider': 'emt'}, executor={'provider': 'ase', 'method': 'spc'})
    sample = explore(ExplorationVariable(**search), runtime=sampling, directory=tmp_path / 'sample')
    branches = []
    for method, port in [('md', 'dynamics'), ('min', 'minimisation')]:
        frames = extract(OutputSelector(sample, port), merge_workers=True, use_archive=False,
                         directory=tmp_path / f'extract_{method}')
        picked = select(frames, SelectorVariable(selection={'method': 'locate', 'indices': ':2'}),
                        directory=tmp_path / f'select_{method}')
        labels = compute(labeling, structures=picked, use_archive=False,
                         directory=tmp_path / f'label_{method}')
        branches.append((method, frames, picked, labels))
    session = OnceSession(directory=tmp_path / 'run')
    default_selection = select(sample, SelectorVariable(selection={'method': 'locate', 'indices': '-1'}),
                               directory=tmp_path / 'select_minima')
    session.run(TargetBarrier([branch[-1] for branch in branches] + [default_selection]))
    assert session.is_finished()
    for method, frames, picked, labels in branches:
        assert {worker.runtime.config.executor.method for worker in frames.workers} == {method}
        expected = sum(len(trajectory) for worker in frames.workers
                       for trajectory in worker.retrieve(include_retrieved=True, use_archive=False))
        assert len(frames.output.get_marked_structures()) == expected
        assert len(picked.output.get_marked_structures()) == 2
        assert len(labels.output.get_marked_structures()) == 2
    assert len(branches[0][1].workers) == 1
    assert len(branches[1][1].workers) == 2  # Initial and post-MD minimisations.

    # The default head is final structures from the candidate DB, not workers
    # or the comparator-filtered retained population.
    from gdpx.exploration.persist.database import GlobalOptimisationDatabase
    candidates = sample.output['minimised_candidates']
    engine = sample.input_nodes[0].value
    rows = list(GlobalOptimisationDatabase(engine.directory / 'candidates.db').connection.select(relaxed=1))
    assert len(candidates) == len(rows)
    assert {a.info['confid'] for a in candidates} == {row.confid for row in rows}
    assert all(isinstance(a, Atoms) and a.calc is not None for a in candidates)
    assert len(candidates) > len(engine.population.candidates)
    assert len(default_selection.output.get_marked_structures()) == 1
