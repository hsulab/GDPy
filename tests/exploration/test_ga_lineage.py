"""GA ancestry must survive repeated relaxation and in-place mutation records."""
from ase import Atoms
from ase.db import connect
from PIL import Image

from gdpx.exploration.genetic_algorithm.lineage import collect_lineage, plot_lineage


def test_original_parents_survive_mutation_and_relaxation(tmp_path):
    db = connect(tmp_path / 'candidates.db')
    atoms = Atoms('Cu')
    for identifier in (1, 2):
        db.write(atoms, confid=identifier, relaxed=0, generation=0,
                 origin='InitialBuilder:random', data={'builder': 'random'})
        db.write(atoms, confid=identifier, relaxed=1, generation=0,
                 target=-float(identifier), extinct=0)
    db.write(atoms, confid=3, relaxed=0, generation=1,
             origin='ClusterCutAndSpliceCrossover', data={'parents': [1, 2]})
    db.write(atoms, confid=3, relaxed=0, mutation=1,
             origin='RattleMutation', data={'parents': [3]})
    for _ in range(2):
        db.write(atoms, confid=3, relaxed=1, generation=1, target=-3., extinct=1)
    db.write(Atoms(), confid=3, queued=1, relaxed=0)
    db.write(atoms, confid=4, relaxed=0, generation=2,
             origin='MutationCandidateUnrelaxed', mutation=1, data={'parents': [2]})
    db.write(atoms, confid=5, relaxed=0, generation=2,
             origin='CompletionBuilder:fresh', data={'builder': 'fresh'})
    db.write(atoms, confid=6, relaxed=0, generation=2,
             origin='Parthenogenesis', data={'parents': [2, 2]})
    before = db.count()
    nodes = collect_lineage(db)
    assert len(nodes) == 6
    assert nodes[3]['parents'] == [1, 2]
    assert nodes[3]['generation'] == 1
    assert nodes[3]['operation'] == 'crossover + mutation'
    assert nodes[3]['target'] == -3. and nodes[3]['extinct']
    assert nodes[4]['parents'] == [2] and nodes[4]['operation'] == 'mutation'
    assert not nodes[4]['evaluated']
    assert nodes[5]['parents'] == [] and nodes[5]['operation'] == 'builder'
    assert nodes[6]['parents'] == [2]
    paths = plot_lineage(db, tmp_path, 'formation_energy')
    assert [path.name for path in paths] == ['family_tree.png']
    with Image.open(paths[0]) as image:
        assert image.size == (1200, 600)
    assert not paths[0].with_suffix('.svg').exists()
    assert db.count() == before and collect_lineage(db) == nodes


def test_empty_and_incomplete_ancestry(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    original = Figure.savefig
    def check(figure, *args, **kwargs):
        assert not figure.texts and not figure.legends
        assert not figure.axes[0].patches
        return original(figure, *args, **kwargs)
    monkeypatch.setattr(Figure, 'savefig', check)
    db = connect(tmp_path / 'candidates.db')
    assert plot_lineage(db, tmp_path) == []
    db.write(Atoms('Cu'), confid=2, relaxed=1, generation=5,
             data={'parents': [99]})
    paths = plot_lineage(db, tmp_path)
    assert len(paths) == 1 and paths[0].exists()


def test_production_size_keeps_all_nodes_and_readable_ids(tmp_path, monkeypatch):
    import gdpx.exploration.genetic_algorithm.lineage as module
    from matplotlib.figure import Figure
    nodes = {}
    for generation in range(11):
        for index in range(20):
            identifier = generation * 20 + index + 1
            parents = [] if generation == 0 else [(generation - 1) * 20 + index + 1,
                                                  (generation - 1) * 20 + (index + 7) % 20 + 1]
            nodes[identifier] = dict(generation=generation, parents=parents,
                                     operation='crossover' if generation else 'builder',
                                     evaluated=True, extinct=False, target=-generation-index/20)
    monkeypatch.setattr(module, 'collect_lineage', lambda connection: nodes)
    original = Figure.savefig
    def check(figure, *args, **kwargs):
        assert not figure.texts and not figure.legends
        assert figure.axes[0].get_title() == ''
        assert [tick.get_text() for tick in figure.axes[0].get_yticklabels()] == [str(i) for i in range(11)]
        assert figure.axes[1].get_ylabel() == 'energy [eV]'
        assert figure.axes[1]._colorbar.mappable.cmap.name == 'coolwarm'
        labels = figure.axes[0].texts
        assert len(labels) == 220
        assert all(label.get_fontsize() >= 6 for label in labels)
        assert len(figure.axes[0].collections) == 220
        assert len(figure.axes[0].patches) == 400
        return original(figure, *args, **kwargs)
    monkeypatch.setattr(Figure, 'savefig', check)
    paths = module.plot_lineage(None, tmp_path)
    with Image.open(paths[0]) as image:
        assert image.size == (1200, 600)
