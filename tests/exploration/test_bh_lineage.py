import json
from types import SimpleNamespace

from gdpx.exploration.basin_hopping.lineage import collect_lineage, plot_lineage
from gdpx.exploration.checkpoint import save_data


class Row(dict):
    def __getattr__(self, key):
        return self[key]


def test_lineage_preserves_rejected_extinct_and_restart(tmp_path):
    rounds = tmp_path / 'tmp_folder/gen1/rounds'
    rounds.mkdir(parents=True)
    events = [dict(step=0, decisions=[0], candidates=[1]),
              dict(step=1, decisions=[1], candidates=[1]),
              dict(step=2, decisions=[4], candidates=[2])]
    content = ''.join(json.dumps(event) + '\n' for event in events).encode()
    (rounds / 'events.jsonl').write_bytes(content + b'{uncommitted tail')
    save_data(rounds / 'final.json', dict(version=5, context=dict(journal_offset=len(content))))
    rows = [Row(confid=1, generation=0, energy=-1, data={}),
            Row(confid=2, generation=1, energy=-2, data=dict(round=1, chain=0, parents=[1], accepted=False)),
            Row(confid=3, generation=1, energy=-3, extinct=1, data=dict(round=2, chain=0, parents=[1], accepted=True)),
            Row(confid=4, generation=1, energy=-4, data=dict(round=3, chain=0, parents=[2], accepted=True))]
    connection = SimpleNamespace(select=lambda **kwargs: rows)
    nodes, restarts = collect_lineage(connection, tmp_path)
    assert set(nodes) == {1, 2, 3}
    assert nodes[2]['parents'] == [1] and not nodes[2]['accepted']
    assert nodes[3]['extinct']
    assert restarts == [(3, 2)]
    paths = plot_lineage(connection, tmp_path)
    assert [p.name for p in paths] == ['gen0001.png']
    path = paths[-1]
    assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert not path.with_suffix('.svg').exists()


def test_empty_lineage(tmp_path):
    connection = SimpleNamespace(select=lambda **kwargs: [])
    assert plot_lineage(connection, tmp_path) == []


def test_dense_generation_is_compact_and_has_no_text(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    from gdpx.exploration.basin_hopping.lineage import _plot_generation
    from PIL import Image

    nodes = {}
    for chain in range(20):
        for step in range(1, 51):
            identifier = chain * 50 + step
            nodes[identifier] = dict(generation=1, step=step, chain=chain,
                                     parents=[identifier - 1] if step > 1 else [],
                                     accepted=True, extinct=False, energy=-step)
    original = Figure.savefig
    def check(figure, *args, **kwargs):
        assert len(figure.axes) == 2
        ax = figure.axes[0]
        assert ax.axison
        assert ax.get_xlabel() == 'round'
        assert figure.axes[1].get_ylabel() == 'energy [eV]'
        assert not any(text.get_text() for text in ax.texts)
        assert all(text.arrow_patch.get_edgecolor()[:3] != (.65, .65, .65)
                   for text in ax.texts if text.arrow_patch is not None)
        return original(figure, *args, **kwargs)
    monkeypatch.setattr(Figure, 'savefig', check)
    path = tmp_path / 'dense.png'
    _plot_generation(nodes, [], 1, path)
    with Image.open(path) as image:
        assert image.size == (1200, 600)


def test_markers_and_ids_grow_to_fit_available_space():
    import matplotlib.pyplot as plt
    from gdpx.exploration.basin_hopping.lineage import _node_style
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    try:
        ax.set(xlim=(-1, 10), ylim=(-1, 10))
        fig.canvas.draw()
        diameter, font_size = _node_style({1: (1, 1), 2: (8, 8)}, ax)
        assert diameter == 24 and font_size == 18
        crowded_diameter, crowded_font = _node_style({1000: (1, 1), 1001: (1.2, 1)}, ax)
        assert crowded_diameter < diameter and crowded_font is None
    finally:
        plt.close(fig)


def test_generation_figures_include_initial_and_saved_population(tmp_path, monkeypatch):
    import gdpx.exploration.basin_hopping.lineage as module
    nodes = {1: dict(generation=0), 2: dict(generation=0),
             3: dict(generation=1), 4: dict(generation=1), 5: dict(generation=2)}
    monkeypatch.setattr(module, 'collect_lineage', lambda *args: (nodes, []))
    calls = []
    monkeypatch.setattr(module, '_plot_generation',
                        lambda nodes, links, generation, path, population: calls.append((generation, population)))
    connection = SimpleNamespace(metadata={'generation_plans': {'2': {'parents': [3], 'population': [3, 4]}}})
    paths = module.plot_lineage(connection, tmp_path)
    assert [path.name for path in paths] == ['gen0001.png', 'gen0002.png']
    assert calls == [(1, [1, 2]), (2, [3, 4])]
    calls.clear()
    connection.metadata['generation_plans']['2'].pop('population')
    module.plot_lineage(connection, tmp_path)
    assert calls[-1] == (2, [3])
