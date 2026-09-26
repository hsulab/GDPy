"""Plot BH ancestry and restart links using metadata, without loading Atoms."""
import json
from pathlib import Path

import numpy as np

from ..checkpoint import load_data


def collect_lineage(connection, directory):
    """Return candidate metadata and restart links from committed BH rounds."""
    histories = {}
    for rounds in sorted((Path(directory) / 'tmp_folder').glob('gen*/rounds')):
        generation = int(rounds.parent.name[3:])
        if (rounds / 'final.json').exists():
            state = load_data(rounds / 'final.json')
        elif (rounds / 'current.json').exists():
            manifest = json.loads((rounds / 'current.json').read_text())
            state = load_data(rounds / manifest['snapshots'][0] / 'state.json')
        else:
            continue
        if state.get('version') != 5:
            raise ValueError('Unsupported BH event history version.')
        offset = state['context']['journal_offset']
        with (rounds / 'events.jsonl').open('rb') as stream:
            content = stream.read(offset)
        if len(content) != offset:
            raise ValueError('Incomplete BH event journal.')
        histories[generation] = [json.loads(line) for line in content.splitlines()]

    committed_steps = {generation: max((event['step'] for event in events), default=0)
                       for generation, events in histories.items()}
    nodes = {}
    for row in connection.select(relaxed=1):
        generation = row.get('generation', 0)
        data = row.data
        step = data.get('round', 0)
        if generation and (generation not in histories or
                           step > committed_steps[generation]):
            continue
        nodes[row.confid] = dict(id=row.confid, generation=generation, step=step,
                                chain=data.get('chain', 0), parents=list(data.get('parents', [])),
                                accepted=bool(data.get('accepted', True)),
                                extinct=bool(row.get('extinct', 0)), energy=row.get('energy', np.nan))
    trials = {(node['generation'], node['step'], node['chain']): identifier
              for identifier, node in nodes.items() if node['generation']}
    restarts = []
    for generation, events in histories.items():
        for event in events:
            if not event['step']:
                continue
            for chain, decision in enumerate(event['decisions']):
                if decision == 4:
                    source = trials.get((generation, event['step'], chain))
                    target = event['candidates'][chain]
                    if source in nodes and target in nodes:
                        restarts.append((source, target))
    return nodes, restarts


def _boxes_fit(boxes, bounds):
    """Check marker/label envelopes in output pixels, including frame clearance."""
    boxes = sorted(boxes)
    for index, (left, bottom, right, top) in enumerate(boxes):
        if left < bounds.x0 or right > bounds.x1 or bottom < bounds.y0 or top > bounds.y1:
            return False
        for other_left, other_bottom, _, other_top in boxes[index + 1:]:
            if other_left >= right:
                break
            if other_bottom < top and other_top > bottom:
                return False
    return True


def _node_style(positions, ax):
    """Choose the largest readable marker/ID pair fitting the actual raster."""
    from matplotlib.font_manager import FontProperties

    renderer = ax.figure.canvas.get_renderer()
    scale = ax.figure.dpi / 72
    points = [(str(identifier), *ax.transData.transform(position))
              for identifier, position in positions.items()]
    for font_size in range(18, 7, -1):
        diameter = font_size + 6
        radius = diameter * scale / 2
        boxes = []
        font = FontProperties(size=font_size)
        for label, x, y in points:
            width, height, descent = renderer.get_text_width_height_descent(label, font, False)
            half_width = max(radius, width / 2) + 3
            boxes.append((x - half_width, y - radius - 4 * scale - max(height + descent, font_size * scale),
                          x + half_width, y + radius + 3))
        if _boxes_fit(boxes, ax.bbox):
            return diameter, font_size
    for diameter in range(18, 2, -1):
        radius = diameter * scale / 2 + 3
        boxes = [(x - radius, y - radius, x + radius, y + radius) for _, x, y in points]
        if _boxes_fit(boxes, ax.bbox):
            return diameter, None
    return 3, None


def _plot_generation(nodes, restarts, generation, path, population=()):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    current = {identifier for identifier, node in nodes.items() if node['generation'] == generation}
    links = [(source, target) for source, target in restarts if source in current]
    sources = {parent for identifier in current for parent in nodes[identifier]['parents'] if parent in nodes}
    sources.update(target for _, target in links)
    sources.update(identifier for identifier in population if identifier in nodes)
    selected = current | sources
    roots = sorted(selected - current if generation else current)
    chains = max((nodes[i]['chain'] for i in current), default=0) + 1
    height = max(chains - 1, 1)
    positions = {identifier: (0, y) for identifier, y in
                 zip(roots, np.linspace(0, height, len(roots)))}
    for identifier in sorted(current - set(roots)):
        node = nodes[identifier]
        positions[identifier] = (node['step'], node['chain'] + (0 if node['accepted'] else .28))
    # Fixed raster size keeps production reports small regardless of candidate count.
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    fig.subplots_adjust(left=.04, right=.90, bottom=.12, top=.96)
    try:
        ax.set_xlim(-.6, max((pos[0] for pos in positions.values()), default=0) + .6)
        ax.set_ylim(-.6, height + .7)
        ax.set_xlabel('round', color='#123b68')
        ax.set_yticks([])
        last_round = max((nodes[i]['step'] for i in current), default=0)
        stride = max(1, int(np.ceil(last_round / 10)))
        ticks = sorted({0, last_round, *range(stride, last_round + 1, stride)})
        ax.set_xticks(ticks, [('initial' if generation <= 1 else 'population')
                             if tick == 0 else str(tick) for tick in ticks])
        ax.tick_params(axis='x', colors='#123b68')
        for spine in ax.spines.values():
            spine.set_color('#123b68')
        energies = [nodes[i]['energy'] for i in selected if np.isfinite(nodes[i]['energy'])]
        norm = Normalize(min(energies), max(energies)) if energies else Normalize(0, 1)
        cmap = plt.get_cmap('coolwarm')
        colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=.025, fraction=.04)
        colorbar.set_label('energy [eV]', color='#123b68')
        colorbar.ax.tick_params(colors='#123b68')
        colorbar.outline.set_edgecolor('#123b68')
        if not energies:
            colorbar.set_ticks([])
        fig.canvas.draw()
        diameter, font_size = _node_style(positions, ax)
        size = diameter ** 2
        shrink = diameter / 2 + 1
        for identifier in sorted(current):
            for parent in nodes[identifier]['parents']:
                if parent in positions:
                    ax.annotate('', xy=positions[identifier], xytext=positions[parent],
                                arrowprops=dict(arrowstyle='->', color='#123b68', lw=.8,
                                                shrinkA=shrink, shrinkB=shrink, mutation_scale=7), zorder=1)
        for source, target in links:
            ax.annotate('', xy=positions[target], xytext=positions[source],
                        arrowprops=dict(arrowstyle='->', color='#8e24aa', linestyle='--', lw=1.1,
                                        connectionstyle='arc3,rad=.2', shrinkA=shrink, shrinkB=shrink,
                                        mutation_scale=8), zorder=2)
        for identifier in sorted(selected):
            node = nodes[identifier]
            x, y = positions[identifier]
            color = cmap(norm(node['energy'])) if np.isfinite(node['energy']) else '#ff8c00'
            ax.scatter(x, y, s=size, facecolors=[color] if node['accepted'] else 'none',
                       edgecolors='#123b68', linewidths=.8, zorder=3)
            if node['extinct']:
                ax.scatter(x, y, s=size, marker='x', color='#d62728', linewidths=1.2, zorder=4)
            if font_size is not None:
                ax.annotate(str(identifier), (x, y), xytext=(0, -diameter / 2 - 3), textcoords='offset points',
                            ha='center', va='top', fontsize=font_size, color='#123b68')
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=100, facecolor='white')
    finally:
        plt.close(fig)


def plot_lineage(connection, directory):
    """Write one compact PNG per generation and return the output paths."""
    nodes, restarts = collect_lineage(connection, directory)
    paths = []
    generations = sorted({node['generation'] for node in nodes.values()} - {0})
    if not generations and nodes:
        generations = [0]
    metadata = getattr(connection, 'metadata', {})
    plans = metadata.get('generation_plans', {})
    for generation in generations:
        if generation <= 1:
            population = [identifier for identifier, node in nodes.items() if node['generation'] == 0]
        else:
            plan = plans.get(str(generation), {})
            population = plan.get('population', plan.get('parents', []))
        path = Path(directory) / 'results' / 'lineage' / f'gen{generation:04d}.png'
        _plot_generation(nodes, restarts, generation, path, population)
        paths.append(path)
    return paths
