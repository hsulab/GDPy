"""GA family trees reconstructed from persisted candidate production steps."""
from collections import defaultdict
from pathlib import Path

import numpy as np


def collect_lineage(connection):
    """Keep original parents when later mutations refer to the same candidate ID."""
    histories = defaultdict(list)
    for row in connection.select(sort='id', columns=['id', 'numbers', 'key_value_pairs', 'data']):
        if row.get('confid') is not None and row.natoms and row.get('relaxed') in (0, 1):
            histories[row.confid].append(row)
    nodes = {}
    for identifier, rows in histories.items():
        inputs = [row for row in rows if not row.relaxed]
        evaluated = [row for row in rows if row.relaxed]
        first = inputs[0] if inputs else evaluated[0]
        final = evaluated[-1] if evaluated else first
        parents = list(dict.fromkeys(int(parent) for parent in first.data.get('parents', [])
                                     if int(parent) != identifier))
        # Older databases may only attach parent metadata to a later step.
        if not parents:
            for row in rows[1:]:
                parents = list(dict.fromkeys(int(parent) for parent in row.data.get('parents', [])
                                             if int(parent) != identifier))
                if parents:
                    break
        origins = [row.get('origin', '') for row in inputs]
        crossover = any(row.get('pairing') for row in inputs) or any(
            'Crossover' in origin or 'Pairing' in origin or origin == 'Parthenogenesis'
            for origin in origins)
        mutation = any(row.get('mutation') for row in inputs) or any('Mutation' in origin for origin in origins)
        if crossover:
            operation = 'crossover + mutation' if mutation else 'crossover'
        elif mutation:
            operation = 'mutation'
        elif first.data.get('builder') or any('Builder:' in origin for origin in origins):
            operation = 'builder'
        else:
            operation = 'unknown'
        nodes[identifier] = dict(
            generation=int(first.get('generation', final.get('generation', 0))),
            parents=parents, operation=operation, evaluated=bool(evaluated),
            extinct=bool(final.get('extinct', 0)), target=final.get('target'),
            builder=first.data.get('builder'))
    return nodes


def _layout(nodes):
    """Order siblings by their parents' positions to reduce crossings."""
    generations = defaultdict(list)
    for identifier, node in nodes.items():
        generations[node['generation']].append(identifier)
    positions = {}
    width = max(map(len, generations.values()))
    for row, (generation, identifiers) in enumerate(sorted(generations.items())):
        def order(identifier):
            parents = [positions[parent][0] for parent in nodes[identifier]['parents'] if parent in positions]
            return (np.mean(parents) if parents else width / 2, identifier)
        identifiers.sort(key=order)
        for column, identifier in enumerate(identifiers):
            positions[identifier] = ((column + 0.5) * width / len(identifiers), -row)
    return positions, sorted(generations), width


def plot_lineage(connection, directory, objective='energy'):
    """Write a compact PNG without altering candidates or random state."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import FancyArrowPatch

    nodes = collect_lineage(connection)
    if not nodes:
        return []
    positions, generations, width = _layout(nodes)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    try:
        fig.patch.set_facecolor('#fafbfc')
        ax.set_facecolor('#fafbfc')
        fig.subplots_adjust(left=0.075, right=0.89, bottom=0.05, top=0.97)
        ax.set(xlim=(-0.5, width + 0.5), ylim=(-len(generations) + 0.35, 0.65),
               yticks=-np.arange(len(generations)),
               yticklabels=[str(generation) for generation in generations], xticks=[])
        ax.tick_params(axis='y', length=0, labelsize=9, pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for row in range(len(generations)):
            ax.axhline(-row, color='#e4e8ed', linewidth=0.8, zorder=0)
        # Keep the raster fixed; scale markers and IDs to both population size
        # and generation count. Twenty candidates across eleven rows retain IDs.
        spacing = min(12 * 0.815 * 72 / (width + 1),
                      6 * 0.92 * 72 / (len(generations) + 0.3))
        diameter = min(18, max(3, spacing * 0.34))
        font_size = min(8, spacing * 0.23)
        show_ids = font_size >= 6 and max(len(str(identifier)) for identifier in nodes) * font_size * 0.6 < spacing * 0.85
        for identifier, node in nodes.items():
            for parent in node['parents']:
                if parent not in positions:
                    continue
                ax.add_patch(FancyArrowPatch(
                    positions[parent], positions[identifier], arrowstyle='-|>',
                    mutation_scale=max(3, diameter / 3), shrinkA=diameter * 0.6,
                    shrinkB=diameter * 0.6, linewidth=0.8, color='black', alpha=1.0,
                    connectionstyle='arc3,rad=0.04', zorder=1))
        values = [node['target'] for node in nodes.values()
                  if node['evaluated'] and node['target'] is not None and np.isfinite(node['target'])]
        norm = Normalize(min(values), max(values)) if values else Normalize(0, 1)
        if values and norm.vmin == norm.vmax:
            norm = Normalize(norm.vmin - 0.5, norm.vmax + 0.5)
        cmap = plt.get_cmap('coolwarm')
        for identifier, node in nodes.items():
            x, y = positions[identifier]
            value = node['target']
            color = cmap(norm(value)) if node['evaluated'] and value is not None and np.isfinite(value) else '#e1e5eb'
            ax.scatter([x], [y], s=diameter**2, marker='o',
                       color=color, edgecolor='#123b68', linewidth=0.6, zorder=3)
            if show_ids:
                ax.annotate(str(identifier), (x, y), xytext=(0, -diameter * 0.65),
                            textcoords='offset points', ha='center', va='top', fontsize=font_size,
                            color='#233247', zorder=5,
                            bbox=dict(facecolor='#fafbfc', edgecolor='none', pad=0.2, alpha=0.9))
        colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                               cax=fig.add_axes([0.915, 0.08, 0.013, 0.86]))
        colorbar.ax.tick_params(labelsize=8, colors='#123b68')
        colorbar.outline.set_edgecolor('#123b68')
        colorbar.set_label(f'{objective} [eV]', fontsize=8, color='#123b68')
        if not values:
            colorbar.set_ticks([])
        output = Path(directory) / 'results'
        output.mkdir(parents=True, exist_ok=True)
        path = output / 'family_tree.png'
        fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
        return [path]
    finally:
        plt.close(fig)
