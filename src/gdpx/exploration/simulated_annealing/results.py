"""Final structures and candidate evolution for population MD annealing."""

import numpy as np
from ase.io import write


def write_results(directory, candidates, population, objective):
    """Export all evaluated endpoints and highlight the final retained minima."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    directory.mkdir(parents=True, exist_ok=True)
    if candidates:
        write(directory / "all_candidates.xyz", candidates)
    else:
        (directory / "all_candidates.xyz").write_text("")

    retained = {atoms.info["confid"] for atoms in population}
    nodes = {}
    for atoms in candidates:
        pairs = atoms.info["key_value_pairs"]
        target = pairs["target"]
        if np.isfinite(target):
            nodes[atoms.info["confid"]] = (
                pairs["generation"], target, atoms.info.get("data", {}).get("parents", []))

    fig, ax = plt.subplots(figsize=(12, 8))
    try:
        for generation, target, parents in nodes.values():
            for parent in parents:
                if parent in nodes:
                    previous_generation, previous_target, _ = nodes[parent]
                    ax.plot([previous_generation, generation], [previous_target, target],
                            color="0.75", linewidth=0.7, alpha=0.6, zorder=1)
        ax.scatter([node[0] for node in nodes.values()], [node[1] for node in nodes.values()],
                   alpha=0.65, label="Evaluated candidates", zorder=2)
        selected = [node for identifier, node in nodes.items() if identifier in retained]
        if selected:
            ax.scatter([node[0] for node in selected], [node[1] for node in selected],
                       s=100, facecolors="none", edgecolors="darkorange", linewidths=1.8,
                       label="Final retained population", zorder=3)
        generations = sorted({node[0] for node in nodes.values()})
        if generations:
            minima = [min(node[1] for node in nodes.values() if node[0] == generation)
                      for generation in generations]
            ax.step(generations, np.minimum.accumulate(minima), where="post",
                    color="black", linestyle="--", label="Best evaluated so far")
        ax.set(title="SA Candidate Evolution", xlabel="Generation",
               ylabel=f"{objective.replace('_', ' ').capitalize()} [eV]")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        fig.savefig(directory / "pop.png", bbox_inches="tight")
    finally:
        plt.close(fig)
