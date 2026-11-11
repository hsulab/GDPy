"""Reference all SA plot elements to the generation-zero minimum."""
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from matplotlib.figure import Figure

from gdpx.exploration.simulated_annealing.results import write_results


def candidate(identifier, generation, energy, parents=()):
    atoms = Atoms("Cu")
    atoms.calc = SinglePointCalculator(atoms, energy=energy)
    atoms.info.update(confid=identifier,
                      key_value_pairs={"generation": generation, "target": energy},
                      data={"parents": list(parents)})
    return atoms


@pytest.mark.parametrize("objective", ["energy", "formation_energy"])
def test_relative_generation_zero_reference(tmp_path, monkeypatch, objective):
    candidates = [candidate(0, 0, -100), candidate(1, 0, -102),
                  candidate(2, 1, -101, [1]), candidate(3, 2, -103, [2])]
    axes = []
    monkeypatch.setattr(Figure, "savefig", lambda fig, *args, **kwargs: axes.append(fig.axes[0]))
    write_results(tmp_path, candidates, [candidates[1], candidates[3]], objective)
    ax = axes[0]
    np.testing.assert_allclose(ax.collections[0].get_offsets()[:, 1], [2, 0, 1, -1])
    np.testing.assert_allclose(ax.collections[1].get_offsets()[:, 1], [0, -1])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [0, 1])
    np.testing.assert_allclose(ax.lines[1].get_ydata(), [1, -1])
    np.testing.assert_allclose(ax.lines[2].get_ydata(), [0, 0, -1])
    assert ax.get_ylabel() == f"Relative {objective.replace('_', ' ')} [eV] (generation 0 minimum = 0)"
    assert [a.info["key_value_pairs"]["target"] for a in candidates] == [-100, -102, -101, -103]
    assert [a.get_potential_energy() for a in read(tmp_path / "all_candidates.xyz", ":")] == [-100, -102, -101, -103]


@pytest.mark.parametrize("candidates", [[], [candidate(1, 1, -101)]])
def test_missing_generation_zero_has_no_false_reference(tmp_path, monkeypatch, candidates):
    axes = []
    monkeypatch.setattr(Figure, "savefig", lambda fig, *args, **kwargs: axes.append(fig.axes[0]))
    write_results(tmp_path, candidates, [], "energy")
    assert axes[0].get_ylabel() == "Energy [eV]"
    if candidates:
        np.testing.assert_allclose(axes[0].collections[0].get_offsets()[:, 1], [-101])
