"""Select the lowest-energy, force-converged, structurally distinct candidates."""

import numpy as np

from gdpx.analysis.comparators import create_comparator

from .selector import BaseSelector


class MinimaSelector(BaseSelector):
    """Greedy energy-ranked selection using a configured structural comparator.

    Forces are checked on all unconstrained atoms, using the maximum vector
    norm. This tests relaxation convergence, not positive Hessian eigenvalues.
    The comparator selects comparison atoms with its group_expression. The
    complete structure still contributes to energies and force checks.
    Similarity follows every criterion in the configured comparator.
    Returned structures retain the input order, as with other selectors.
    """

    name = "minima"

    def __init__(self, comparator=None, number=8, fmax=0.05, *args, **kwargs):
        obsolete = {"n_top", "ofp", "rmsd"} & kwargs.keys()
        if obsolete:
            raise ValueError(
                f"Removed minima parameters: {sorted(obsolete)}. Use comparator with "
                "method and group_expression; use rmsd_max for permutation_rmsd."
            )
        super().__init__(*args, **kwargs)
        if self.group_by is not None:
            raise ValueError("Grouping is not supported in minima selection.")
        if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
            raise ValueError("number must be a positive integer.")
        if not np.isfinite(fmax) or fmax <= 0:
            raise ValueError("fmax must be positive and finite.")
        if comparator is None:
            raise ValueError("Minima selection requires an explicit comparator configuration.")
        self.number = number
        self.fmax = fmax
        self.comparator = create_comparator(comparator)
        if not all(callable(getattr(self.comparator, method, None))
                   for method in ("prepare_data", "compare_fingerprints")):
            raise TypeError("Minima comparator must implement prepare_data and compare_fingerprints.")

    def _mark_structures(self, data):
        structures = data.get_marked_structures()
        markers = np.argwhere(data.markers)
        energies, eligible = [], []
        for i, atoms in enumerate(structures):
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces(apply_constraint=True)
            if not np.isfinite(energy) or not np.all(np.isfinite(forces)):
                raise ValueError("Minima selection requires finite energies and forces.")
            energies.append(energy)
            if np.max(np.linalg.norm(forces, axis=1)) <= self.fmax:
                eligible.append(i)

        ranked = sorted(eligible, key=lambda index: (energies[index], index))
        prepared = self.comparator.prepare_data([structures[i] for i in ranked])
        if len(prepared) != len(ranked):
            raise ValueError("Comparator must prepare one representation per candidate.")
        selected, fingerprints = [], []
        for i, fingerprint in zip(ranked, prepared):
            if all(not self.comparator.compare_fingerprints(fingerprint, previous)
                   for previous in fingerprints):
                selected.append(i)
                fingerprints.append(fingerprint)
                if len(selected) == self.number:
                    break

        if len(selected) < self.number:
            raise ValueError(
                f"Requested {self.number} distinct minima, found only {len(selected)} "
                f"among {len(eligible)} force-converged candidates."
            )
        data.markers = markers[selected]
