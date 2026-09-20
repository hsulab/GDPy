"""Borrowed, reversible trial structures. No execution or persistence policy."""

import copy
import weakref

import numpy as np


_ACTIVE = {}
_MISSING = object()


class MoveProposal:
    """One exclusive in-place edit, with undo storage proportional to the edit.

    ``atoms`` is borrowed and represents the trial until commit/rollback. Array
    references must not escape a topology-changing transaction. Evaluators must
    capture their input before the caller rolls back. Neither this object nor
    its live undo log is a checkpoint format; persist ``metadata`` instead.
    """

    def __init__(self, atoms):
        active = _ACTIVE.get(id(atoms))
        if active is not None and active() is not None:
            raise RuntimeError("An active move already owns this Atoms object.")
        self.atoms = atoms
        self.metadata = {}
        self.diagnostic = "-"
        self.valid = True
        self.closed = False
        self._undo = []
        self._info = {}
        self._constraints = None
        self._calculator = atoms.calc
        _ACTIVE[id(atoms)] = weakref.ref(self)
        atoms.calc = None

    def _check(self):
        if self.closed:
            raise RuntimeError("Move proposal is already closed.")

    def watch(self, indices, names=("positions",)):
        """Save independent copies of only the rows about to be edited."""
        self._check()
        indices = np.unique(np.asarray(indices, dtype=int))
        rows = {name: self.atoms.arrays[name][indices].copy() for name in names}
        self._undo.append(("rows", indices, rows))

    def set_info(self, key, value):
        self._check()
        if key not in self._info:
            self._info[key] = copy.deepcopy(self.atoms.info[key]) if key in self.atoms.info else _MISSING
        self.atoms.info[key] = value

    def _protect_constraints(self):
        if self._constraints is None:
            self._constraints = copy.deepcopy(self.atoms.constraints)

    def before_append(self):
        """Guard a helper that appends particles (and may trim its own trials)."""
        self._check()
        self._protect_constraints()
        self._undo.append(("append", len(self.atoms), set(self.atoms.arrays)))

    def append(self, particle):
        self.before_append()
        self.atoms.extend(particle)

    def delete(self, indices):
        self._check()
        self._protect_constraints()
        if isinstance(indices, slice):
            indices = np.arange(*indices.indices(len(self.atoms)))
        else:
            indices = np.atleast_1d(np.asarray(indices))
            if indices.dtype == bool:
                if len(indices) != len(self.atoms):
                    raise IndexError("Deletion mask must match the structure length.")
                indices = np.flatnonzero(indices)
            indices = indices.astype(int)
            indices = np.where(indices < 0, indices + len(self.atoms), indices)
        indices = np.unique(indices)
        if np.any(indices < 0) or np.any(indices >= len(self.atoms)):
            raise IndexError("Deletion index outside the structure.")
        rows = {name: array[indices].copy() for name, array in self.atoms.arrays.items()}
        # ASE validates/remaps constraints before replacing the arrays.
        del self.atoms[indices]
        self._undo.append(("delete", indices, rows))

    @property
    def undo_nbytes(self):
        """Array storage retained for changed/removed rows, useful for profiling."""
        return sum(
            index.nbytes + sum(array.nbytes for array in rows.values())
            for kind, index, rows in self._undo if kind in {"rows", "delete"}
        )

    def rollback(self):
        self._check()
        atoms = self.atoms
        for kind, index, rows in reversed(self._undo):
            if kind == "rows":
                for name, values in rows.items():
                    atoms.arrays[name][index] = values
            elif kind == "append":
                for name in list(atoms.arrays):
                    if name not in rows:
                        del atoms.arrays[name]
                    elif len(atoms.arrays[name]) != index:
                        atoms.arrays[name] = atoms.arrays[name][:index].copy()
            elif kind == "delete":
                count = len(atoms) + len(index)
                retained = np.ones(count, dtype=bool)
                retained[index] = False
                for name, values in rows.items():
                    restored = np.empty((count, *values.shape[1:]), dtype=values.dtype)
                    restored[index] = values
                    restored[retained] = atoms.arrays[name]
                    atoms.arrays[name] = restored
        if self._constraints is not None:
            atoms.set_constraint(self._constraints)
        for key, value in self._info.items():
            if value is _MISSING:
                atoms.info.pop(key, None)
            else:
                atoms.info[key] = value
        # This calculator already belongs to the restored state. ASE's public
        # setter calls set_atoms(), which can copy the whole system or reset
        # cached results; restore the borrowed reference without reattaching it.
        atoms._calc = self._calculator
        self._release()
        return atoms

    def commit(self):
        self._check()
        atoms = self.atoms
        self._release()
        return atoms

    def _release(self):
        _ACTIVE.pop(id(self.atoms), None)
        self.atoms = None
        self._undo.clear()
        self._info.clear()
        self._constraints = None
        self._calculator = None
        self.closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *exc):
        if not self.closed:
            self.rollback()
