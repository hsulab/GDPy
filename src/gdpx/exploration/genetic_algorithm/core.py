"""GDPy-owned genetic-operator primitives and deterministic RNG streams."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from ase import Atoms


def _seed_material(seed: int | Mapping[str, Any]) -> bytes:
    if isinstance(seed, (int, np.integer)):
        return str(int(seed)).encode()
    return json.dumps(seed, sort_keys=True, separators=(",", ":")).encode()


class RandomStreamRegistry:
    """Create order-independent named PCG64 streams from one root seed."""

    def __init__(self, seed: int | Mapping[str, Any]):
        self._material = _seed_material(seed)
        self._streams: dict[str, np.random.Generator] = {}

    def get(self, name: str) -> np.random.Generator:
        if name not in self._streams:
            digest = hashlib.sha256(self._material + b"\0" + name.encode()).digest()
            entropy = np.frombuffer(digest[:16], dtype=np.uint32).tolist()
            self._streams[name] = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
        return self._streams[name]

    def seed(self, name: str) -> int:
        """Return a stable scalar seed without advancing any stream."""
        digest = hashlib.sha256(self._material + b"\0seed\0" + name.encode()).digest()
        return int.from_bytes(digest[:8], "little")

    def snapshot(self) -> dict[str, dict]:
        return {name: copy.deepcopy(stream.bit_generator.state) for name, stream in self._streams.items()}

    def restore(self, states: Mapping[str, Mapping[str, Any]]) -> None:
        for name, state in states.items():
            self.get(name).bit_generator.state = copy.deepcopy(dict(state))


class OffspringCreator:
    """Base class for GDPy crossover and mutation operators."""

    descriptor = "OffspringCreator"
    min_inputs = 0

    def __init__(self, verbose: bool = False, num_muts: int = 1, rng=None):
        self.verbose = verbose
        self.num_muts = num_muts
        self.rng = np.random.default_rng() if rng is None else rng

    def get_min_inputs(self) -> int:
        return self.min_inputs

    @classmethod
    def initialize_individual(cls, parent: Atoms, individual: Atoms | None = None) -> Atoms:
        individual = Atoms(pbc=parent.pbc, cell=parent.cell) if individual is None else individual.copy()
        individual.info["key_value_pairs"] = {"extinct": 0}
        individual.info["data"] = {}
        return individual

    def finalize_individual(self, individual: Atoms) -> Atoms:
        individual.info["key_value_pairs"]["origin"] = self.descriptor
        return individual


class OperationSelector:
    """Select operations according to relative positive weights."""

    def __init__(self, probabilities: Sequence[float], oplist: Sequence, rng=None):
        if len(probabilities) != len(oplist):
            raise ValueError("Operation probabilities and operators must have equal lengths.")
        if any(weight < 0 for weight in probabilities) or (probabilities and sum(probabilities) <= 0):
            raise ValueError("Operation probabilities must be non-negative with a positive sum.")
        self.oplist = list(oplist)
        self.rho = np.cumsum(probabilities)
        self.rng = np.random.default_rng() if rng is None else rng

    def _index(self) -> int:
        if not self.oplist:
            raise RuntimeError("Cannot select from an empty operation list.")
        return int(np.searchsorted(self.rho, self.rng.random() * self.rho[-1], side="right"))

    def get_new_individual(self, candidate_list):
        return self.oplist[self._index()].get_new_individual(candidate_list)

    def get_operator(self):
        return self.oplist[self._index()]


class CombinationMutation(OffspringCreator):
    """Apply two or more mutation objects successively."""

    descriptor = "CombinationMutation"

    def __init__(self, *mutations, verbose: bool = False, rng=None):
        if len(mutations) < 2:
            raise ValueError("CombinationMutation requires at least two mutations.")
        super().__init__(verbose=verbose, rng=rng)
        self.operators = mutations

    def mutate(self, atoms: Atoms):
        candidate = atoms
        for mutation in self.operators:
            candidate = mutation.mutate(candidate)
            if candidate is None:
                return None
        return candidate

    def get_new_individual(self, parents):
        candidate = self.mutate(parents[0])
        if candidate is None:
            return None, f"mutation: {self.descriptor}"
        candidate = self.initialize_individual(parents[0], candidate)
        candidate.info["data"]["parents"] = [parents[0].info["confid"]]
        return self.finalize_individual(candidate), f"mutation: {self.descriptor}"
