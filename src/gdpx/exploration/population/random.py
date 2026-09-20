"""Deterministic named streams for population searches."""
import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any
import numpy as np

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


