"""Versioned, exact fingerprints and lossless snapshots of calculation inputs."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import tempfile
from collections.abc import Mapping

import numpy as np
from ase import Atoms
from ase.io.jsonio import decode, encode

FINGERPRINT_VERSION = 1


def normalise_value(value):
    """Normalize configuration/constraint values without lossy string fallbacks."""
    if isinstance(value, np.ndarray):
        return normalise_value(value.tolist())
    if isinstance(value, np.generic):
        return normalise_value(value.item())
    if isinstance(value, pathlib.Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite values cannot be fingerprinted.")
        return 0.0 if value == 0 else value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Fingerprint mappings require string keys.")
        return {key: normalise_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [normalise_value(item) for item in value]
    raise TypeError(f"Unsupported fingerprint value: {type(value).__name__}")


def payload_digest(payload) -> str:
    canonical = json.dumps(normalise_value(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _array(value):
    array = np.asarray(value)
    kind = array.dtype.kind
    if kind in "fc":
        if array.dtype.itemsize > (8 if kind == "f" else 16):
            raise ValueError("Structure floats must fit in float64 without precision loss.")
        if not np.isfinite(array).all():
            raise ValueError("Non-finite structure arrays cannot be fingerprinted.")
        array = np.array(array, dtype="<f8" if kind == "f" else "<c16", order="C")
        # Signed zero has no physical significance.
        if kind == "c":
            array.real[array.real == 0] = 0
            array.imag[array.imag == 0] = 0
        else:
            array[array == 0] = 0
    elif kind in "iu":
        if kind == "u" and array.size and array.max() > np.iinfo(np.int64).max:
            raise ValueError("Structure integers must fit in int64.")
        array = np.asarray(array, dtype="<i8", order="C")
    elif kind == "b":
        array = np.asarray(array, dtype="u1", order="C")
    elif kind in "US":
        return {"dtype": "text", "shape": list(array.shape), "values": array.astype(str).tolist()}
    else:
        raise TypeError(f"Unsupported structure array dtype: {array.dtype}")
    return {"dtype": array.dtype.str, "shape": list(array.shape), "bytes": array.tobytes().hex()}


def structure_digest(frames) -> str:
    """Hash ordered frames/atoms, all arrays, cell, PBC, and constraints.

    Optional standard arrays are normalized to ASE defaults. Info, calculators,
    results, and cell-display offsets are excluded. No coordinate rounding or
    permutation/translation/rotation equivalence is applied.
    """
    structures = []
    for atoms in frames:
        arrays = dict(atoms.arrays)
        arrays.update(
            tags=atoms.get_tags(),
            masses=atoms.get_masses(),
            momenta=atoms.get_momenta(),
            initial_charges=atoms.get_initial_charges(),
            initial_magmoms=atoms.get_initial_magnetic_moments(),
        )
        structures.append({
            "arrays": {name: _array(array) for name, array in sorted(arrays.items())},
            "cell": _array(atoms.cell.array),
            "pbc": _array(atoms.pbc),
            "constraints": [constraint.todict() for constraint in atoms.constraints],
        })
    return payload_digest({"format": "gdpx-structures", "version": FINGERPRINT_VERSION,
                           "structures": structures})


def atomic_write_text(path, content):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        temporary = pathlib.Path(handle.name)
        try:
            handle.write(content)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_structure_inputs(path, frames):
    """Save ASE objects losslessly, verifying the serialization before publishing."""
    frames = [frame.copy() for frame in frames]
    for frame in frames:
        frame.info = {}
    digest = structure_digest(frames)
    content = encode({"version": FINGERPRINT_VERSION, "structure_digest": digest, "frames": frames})
    restored = decode(content)
    if structure_digest(restored["frames"]) != digest:
        raise ValueError("Structure input serialization changed its fingerprint.")
    atomic_write_text(path, content)
    return digest


def read_structure_inputs(path, expected_digest=None):
    data = decode(pathlib.Path(path).read_text(encoding="utf-8"))
    if data.get("version") != FINGERPRINT_VERSION:
        raise ValueError("Unsupported structure fingerprint version; prepare a new run.")
    frames = data["frames"]
    if not isinstance(frames, list) or not all(isinstance(frame, Atoms) for frame in frames):
        raise ValueError("Invalid structure snapshot.")
    digest = structure_digest(frames)
    if digest != data["structure_digest"] or (expected_digest is not None and digest != expected_digest):
        raise ValueError(f"Structure fingerprint mismatch: {path}")
    return frames
