"""A solid hemisphere oriented by a plane normal."""

import numpy as np

from .sphere import SphereRegion


class HemisphereRegion(SphereRegion):
    """Include points within radius with (position - origin) dot normal >= 0.

    The origin is the sphere center and the center of the flat face. The
    normal points toward the included half and defaults to the positive z axis.
    Both the spherical surface and the flat face are included.
    """

    def __init__(self, origin: list[float], radius: float, normal=(0.0, 0.0, 1.0)):
        origin = np.asarray(origin, dtype=float)
        normal = np.asarray(normal, dtype=float)
        radius = float(radius)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("Hemisphere origin must contain three finite coordinates.")
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("Hemisphere radius must be finite and positive.")
        if normal.shape != (3,) or not np.all(np.isfinite(normal)) or not np.any(normal):
            raise ValueError("Hemisphere normal must be a finite, nonzero three-vector.")
        # Scale first to avoid overflow or underflow for valid finite normals.
        normal = normal / np.max(np.abs(normal))
        self._normal = normal / np.linalg.norm(normal)
        super().__init__(origin=origin, radius=radius)

    @staticmethod
    def from_str(command: str):
        """Parse 'hemisphere ox oy oz radius [nx ny nz]'."""
        values = [float(x) for x in command.strip().split()[1:]]
        if len(values) not in (4, 7):
            raise ValueError("Hemisphere requires an origin, radius, and optional normal.")
        normal = values[4:] if len(values) == 7 else (0.0, 0.0, 1.0)
        return HemisphereRegion(values[:3], values[3], normal)

    def _get_a_random_position(self, rng):
        """Sample uniformly in volume, reflecting the excluded half-ball."""
        radial, cosine, azimuth = rng.uniform(0.0, 1.0, 3)
        radius = self._radius * np.cbrt(radial)
        cosine = 2.0 * cosine - 1.0
        azimuth *= 2.0 * np.pi
        sine = np.sqrt(max(0.0, 1.0 - cosine**2))
        offset = radius * np.array([sine * np.cos(azimuth), sine * np.sin(azimuth), cosine])
        projection = np.dot(offset, self._normal)
        if projection < 0.0:
            offset -= 2.0 * projection * self._normal
        return self._origin + offset

    def _is_within_region(self, position) -> bool:
        offset = np.asarray(position) - self._origin
        return bool(np.linalg.norm(offset) <= self._radius and np.dot(offset, self._normal) >= 0.0)

    def get_volume(self) -> float:
        return super().get_volume() / 2.0

    def as_dict(self) -> dict:
        return {
            "method": "hemisphere",
            "origin": self._origin.tolist(),
            "radius": self._radius,
            "normal": self._normal.tolist(),
        }
