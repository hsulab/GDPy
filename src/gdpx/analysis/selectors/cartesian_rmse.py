"""Greedy displacement sampling, independently for each ordered trajectory."""
import numpy as np
from ase.geometry import find_mic

from gdpx.structures.groups import evaluate_group_expression
from .clustering import group_structures
from .selector import BaseSelector


class CartesianRMSESelector(BaseSelector):
    name = "cartesian_rmse"

    def __init__(self, rmse_interval, include_first=False, include_last=True,
                 mic=True, group_expression=None, **kwargs):
        super().__init__(**kwargs)
        if isinstance(rmse_interval, bool) or not np.isfinite(rmse_interval) or rmse_interval <= 0:
            raise ValueError("rmse_interval must be a positive finite distance in Angstrom.")
        for name, value in (("include_first", include_first), ("include_last", include_last), ("mic", mic)):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be boolean.")
        self.rmse_interval = float(rmse_interval)
        self.include_first, self.include_last = include_first, include_last
        self.mic, self.group_expression = mic, group_expression

    def _mark_structures(self, data):
        coordinates = np.argwhere(data.markers).tolist()
        if self.group_by is not None:
            groups = group_structures(data, self.group_by)
        else:
            # Flat CLI input can carry explicit IDs; otherwise the final axis is time.
            has_ids = ["trajectory_id" in data[tuple(m)].info for m in coordinates]
            if any(has_ids) and not all(has_ids):
                raise ValueError("trajectory_id must be present on every frame or none.")
            groups = {}
            for marker in coordinates:
                key = data[tuple(marker)].info["trajectory_id"] if all(has_ids) else tuple(marker[:-1])
                groups.setdefault(key, []).append(marker)
        selected = []
        for markers in groups.values():
            markers = sorted(markers)
            frames = [data[tuple(m)] for m in markers]
            first = frames[0]
            indices = (evaluate_group_expression(first, self.group_expression)
                       if self.group_expression is not None else list(range(len(first))))
            if not indices:
                raise ValueError("RMSE atom group is empty.")
            for frame in frames:
                if not np.array_equal(frame.numbers, first.numbers):
                    raise ValueError("Cartesian RMSE requires identical atom ordering and composition.")
                if not np.isfinite(frame.positions).all():
                    raise ValueError("Cartesian RMSE requires finite coordinates.")
                if self.mic and (not np.array_equal(frame.pbc, first.pbc) or
                                 (first.pbc.any() and not np.allclose(frame.cell, first.cell))):
                    raise ValueError("Minimum-image RMSE requires fixed cell and periodicity within each trajectory.")
            reference = first
            if self.include_first:
                selected.append(markers[0])
            for marker, frame in zip(markers[1:], frames[1:]):
                displacement = frame.positions[indices] - reference.positions[indices]
                if self.mic and first.pbc.any():
                    displacement = find_mic(displacement, first.cell, first.pbc)[0]
                rmse = float(np.sqrt(np.mean(displacement ** 2)))
                if rmse >= self.rmse_interval:
                    selected.append(marker)
                    reference = frame
            if self.include_last and markers[-1] not in selected:
                selected.append(markers[-1])
        data.markers = selected
