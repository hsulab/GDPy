import numpy as np

from gdpx.data.array import AtomsNDArray

from .selector import BaseSelector


class StructureInfoSelector(BaseSelector):
    name: str = "structure_info"

    def __init__(self, targets: dict, inverse: bool = True, *args, **kwargs) -> None:
        """Initialise the selector.

        Args:
            targets: The target structure info criteria.
            inverse: Whether to select structures that do NOT meet the criteria.

        """
        super().__init__(*args, **kwargs)

        if self.group_by is not None:
            raise Exception("Grouping is not supported in comparison.")

        self.targets = targets
        self.inverse = inverse

        return

    def _mark_structures(self, data: AtomsNDArray) -> None:
        """Mark structures by sifting structure info as markers."""
        structures = data.get_marked_structures()

        check_value = lambda v, t: v in t if not self.inverse else v not in t

        selected_indices = []
        for i, structure in enumerate(structures):
            for key, values in self.targets.items():
                info_value = structure.info.get(key, None)
                if info_value is not None and check_value(info_value, values):
                    selected_indices.append(i)
                    break

        markers = np.argwhere(data.markers)
        selected_markers = np.array([markers[i] for i in selected_indices])
        data.markers = selected_markers

        return
