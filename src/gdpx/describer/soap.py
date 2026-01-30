import copy
from typing import Optional

import numpy as np
import numpy.typing
from ase import Atoms
from ase.io import write
from dscribe.descriptors import SOAP

from gdpx.geometry.cleave import cleave_structures_by_group
from gdpx.utils.profiler import CustomTimer

from .describer import BaseDescriber


class SoapDescriber(BaseDescriber):
    cache_features: str = "soap_features.npy"

    def __init__(self, params: dict, group: Optional[str] = None, dump_cleaved: bool = False, *args, **kwargs) -> None:
        """"""
        super().__init__(*args, **kwargs)

        self.descriptor = copy.deepcopy(params)

        self.group = group
        self.dump_cleaved = dump_cleaved

        return

    def run(self, structures):
        """"""
        self._print(f"soap is using n_jobs: {self.njobs}")

        cache_features = self.directory / self.cache_features
        if not cache_features.exists():
            # Cleave structures by group expression if specified
            if self.group is not None:
                self._print(f"Cleave structures by group expression: {self.group}")
                new_structures, _ = cleave_structures_by_group(structures, grp_expr=self.group)
                num_atoms_in_group = [len(s) for s in new_structures]
                self._print(f"Number of structures after cleaving: {len(structures)}")
                self._print(
                    f"num_atoms statistics in the specified group: min={min(num_atoms_in_group)}, max={max(num_atoms_in_group)}, avg={sum(num_atoms_in_group) / len(num_atoms_in_group):.2f}"
                )
                num_new_structures = len(new_structures)
                if num_new_structures != len(structures):
                    raise Exception(
                        f"Each structure must contain the specified group, and the number of structures after cleaving ({num_new_structures}) does not match original ({len(structures)})."
                    )
                if self.dump_cleaved:
                    write(self.directory / "cleaved_structures.xyz", new_structures)
            else:
                new_structures = structures

            with CustomTimer("soap feature calculation", func=self._print):
                features = self._compute_descripter(frames=new_structures)
                np.save(cache_features, features)
        else:
            features = np.load(cache_features)
            self._print(f"Loaded cached features from {cache_features}.")
        self._debug(f"shape of features: {features.shape}")

        return features

    def _compute_descripter(self, frames: list[Atoms]) -> numpy.typing.NDArray:
        """Calculate vector-based descriptors.

        Each structure is represented by a vector.

        """
        self._print("start calculating features...")
        desc_params = copy.deepcopy(self.descriptor)

        soap = SOAP(**desc_params)
        ndim = soap.get_number_of_features()
        self._print(f"soap descriptor dimension: {ndim}")
        features = soap.create(frames, n_jobs=self.njobs)
        self._print("finished calculating features...")

        # Save calculated features
        assert isinstance(features, np.ndarray)
        features = features.reshape(-1, ndim)

        return features
