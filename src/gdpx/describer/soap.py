import copy

import numpy as np
import numpy.typing
from ase import Atoms
from dscribe.descriptors import SOAP

from gdpx.utils.profiler import CustomTimer

from .describer import BaseDescriber


class SoapDescriber(BaseDescriber):
    cache_features: str = "soap_features.npy"

    def __init__(self, params: dict, *args, **kwargs) -> None:
        """"""
        super().__init__(*args, **kwargs)

        self.descriptor = copy.deepcopy(params)

        return

    def run(self, structures):
        """"""
        ...
        self._print(f"soap is using n_jobs: {self.njobs}")

        # For data systems,
        # features = []
        # for system in structures:
        #     curr_frames = system._images
        #     if not (self.directory / system.prefix).exists():
        #         (self.directory / system.prefix).mkdir(parents=True)
        #     cache_features = self.directory / system.prefix / self.cache_features
        #     if not cache_features.exists():
        #         curr_features = self._compute_descripter(frames=curr_frames)
        #         np.save(cache_features, curr_features)
        #     else:
        #         curr_features = np.load(cache_features)
        #     features.extend(curr_features.tolist())
        # features = np.array(features)

        # For list of Atoms
        cache_features = self.directory / self.cache_features
        if not cache_features.exists():
            with CustomTimer("SOAP feature calculation", func=self._print):
                features = self._compute_descripter(frames=structures)
                np.save(cache_features, features)
        else:
            features = np.load(cache_features)
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
