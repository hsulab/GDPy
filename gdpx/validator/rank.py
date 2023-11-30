#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Mapping

from ..data.array import AtomsNDArray
from .validator import AbstractValidator

class RankValidator(AbstractValidator):

    def run(self, dataset: Mapping[str, AtomsNDArray], *args, **kwargs):
        """"""
        super().run(*args, **kwargs)

        # -
        prediction = dataset["prediction"].get_marked_structures()
        reference = dataset["reference"].get_marked_structures()
        self._print(prediction)
        self._print(reference)

        return


if __name__ == "__main__":
    ...