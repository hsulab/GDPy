#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import abc
import pathlib
from typing import Any, Optional, Union

from gdpx.core.component import BaseComponent
from gdpx.data.array import AtomsNDArray
from gdpx.data.dataset import AbstractDataloader
from gdpx.nodes.builder import canonicalise_builder
from gdpx.nodes.computer import canonicalise_worker
from gdpx.worker.drive import DriverBasedWorker


def canonicalise_structures_to_validate(structures) -> dict[str, Any]:
    """Validator can accept various formats of input structures.

    Note: 
        In an active session, the dataset is dynamic, thus,
        we need load the dataset before run.

    Returns:
        A dict of structures to validate.

    """
    if hasattr(structures, "items"):  # check if the input is a dict-like object
        stru_dict = structures
    else:  # assume it is just an AtomsNDArray
        stru_dict = {}
        stru_dict["reference"] = structures

    v_dict = {}
    for k, v in stru_dict.items():
        if isinstance(v, dict):
            ...
        elif isinstance(v, list):
            ...
        elif isinstance(v, AtomsNDArray):
            ...
        elif isinstance(v, AbstractDataloader):
            v = v.load_frames()
        else:
            raise Exception(f"{k} structures {type(v)} is not a dict or loader.")
        v_dict[k] = v

    return v_dict


class BaseValidator(BaseComponent):

    def __init__(
        self,
        structures: Optional[Any]=None,
        worker: Optional[DriverBasedWorker]=None,
        directory: Union[str, pathlib.Path] = "./",
        random_seed: Optional[Union[int, dict]] = None,
    ) -> None:
        """Base class for validators.

        Args:
            structures: The reference structures to validate.

        """
        super().__init__(directory=directory, random_seed=random_seed)

        self.structures = canonicalise_builder(structures)

        self.worker = canonicalise_worker(worker)

        return

    @abc.abstractmethod
    def run(self, *args, **kwargs) -> bool:
        """"""
        if not self.directory.exists():
            self.directory.mkdir(parents=True)

        ...


# For backwards compatibility
AbstractValidator = BaseValidator


if __name__ == "__main__":
    ...

