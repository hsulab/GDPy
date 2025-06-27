#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools
import pathlib
from typing import Optional

import numpy as np
from ase import Atoms
from ase.io import write

from gdpx.core.register import registers
from gdpx.data.array import AtomsNDArray
from gdpx.session.operation import Operation


def split_structures_by_ratio(
    structures: list[Atoms], ratio: float = 1.0, rng=np.random.default_rng()
) -> tuple[list[Atoms], ...]:
    """"""
    whether_split = not np.isclose(ratio, 1.0)

    if whether_split:
        num_structures = len(structures)
        indices = np.arange(num_structures)
        rng.shuffle(indices)

        split_idx = int(num_structures * ratio)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        print(train_indices)
        print(test_indices)

        train_structures = [structures[i] for i in train_indices]
        test_structures = [structures[i] for i in test_indices]
        datasets = [train_structures, test_structures]
    else:
        datasets = [structures]

    return datasets  # type: ignore


def transfer_structures_to_dataset(dataset, system_dirpath, structures, version: str, print_func) -> None:
    """"""
    root_dirpath = dataset.directory.resolve()

    strname = version + ".xyz"
    target_destination = system_dirpath / strname
    relative_desination = target_destination.relative_to(root_dirpath)

    num_structures = len(structures)
    if not target_destination.exists():
        write(target_destination, structures)
        print_func(f"num_structures {num_structures} -> {str(relative_desination)}")
    else:
        print_func(f"{str(relative_desination)} exists.")

    return


@registers.operation.register
class transfer(Operation):
    """Transfer worker results to target destination."""

    def __init__(
        self,
        structures,
        dataset,
        version,
        prefix: str = "",
        system: str = "mixed",
        side_dataset: Optional[str] = None,
        split_ratio: float = 1.0,
        clean_info: bool = False,
        set_pbc: bool = True,
        directory="./",
    ) -> None:
        """"""
        input_nodes = [structures, dataset]
        super().__init__(input_nodes=input_nodes, directory=directory)

        self.version = version

        self.prefix = prefix
        self.system = system  # molecule/cluster, surface, bulk

        self.side_dataset = side_dataset
        self.split_ratio = split_ratio

        self.clean_info = clean_info  # whether clean atoms info
        self.set_pbc = set_pbc  # Whether set structures to full pbc

        return

    def forward(self, structures: list[Atoms], dataset):
        """"""
        super().forward()

        if isinstance(structures, AtomsNDArray):
            structures = structures.get_marked_structures()
        num_structures = len(structures)
        self._print(f"{num_structures = }")

        target_dirpaths = [dataset.directory.resolve()]
        if self.side_dataset is not None:
            target_dirpaths.append(pathlib.Path(self.side_dataset).resolve())

        for target_dirpath in target_dirpaths:
            self._print(f"target dir: {str(target_dirpath)}")

        # Check chemical symbols
        system_dict = {}  # {formula: [indices]}

        # We need aggregate by ourselves
        # as groupby only collects contiguous data
        formulae = [a.get_chemical_formula() for a in structures]
        for k, v in itertools.groupby(enumerate(formulae), key=lambda x: x[1]):
            if k not in system_dict:
                system_dict[k] = [x[0] for x in v]
            else:
                system_dict[k].extend([x[0] for x in v])

        # Transfer data
        acc_num_structures = 0
        for formula, curr_indices in system_dict.items():
            curr_structures = [structures[i] for i in curr_indices]
            curr_num_frames = len(curr_structures)

            if self.set_pbc:
                for atoms in curr_structures:
                    atoms.set_pbc(True)

            if self.clean_info:
                self._clean_structures(curr_structures)

            system_type = self.system  # currently, use user input one
            dirname = "-".join([self.prefix, formula, system_type])

            split_structures = split_structures_by_ratio(curr_structures, self.split_ratio, rng=dataset.rng)
            for target_dirpath, target_structures in zip(target_dirpaths, split_structures):
                dataset.directory = target_dirpath
                target_subdir = dataset.directory.resolve() / dirname
                target_subdir.mkdir(parents=True, exist_ok=True)

                num_target_structures = len(target_structures)
                if num_target_structures == 0:
                    self._print(f"Skip {dirname} as it has no structures.")
                else:
                    transfer_structures_to_dataset(
                        dataset,
                        target_subdir,
                        target_structures,
                        self.version,
                        self._print,
                    )

            acc_num_structures += curr_num_frames

        dataset.directory = target_dirpaths[0]
        assert num_structures == acc_num_structures

        self.status = "finished"

        return dataset

    def _clean_structures(self, structures: list[Atoms]):
        """"""
        for atoms in structures:
            info_keys = copy.deepcopy(list(atoms.info.keys()))
            for k in info_keys:
                if k not in ["energy", "free_energy"]:
                    del atoms.info[k]

        return


if __name__ == "__main__":
    ...
