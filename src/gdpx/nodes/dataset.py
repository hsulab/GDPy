#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools
import json
from typing import Mapping, Union

import numpy as np
from ase import Atoms
from ase.io import write

from gdpx.session.registry import workflow_registers as registers
from gdpx.data.array import AtomsNDArray
from gdpx.session.operation import Operation


def split_structures_by_ratio(
    structures: list[Atoms], splits: list[float] = [1.0], rng=np.random.default_rng()
) -> tuple[tuple[list[Atoms], ...], list[np.ndarray]]:
    """"""
    num_splits = len(splits)
    whether_split = num_splits > 1

    if whether_split:
        num_structures = len(structures)
        indices = np.arange(num_structures)
        rng.shuffle(indices)

        split_numbers = np.zeros(num_splits, dtype=int).tolist()
        for i, ratio in enumerate(splits[:-1]):
            split_number = int(num_structures * ratio)
            split_numbers_sum = sum(split_numbers)
            if split_numbers_sum + split_number > num_structures:
                split_number = num_structures - split_numbers_sum
            split_numbers[i] = split_number
        last_split_number = num_structures - sum(split_numbers)
        assert last_split_number >= 0, f"num_structures {num_structures}, split_numbers {split_numbers}"
        split_numbers[-1] = last_split_number

        split_numbers.insert(0, 0)
        edges = np.cumsum(split_numbers)

        datasets, split_indices = [], []
        for i in range(len(edges) - 1):
            start, end = edges[i], edges[i + 1]
            curr_indices = indices[start:end]
            curr_structures = [structures[j] for j in curr_indices]
            datasets.append(curr_structures)
            split_indices.append(curr_indices)
    else:
        datasets = [structures]
        split_indices = [np.arange(len(structures))]

    return datasets, split_indices  # type: ignore


def transfer_structures_to_dataset(dataset, system_dirpath, structures, version: str, print_func) -> None:
    """"""
    root_dirpath = dataset.directory.resolve()

    strname = version + ".xyz"
    target_destination = system_dirpath / strname
    relative_desination = target_destination.relative_to(root_dirpath)

    num_structures = len(structures)
    if not target_destination.exists():
        write(target_destination, structures)
        print_func(f"-> {dataset.directory.name:<21s} num_structures {num_structures} -> {str(relative_desination)}")
    else:
        print_func(f"-> {dataset.directory.name:<21s} {str(relative_desination)} exists.")

    return


@registers.operation.register
class transfer(Operation):
    """Transfer worker results to target destination."""

    def __init__(
        self,
        structures,
        version,
        prefix: str = "",
        suffix: str = "mixed",
        clean_info: bool = False,
        set_pbc: bool = True,
        directory="./",
        split_ratio: Union[float, Mapping[str, float]] = 1.0,
        **datasets,
    ) -> None:
        """"""
        datasets, splits = self._canonicalise_datasets(datasets, split_ratio=split_ratio)

        input_nodes = [structures, *datasets]
        super().__init__(input_nodes=input_nodes, directory=directory)

        self.version = version

        self.prefix = prefix
        self.suffix = suffix  # molecule/cluster, surface, bulk

        self.splits = splits

        self.clean_info = clean_info  # whether clean atoms info
        self.set_pbc = set_pbc  # Whether set structures to full pbc

        return

    def _canonicalise_datasets(self, datasets: dict, split_ratio: Union[float, Mapping[str, float]]):
        """"""
        if not isinstance(split_ratio, Mapping):
            split_ratio = dict(dataset=split_ratio)

        if "dataset" not in datasets:
            raise Exception("At least one dataset must be provided with the name `dataset`.")

        dataset_names = list(datasets.keys())
        dataset_names.insert(
            0, dataset_names.pop(dataset_names.index("dataset"))
        )  # Ensure the first dataset is named `dataset`

        sorted_datasets = []
        for name, dataset in datasets.items():
            if not name.startswith("dataset"):
                raise Exception(f"Dataset name `{name}` must start with `dataset`, but got `{name}` for `{dataset}`.")
            sorted_datasets.append(dataset)

        num_datasets, num_ratios = len(sorted_datasets), len(split_ratio)
        if num_datasets == num_ratios:
            sorted_ratios = [split_ratio[name] for name in dataset_names]
        else:
            raise Exception(
                f"Number of datasets `{num_datasets}` does not match number of split ratios `{num_ratios}`."
            )

        # Check if datasets have different directories
        dirpaths = [dataset.directory for dataset in sorted_datasets]
        if len(set(dirpaths)) != len(dirpaths):
            raise Exception("All datasets must have different directories.")

        ratio_sum = sum(sorted_ratios)
        if not np.isclose(ratio_sum, 1.0):
            raise Exception(f"Split ratios must sum to 1.0, but got {ratio_sum}.")

        return sorted_datasets, sorted_ratios

    def forward(self, structures: list[Atoms], *datasets):
        """"""
        super().forward()

        if isinstance(structures, AtomsNDArray):
            structures = structures.get_marked_structures()
        num_structures = len(structures)
        self._print(f"{num_structures = }")

        self._print("target datasets:")
        target_dirpaths = [dataset.directory.resolve() for dataset in datasets]
        for target_dirpath in target_dirpaths:
            self._print(f"-> dataset: {str(target_dirpath)}")
        main_dataset = datasets[0]

        # Skip transfer if cache_splits.json exists
        if (self.directory / "cache_splits.json").exists():
            self._print("cache_splits.json exists, skip transfer.")
            self.status = "finished"
            return main_dataset

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
        cache_splits = {}
        acc_num_structures = 0
        for formula, curr_indices in system_dict.items():
            self._print(f"{formula:<24s} has {len(curr_indices):>4d} structures.")
            curr_structures = [structures[i] for i in curr_indices]
            curr_num_frames = len(curr_structures)

            if self.set_pbc:
                for atoms in curr_structures:
                    atoms.set_pbc(True)

            if self.clean_info:
                self._clean_structures(curr_structures)

            system_type = self.suffix  # currently, use user input one
            dirname = "-".join([self.prefix, formula, system_type])

            cache_info = dict(rng_state=main_dataset.rng.bit_generator.state)
            split_structures, split_indices = split_structures_by_ratio(
                curr_structures, self.splits, rng=main_dataset.rng
            )
            cache_info["index"] = [indices.tolist() for indices in split_indices]
            for dataset, target_structures in zip(datasets, split_structures):
                target_subdir = dataset.directory.resolve() / dirname
                target_subdir.mkdir(parents=True, exist_ok=True)

                num_target_structures = len(target_structures)
                if num_target_structures == 0:
                    self._print(f"-> {dataset.directory.name:<24s} skips {dirname} as it has no structures.")
                else:
                    transfer_structures_to_dataset(
                        dataset,
                        target_subdir,
                        target_structures,
                        self.version,
                        self._print,
                    )
            assert formula not in cache_splits, f"Formula {formula} already exists in cache_split_indices."
            cache_splits[formula] = cache_info
            acc_num_structures += curr_num_frames

        with open(self.directory / "cache_splits.json", "w") as fopen:
            json.dump(cache_splits, fopen, indent=2)
        self._print("save cache_splits.json")

        assert num_structures == acc_num_structures

        self.status = "finished"

        return main_dataset

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
