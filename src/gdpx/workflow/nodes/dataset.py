#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools
import json
import os
import pathlib
import tempfile
from typing import Mapping, Union

import numpy as np
import yaml
from ase import Atoms
from ase.io import write

from gdpx.data.array import AtomsNDArray
from gdpx.data.loaders.dataset import XyzSnapshotDataloader
from gdpx.workflow.session.operation import Operation
from gdpx.workflow.session.registry import workflow_registers as registers
from gdpx.workflow.state import NamedOutputs


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


def transfer_structures_to_dataset(
    dataset_name, root_dirpath, system_dirpath, structures, version: str, print_func
) -> None:
    """"""
    root_dirpath = pathlib.Path(root_dirpath).resolve()

    strname = version + ".xyz"
    target_destination = (system_dirpath / strname).resolve()
    relative_desination = target_destination.relative_to(root_dirpath)

    num_structures = len(structures)
    if not target_destination.exists():
        write(target_destination, structures)
        print_func(f"-> {dataset_name:<21s} num_structures {num_structures} -> {str(relative_desination)}")
    else:
        print_func(f"-> {dataset_name:<21s} {str(relative_desination)} exists.")

    return


@registers.operation.register
class transfer(Operation):
    """Transfer worker results to target destination."""

    @classmethod
    def validates_output(cls, spec, output: str) -> bool:
        return output in spec.inputs and output.startswith("dataset")

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
        dataset_names, datasets, splits = self._canonicalise_datasets(
            datasets, split_ratio=split_ratio
        )

        input_nodes = [structures, *datasets]
        super().__init__(input_nodes=input_nodes, directory=directory)

        self.version = version

        self.prefix = prefix
        self.suffix = suffix  # molecule/cluster, surface, bulk

        self.splits = splits
        self.dataset_names = dataset_names

        self.clean_info = clean_info  # whether clean atoms info
        self.set_pbc = set_pbc  # Whether set structures to full pbc
        self._state_artifacts = {}

        return

    def bind_state_artifact(
        self,
        output: str,
        state_name: str,
        root: str | pathlib.Path,
        iteration: int,
    ) -> None:
        """Route one state-carrying output to its central artifact collection."""
        self._state_artifacts[output] = {
            "root": pathlib.Path(root).resolve(),
            "state": state_name,
            "iteration": iteration,
        }

    def _write_version_manifest(self, name, snapshot, new_shards):
        binding = self._state_artifacts.get(name)
        if binding is None:
            return snapshot.extend_shards(new_shards)

        artifact_root = binding["root"] / "artifacts" / "datasets" / binding["state"]
        version = f"{binding['iteration']:04d}"
        all_shards = [*snapshot.shards, *new_shards]
        systems = {}
        for shard in all_shards:
            path = pathlib.Path(shard["path"]).resolve()
            if any(self._is_within(path, source) for source in snapshot.sources):
                continue
            system = "/".join(shard["system"])
            try:
                location = str(path.relative_to(binding["root"]))
            except ValueError:
                location = str(path)
            systems.setdefault(system, []).append(location)
        manifest = {
            "format": "gdpx.structure-dataset/v1",
            "iteration": binding["iteration"],
            "codec": "extxyz",
            "loader": {
                "batchsize": snapshot.batchsize,
                "train_ratio": snapshot.train_ratio,
                "random_seed": snapshot.random_seed,
                "prop_keys": snapshot.prop_keys,
                "sources": [
                    self._path_reference(path, binding["root"])
                    for path in snapshot.sources
                ],
            },
            "systems": systems,
        }
        target = artifact_root / "versions" / f"{version}.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=target.parent, delete=False, encoding="utf-8"
        ) as stream:
            temporary = pathlib.Path(stream.name)
            yaml.safe_dump(manifest, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return snapshot.extend_shards(new_shards, manifest=target)

    @staticmethod
    def _is_within(path: pathlib.Path, directory: pathlib.Path) -> bool:
        try:
            path.relative_to(directory.resolve())
            return True
        except ValueError:
            return False

    @staticmethod
    def _path_reference(path: pathlib.Path, root: pathlib.Path) -> str:
        path = path.resolve()
        try:
            return str(path.relative_to(root.resolve()))
        except ValueError:
            return str(path)

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
        for name in dataset_names:
            dataset = datasets[name]
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
        dirpaths = [
            dataset.value.directory if hasattr(dataset, "value") else dataset.directory
            for dataset in sorted_datasets
        ]
        if len(set(dirpaths)) != len(dirpaths):
            raise Exception("All datasets must have different directories.")

        ratio_sum = sum(sorted_ratios)
        if not np.isclose(ratio_sum, 1.0):
            raise Exception(f"Split ratios must sum to 1.0, but got {ratio_sum}.")

        return dataset_names, sorted_datasets, sorted_ratios

    def forward(self, structures: list[Atoms], *datasets):
        """"""
        super().forward()

        if isinstance(structures, AtomsNDArray):
            structures = structures.get_marked_structures()
        num_structures = len(structures)
        self._print(f"{num_structures = }")

        self._print("target datasets:")
        snapshots = [XyzSnapshotDataloader.from_loader(dataset) for dataset in datasets]
        dataset_names = self.dataset_names
        target_dirpaths = []
        for name in dataset_names:
            binding = self._state_artifacts.get(name)
            if binding is None:
                target = self.directory / "datasets" / name
            else:
                target = (
                    binding["root"]
                    / "artifacts"
                    / "datasets"
                    / binding["state"]
                    / "systems"
                )
            target_dirpaths.append(target)
        for target_dirpath in target_dirpaths:
            self._print(f"-> dataset: {str(target_dirpath)}")
        main_dataset = datasets[0]

        # Skip transfer if cache_splits.json exists
        if (self.directory / "cache_splits.json").exists():
            self._print("cache_splits.json exists, skip transfer.")
            self.status = "finished"
            outputs = {}
            for name, snapshot, target in zip(dataset_names, snapshots, target_dirpaths):
                version = self._state_artifacts.get(name, {}).get("iteration")
                version = f"{version:04d}" if version is not None else self.version
                new_shards = [
                    {"system": (path.parent.name,), "path": path}
                    for path in target.rglob(f"{version}.xyz")
                ]
                outputs[name] = self._write_version_manifest(name, snapshot, new_shards)
            return NamedOutputs(outputs)

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
            dirname = "-".join(part for part in [self.prefix, formula, system_type] if part)

            cache_info = dict(rng_state=main_dataset.rng.bit_generator.state)
            split_structures, split_indices = split_structures_by_ratio(
                curr_structures, self.splits, rng=main_dataset.rng
            )
            cache_info["index"] = [indices.tolist() for indices in split_indices]
            for dataset_name, target_root, target_structures in zip(
                dataset_names, target_dirpaths, split_structures
            ):
                target_subdir = target_root / dirname
                target_subdir.mkdir(parents=True, exist_ok=True)

                num_target_structures = len(target_structures)
                if num_target_structures == 0:
                    self._print(
                        f"-> {dataset_name:<24s} skips {dirname} as it has no structures."
                    )
                else:
                    binding = self._state_artifacts.get(dataset_name)
                    version = (
                        f"{binding['iteration']:04d}" if binding is not None else self.version
                    )
                    transfer_structures_to_dataset(
                        dataset_name,
                        target_root,
                        target_subdir,
                        target_structures,
                        version,
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

        outputs = {}
        for name, snapshot, target in zip(dataset_names, snapshots, target_dirpaths):
            binding = self._state_artifacts.get(name)
            version = f"{binding['iteration']:04d}" if binding is not None else self.version
            new_shards = [
                {"system": (path.parent.name,), "path": path}
                for path in target.rglob(f"{version}.xyz")
            ]
            outputs[name] = self._write_version_manifest(name, snapshot, new_shards)
        return NamedOutputs(outputs)

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
