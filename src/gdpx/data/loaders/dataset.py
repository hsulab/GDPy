import copy
import pathlib
import traceback
from typing import Optional, Union

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.formula import Formula
from ase.io import read

from gdpx.core.component import BaseComponent
from gdpx.data.interfaces import DatasetSource

from .utils import get_composition_from_system_tree, is_a_valid_system_name

#: How to map keys in structures.
DEFAULT_PROP_MAP_KEYS: list[tuple[str, str]] = [
    ("energy", "energy"),
    ("forces", "forces"),
]


def map_atoms_data(atoms: Atoms, prop_map_keys) -> None:
    """"""
    assert type(atoms.calc) == SinglePointCalculator, f"Atoms {atoms} has {atoms.calc}."

    # For some small dataset, structures can be used both as train and test,
    # thus, we need add a flag to avoid repeated mapping.
    is_mapped = atoms.info.get("is_mapped", False)

    if not is_mapped:
        results = copy.deepcopy(atoms.calc.results)
        for mapping_pairs in prop_map_keys:
            dst_key, src_key = mapping_pairs
            if src_key in results:
                # NOTE: The dst_data is overwritten if it exists
                src_val = results.pop(src_key)
                results[dst_key] = src_val
            else:
                raise KeyError(f"Atoms {atoms} has no key {src_key}.")
        atoms.calc.results = results
        atoms.info["is_mapped"] = True
    else:
        ...

    return


def traverse_xyzdirs(wdir):
    """"""
    data_dirs = []

    def recursive_traverse(wdir):
        for p in wdir.iterdir():
            if p.is_dir():
                xyzpaths = list(p.glob("*.xyz"))
                if len(xyzpaths) > 0:
                    data_dirs.append(p)
                recursive_traverse(p)
            else:
                ...
        return

    recursive_traverse(wdir)

    return data_dirs


def split_train_and_test_into_batches(num_frames: int, batchsize: int, train_ratio: float, rng):
    """"""
    # TODO: adjust batchsize of train and test separately
    if num_frames <= batchsize:
        # NOTE: use same train and test set
        #       since they are very important structures...
        if num_frames == 1 or batchsize == 1:
            new_batchsize = 1
        else:
            new_batchsize = int(2 ** np.floor(np.log2(num_frames)))
        train_index = list(range(num_frames))
        test_index = []
    else:
        if num_frames == 1 or batchsize == 1:
            new_batchsize = 1
            train_index = list(range(num_frames))
            test_index = []
        else:
            new_batchsize = batchsize
            # - assure there is at least one batch for test
            #          and number of train frames is integer times of batchsize
            if (1.0 - train_ratio) > 1e-4:
                ntrain = int(np.floor(num_frames * train_ratio / new_batchsize) * new_batchsize)
                if ntrain > 0:
                    train_index = rng.choice(num_frames, ntrain, replace=False)
                    test_index = [x for x in range(num_frames) if x not in train_index]
                else:
                    train_index = list(range(num_frames))
                    test_index = list(range(num_frames))
            else:
                train_index = list(range(num_frames))
                test_index = []

    return new_batchsize, train_index, test_index


def parse_batchsize_setting(batchsize: Union[int, str], num_atoms: int) -> int:
    """"""
    if isinstance(batchsize, int):
        new_batchsize = batchsize
    elif isinstance(batchsize, str):  # must be a string
        method, number = batchsize.split(":")
        if method == "n_structures":
            new_batchsize = int(number)
        elif method == "n_atoms":
            new_batchsize = int(2 ** np.floor(np.log2(int(number) / num_atoms)))
            if new_batchsize < 1:
                new_batchsize = 1
        else:
            raise RuntimeError(f"Improper batchsize `{batchsize}.`")
    else:
        raise RuntimeError(f"Improper batchsize `{batchsize}.`")

    return new_batchsize


class AbstractDataloader(BaseComponent, DatasetSource): ...


class XyzDataloader(AbstractDataloader):
    name = "xyz"

    """A directory-based dataset.

    There are several subdirs in the main directory. Each dirname follows the format that 
    `description-formula-type`, for example, `water-H2O-molecule`, is a system with structures 
    that have one single water molecule.

    """

    def __init__(
        self,
        dataset_path: Union[str, pathlib.Path] = "./",
        batchsize: Union[int, str] = 32,
        train_ratio: float = 0.9,
        random_seed: Optional[int] = None,
        prop_keys: list[tuple[str, str]] = DEFAULT_PROP_MAP_KEYS,
    ) -> None:
        """"""
        super().__init__(directory=dataset_path, random_seed=random_seed)

        self.batchsize = batchsize
        self.train_ratio = train_ratio

        self.prop_keys = prop_keys

        return

    def load(self) -> list[pathlib.Path]:
        """Load dataset.

        All directories that have xyz files in `self.directory`.

        TODO:
            * Other file formats.

        """
        data_dirs = traverse_xyzdirs(self.directory)
        data_dirs = sorted(data_dirs)

        return data_dirs

    def _relative_parts(self, path: pathlib.Path) -> tuple[str, ...]:
        return tuple(path.relative_to(self.directory).parts)

    def _system_files(self) -> list[tuple[tuple[str, ...], list[pathlib.Path]]]:
        """Return exact XYZ shards grouped by their dataset system."""
        system_groups: dict[tuple[str, ...], list[pathlib.Path]] = {}
        for data_dir in self.load():
            system_path = None
            for parent in (data_dir, *data_dir.parents):
                if is_a_valid_system_name(parent.name):
                    system_path = parent
                    break
            if system_path is None:
                raise RuntimeError(f"No system folder found in `{data_dir}`")
            key = self._relative_parts(system_path)
            system_groups.setdefault(key, []).extend(sorted(data_dir.glob("*.xyz")))
        return sorted(system_groups.items())

    def load_frames(self):
        """"""
        system_files = self._system_files()

        nframes_tot, pairs = 0, []
        for i, (name, xyzpaths) in enumerate(system_files):
            curr_frames = []
            for x in xyzpaths:
                curr_frames.extend(read(x, ":"))
            curr_nframes = len(curr_frames)
            nframes_tot += curr_nframes
            self._debug(f"{i:>4d} {'/'.join(name)} -> {len(curr_frames)}")
            pairs.append([name, curr_frames])
        self._debug(f"Number of frames: {nframes_tot}")

        # - map keys
        should_map_keys = False
        for mapping_pairs in self.prop_keys:
            dst_key, src_key = mapping_pairs
            if dst_key != src_key:
                should_map_keys = True
            else:
                ...
        else:
            ...
        if should_map_keys:
            for n, x in pairs:
                for a in x:
                    map_atoms_data(a, self.prop_keys)

        # convert pairs to mapped
        mapped_data = {p: f for p, f in pairs}

        return mapped_data

    def split_train_and_test(
        self,
    ):
        """Read structures and split them into train and test."""
        self._print("--- auto data reader ---")
        system_groups = self._system_files()
        self._debug(system_groups)

        # Check batchsize
        batchsizes = self.batchsize
        nsystems = len(system_groups)
        if isinstance(batchsizes, int) or isinstance(batchsizes, str):
            batchsizes = [batchsizes] * nsystems
        else:
            ...  # assume self.batchsize is a list
        assert len(batchsizes) == nsystems, "Number of systems and batchsizes are inconsistent."

        # Load configurations
        set_names = []
        train_size, test_size = [], []
        train_frames, test_frames = [], []
        adjusted_batchsizes = []  # auto-adjust batchsize based on nframes
        accumulated_batches = 0
        for _, (curr_system_group, curr_batchsize) in enumerate(zip(system_groups, batchsizes)):
            set_tree = list(curr_system_group[0])
            set_name = "+".join(set_tree)
            set_names.append(set_name)
            try:
                composition = get_composition_from_system_tree(set_tree)
            except Exception:
                self._print(traceback.format_exc())
                self._print(f"{set_name =}")
                raise RuntimeError()

            # convert batchsize to an integer
            try:
                num_atoms = sum(Formula(composition).count().values())
            except Exception:
                self._print(traceback.format_exc())
                self._print(f"{composition =}")
                raise RuntimeError()

            curr_batchsize = parse_batchsize_setting(curr_batchsize, num_atoms)

            self._print(f"System {set_name}")
            self._print(f"  {composition=}  batchsize={curr_batchsize}")
            frames = []  # all frames in this subsystem
            for p in curr_system_group[1]:
                p_frames = read(p, ":")
                p_nframes = len(p_frames)
                frames.extend(p_frames)
                self._print(f"    shard: {p.name} number {p_nframes}")

            # split dataset and get adjusted batchsize
            num_frames = len(frames)
            new_batchsize, train_index, test_index = split_train_and_test_into_batches(
                num_frames, curr_batchsize, self.train_ratio, self.rng
            )

            adjusted_batchsizes.append(new_batchsize)

            ntrain, ntest = len(train_index), len(test_index)
            train_size.append(ntrain)
            test_size.append(ntest)

            num_batches_train = int(np.ceil(ntrain / new_batchsize))
            accumulated_batches += num_batches_train

            self._print(f"    ntrain: {ntrain} ntest: {ntest} ntotal: {num_frames}")
            self._print(f"    batchsize: {new_batchsize} batches: {num_batches_train}")
            assert ntrain > 0

            curr_train_frames = [frames[train_i] for train_i in train_index]
            curr_test_frames = [frames[test_i] for test_i in test_index]

            # train
            train_frames.append(curr_train_frames)
            n_train_frames = sum([len(x) for x in train_frames])

            # test
            test_frames.append(curr_test_frames)
            n_test_frames = sum([len(x) for x in test_frames])
            self._print(f"  Current Dataset -> ntrain: {n_train_frames} ntest: {n_test_frames}")

        assert len(train_size) == len(test_size), "inconsistent train_size and test_size"
        train_size = sum(train_size)
        test_size = sum(test_size)
        self._print(f"Total Dataset -> ntrain: {train_size} ntest: {test_size} nbatches: {accumulated_batches}")

        if train_size == 0:
            raise Exception("The dataset must have at least one structure.")

        # Map property keys
        should_map_keys = False
        for mapping_pairs in self.prop_keys:
            dst_key, src_key = mapping_pairs
            if dst_key != src_key:
                should_map_keys = True
            else:
                ...
        else:
            ...

        if should_map_keys:
            for curr_frames in train_frames:
                for a in curr_frames:
                    map_atoms_data(a, self.prop_keys)
            for curr_frames in test_frames:
                for a in curr_frames:
                    map_atoms_data(a, self.prop_keys)

        return set_names, train_frames, test_frames, adjusted_batchsizes

    def as_dict(self):
        """"""
        dataset_params = {}
        dataset_params["name"] = self.name
        dataset_params["dataset_path"] = str(self.directory.resolve())
        dataset_params["batchsize"] = self.batchsize
        dataset_params["train_ratio"] = self.train_ratio

        dataset_params = copy.deepcopy(dataset_params)

        return dataset_params


class XyzSnapshotDataloader(XyzDataloader):
    """Read-only composition of an initial XYZ dataset and iteration deltas."""

    name = "xyz_snapshot"

    def __init__(self, sources=None, shards=None, manifest=None, **kwargs):
        paths = tuple(pathlib.Path(path).resolve() for path in (sources or ()))
        entries = []
        for source in paths:
            loader = XyzDataloader(source, **kwargs)
            for system, files in loader._system_files():
                entries.extend({"system": system, "path": path.resolve()} for path in files)
        entries.extend(
            [
                {
                    "system": tuple(item["system"]),
                    "path": pathlib.Path(item["path"]).resolve(),
                }
                for item in (shards or ())
            ]
        )
        if not entries and not paths:
            raise ValueError("An XYZ dataset snapshot requires at least one source or shard.")
        self.shards = tuple(
            {item["path"]: item for item in entries}.values()
        )
        self.sources = paths
        self.manifest = pathlib.Path(manifest).resolve() if manifest else None
        super().__init__(dataset_path=paths[0] if paths else entries[0]["path"].parent, **kwargs)

    @classmethod
    def from_loader(cls, loader: XyzDataloader) -> "XyzSnapshotDataloader":
        if isinstance(loader, cls):
            return loader
        if not isinstance(loader, XyzDataloader):
            raise TypeError(
                f"Immutable dataset transfer currently supports XYZ datasets, got {type(loader).__name__}."
            )
        return cls(
            sources=[loader.directory],
            batchsize=loader.batchsize,
            train_ratio=loader.train_ratio,
            random_seed=loader.random_seed,
            prop_keys=loader.prop_keys,
        )

    def extend(self, source: str | pathlib.Path) -> "XyzSnapshotDataloader":
        source = pathlib.Path(source).resolve()
        loader = XyzDataloader(source)
        shards = list(self.shards)
        for system, files in loader._system_files():
            shards.extend({"system": system, "path": path} for path in files)
        return type(self)(
            sources=(*self.sources, source),
            shards=shards,
            batchsize=self.batchsize,
            train_ratio=self.train_ratio,
            random_seed=self.random_seed,
            prop_keys=self.prop_keys,
        )

    def extend_shards(self, shards, manifest=None) -> "XyzSnapshotDataloader":
        return type(self)(
            sources=self.sources,
            shards=[*self.shards, *shards],
            manifest=manifest,
            batchsize=self.batchsize,
            train_ratio=self.train_ratio,
            random_seed=self.random_seed,
            prop_keys=self.prop_keys,
        )

    def load(self) -> list[pathlib.Path]:
        return sorted({item["path"].parent for item in self.shards if item["path"].exists()})

    def _system_files(self) -> list[tuple[tuple[str, ...], list[pathlib.Path]]]:
        groups: dict[tuple[str, ...], list[pathlib.Path]] = {}
        for item in self.shards:
            groups.setdefault(item["system"], []).append(item["path"])
        return [(key, sorted(paths)) for key, paths in sorted(groups.items())]

    def _relative_parts(self, path: pathlib.Path) -> tuple[str, ...]:
        resolved = path.resolve()
        for source in self.sources:
            try:
                return tuple(resolved.relative_to(source).parts)
            except ValueError:
                continue
        raise ValueError(f"Dataset path {path} is outside snapshot sources.")

    def as_dict(self):
        return {
            "name": self.name,
            "sources": [str(path) for path in self.sources],
            "shards": [
                {"system": list(item["system"]), "path": str(item["path"])}
                for item in self.shards
            ],
            "manifest": str(self.manifest) if self.manifest else None,
            "batchsize": self.batchsize,
            "train_ratio": self.train_ratio,
            "random_seed": self.random_seed,
            "prop_keys": self.prop_keys,
        }
