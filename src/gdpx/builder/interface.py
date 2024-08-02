#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
from typing import List, Union

import omegaconf
from ase import Atoms
from ase.io import read, write

from ..core.operation import Operation
from ..core.register import registers
from ..core.variable import Variable
from ..data.array import AtomsNDArray


@registers.variable.register
class BuilderVariable(Variable):
    """Build structures from the scratch."""

    def __init__(self, directory: Union[str, pathlib.Path] = "./", *args, **kwargs):
        """"""
        # - create a validator
        method = kwargs.get("method", "direct")
        builder = registers.create("builder", method, convert_name=False, **kwargs)

        super().__init__(initial_value=builder, directory=directory)

        return

    @Variable.directory.setter
    def directory(self, directory_) -> None:
        """"""
        self._directory = pathlib.Path(directory_)

        # Value is the attached builder
        self.value.directory = self._directory

        return

    def _reset_random_seed(self, mode: str = "init"):
        """Rewind random state to the one at the initialisation of the object."""
        if mode == "init":
            self.value.set_rng(seed=self.value.init_random_seed)
        elif mode == "zero":
            self.value.set_rng(seed=None)
        else:
            raise RuntimeError(f"INCORRECT RESET RANDOM SEED MODE {mode}.")

        return


@registers.operation.register
class read_stru(Operation):

    def __init__(
        self, fname, format=None, index=":", input_nodes=[], directory="./", **kwargs
    ) -> None:
        """"""
        super().__init__(input_nodes, directory)

        self.fname = fname  # This is broadcastable...
        self.format = format
        self.index = index
        self.kwargs = kwargs

        return

    def forward(self, *args, **kwargs) -> AtomsNDArray:
        """"""
        super().forward()

        # - check params
        if isinstance(self.fname, str):
            fname_ = [self.fname]
        else:  # assume it is an iterable object
            fname_ = self.fname

        # - read structures
        frames = []
        for curr_fname in fname_:
            self._print(f"read {curr_fname}")
            curr_frames = read(
                curr_fname, format=self.format, index=self.index, **self.kwargs
            )
            if isinstance(curr_frames, Atoms):
                curr_frames = [
                    curr_frames
                ]  # if index is single, then read will give Atoms
            frames.extend(curr_frames)

        frames = AtomsNDArray(frames)
        self._print(f"shape of structures: {frames.shape}")

        self.status = "finished"

        return frames


@registers.operation.register
class write_stru(Operation):

    def __init__(
        self, structures, fname=None, format="extxyz", directory="./", *args, **kwargs
    ) -> None:
        """"""
        input_nodes = [structures]
        super().__init__(input_nodes, directory)

        self.fname = fname
        self.format = format
        self.kwargs = kwargs

        return

    def forward(self, structures, *args, **kwargs):
        """"""
        super().forward()

        if isinstance(structures, AtomsNDArray):
            structures = structures.get_marked_structures()

        self._print(f"write structures to {str(self.directory/'structures.xyz')}")
        write(self.directory/"structures.xyz", structures, format=self.format)

        if self.fname is not None:
            fpath = pathlib.Path(self.fname)
            fpath.parent.mkdir(parents=True, exist_ok=True)
            if fpath.exists():
                self._print("remove previous saved_structures")
                fpath.unlink()
            fpath.symlink_to(self.directory/"structures.xyz")
        else:
            ...

        self.status = "finished"

        return


@registers.operation.register
class build(Operation):
    """Build structures without substrate structures."""

    def __init__(self, builder, size: int = 1, directory="./") -> None:
        super().__init__(input_nodes=[builder], directory=directory)

        self.size = size

        return

    @Operation.directory.setter
    def directory(self, directory_) -> None:
        """"""
        self._directory = pathlib.Path(directory_)

        # self.input_nodes[0].directory = self._directory / "builder"

        return

    def forward(self, builder) -> List[Atoms]:
        """"""
        super().forward()

        cache_path = self.directory / f"{builder.name}-out.xyz"
        if not cache_path.exists():
            frames = builder.run(size=self.size)
            write(cache_path, frames)
        else:
            frames = read(cache_path, ":")
        # self._print(f"{builder.name} nframes: {len(frames)}")

        self.status = "finished"

        return frames

    def _preprocess_input_nodes(self, input_nodes):
        """"""
        builder = input_nodes[0]
        if isinstance(builder, dict) or isinstance(
            builder, omegaconf.dictconfig.DictConfig
        ):
            builder = BuilderVariable(directory=self.directory / "builder", **builder)

        return [builder]


@registers.operation.register
class modify(Operation):

    def __init__(
        self, substrates, modifier, size: int = 1, repeat: int = 1, directory="./"
    ) -> None:
        """"""
        super().__init__(input_nodes=[substrates, modifier], directory=directory)

        self.size = size  # create number of new structures
        self.repeat = repeat  # repeat modification times for one structure

        return

    @Operation.directory.setter
    def directory(self, directory_) -> None:
        """"""
        self._directory = pathlib.Path(directory_)

        # self.input_nodes[0].directory = self._directory / "substrates"
        # self.input_nodes[1].directory = self._directory / "modifier"

        return

    def _preprocess_input_nodes(self, input_nodes):
        """"""
        substrates, modifier = input_nodes
        if isinstance(substrates, str) or isinstance(substrates, pathlib.Path):
            # TODO: check if it is a molecule name
            substrates = build(
                BuilderVariable(
                    directory=self.directory / "substrates",
                    method="reader",
                    fname=substrates,
                )
            )
        if isinstance(modifier, dict) or isinstance(
            modifier, omegaconf.dictconfig.DictConfig
        ):
            modifier = BuilderVariable(
                directory=self.directory / "modifier", **modifier
            )

        return substrates, modifier

    def forward(self, substrates: List[Atoms], modifier) -> List[Atoms]:
        """Modify inputs structures.

        A modifier only accepts one structure each time.

        """
        super().forward()

        cache_path = self.directory / f"{modifier.name}-out.xyz"
        if not cache_path.exists():
            frames = modifier.run(substrates, size=self.size)
            write(cache_path, frames)
        else:
            frames = read(cache_path, ":")

        self.status = "finished"

        return frames


if __name__ == "__main__":
    ...
