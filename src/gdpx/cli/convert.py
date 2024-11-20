#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib

from ase.io import read, write

from .. import config
from ..core.register import registers
from ..data.interface import DatasetVariable


def convert_dataset(dataset_path, inp_format: str, out_format: str, directory: pathlib.Path):
    """"""
    kwargs = dict(
        dataset_path=dataset_path,
    )
    ds = registers.create("dataloader", inp_format, convert_name=False, **kwargs)
    config._print(f"{ds =}")

    systems = ds.load_frames()

    if out_format == "multi_xyz":
        for (sys_name, sys_frames) in systems:
            sys_fpath = directory/"converted"/sys_name
            sys_fpath.mkdir(parents=True)
            write(sys_fpath/"converted.xyz", sys_frames)
    else:
        raise RuntimeError(f"Unknown output format `{out_format}`.")

    return


if __name__ == "__main__":
    ...
