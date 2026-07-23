#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib

from ase.io import write

from gdpx import config
from gdpx.factory.dataloader import create_dataloader


def convert_dataset(dataset_path, inp_format: str, out_format: str, directory: pathlib.Path):
    """"""
    kwargs = dict(
        dataset_path=dataset_path,
    )
    ds = create_dataloader(dict(name=inp_format, **kwargs))
    config._print(f"{ds =}")

    systems = ds.load_frames()

    if out_format == "multi_xyz":
        for sys_name, sys_frames in systems:
            sys_fpath = directory / "converted" / sys_name
            sys_fpath.mkdir(parents=True)
            write(sys_fpath / "converted.xyz", sys_frames)
    else:
        raise RuntimeError(f"Unknown output format `{out_format}`.")

    return


if __name__ == "__main__":
    ...
