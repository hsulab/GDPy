#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import logging
import os
import pathlib
import tempfile

import pytest
import yaml
from ase.io import read, write

from gdpx import config
from gdpx.cli.compute import convert_config_to_potter

#  NOTE: AseDriver must dump the last frame of the trajectory
#        For example, a 9-frame trajectory with a dump period of 3,
#        it must dump step 0, 3, 6, 8 even the last step is not the times of 3.
#        dump_period = 5, it dumps 0, 5, 8.


@pytest.mark.parametrize("dump_period,num_frames", [(1, 24), (3, 9), (5, 6)])
def test_ase_min(dump_period, num_frames):
    """"""
    structures = read("./assets/Pd38_oct.xyz", ":")

    with open("./assets/emtmin.yaml", "r") as fopen:
        worker_params = yaml.safe_load(fopen)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # tmpdirname = f"_xxx_{num_frames}"

        config = copy.deepcopy(worker_params)
        config["driver"]["init"]["dump_period"] = dump_period

        worker = convert_config_to_potter(config)[0]
        worker.directory = tmpdirname

        worker.run(structures)
        results = worker.retrieve(include_retrieved=True)

    assert len(results[-1]) == num_frames

    return


@pytest.mark.parametrize("dump_period,num_frames", [(1, 24), (3, 9), (5, 6)])
def test_ase_nvt(dump_period, num_frames):
    """"""
    structures = read("./assets/Pd38_oct.xyz", ":")

    with open("./assets/emtnvt.yaml", "r") as fopen:
        worker_params = yaml.safe_load(fopen)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # tmpdirname = f"_xxx_{num_frames}"

        config = copy.deepcopy(worker_params)
        config["driver"]["init"]["dump_period"] = dump_period

        worker = convert_config_to_potter(config)[0]
        worker.directory = tmpdirname

        worker.run(structures)
        results = worker.retrieve(include_retrieved=True)

    assert len(results[-1]) == num_frames

    return


@pytest.mark.parametrize("dump_period,ckpt_period", [(1, 1), (1, 3), (3, 5), (3, 7)])
# @pytest.mark.parametrize("dump_period,ckpt_period", [(1, 1)])
def test_ase_min_restart(dump_period, ckpt_period):
    """"""
    structures = read("./assets/Pd38_oct.xyz", ":")

    with open("./assets/emtmin.yaml", "r") as fopen:
        worker_params = yaml.safe_load(fopen)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # tmpdirname = f"_xxx_{dump_period}_{ckpt_period}"
        tmpdirname = pathlib.Path(tmpdirname)

        # run first 24 steps
        config = copy.deepcopy(worker_params)
        config["driver"]["init"]["dump_period"] = dump_period
        config["driver"]["init"]["ckpt_period"] = ckpt_period

        worker = convert_config_to_potter(config)[0]
        worker.directory = tmpdirname
        worker.run(structures)

        #
        os.remove(tmpdirname / "_local_jobs.json")
        config["driver"]["run"]["steps"] = 37
        worker = convert_config_to_potter(config)[0]
        worker.directory = tmpdirname
        worker.run(structures)

        results = worker.retrieve(include_retrieved=True)

    step = results[-1][-1].info["step"]
    energy = results[-1][-1].get_potential_energy()

    assert step == 37
    # assert energy == pytest.approx(16.641920)

    return


@pytest.mark.parametrize("dump_period,ckpt_period", [(1, 1), (1, 3), (3, 5), (3, 7)])
def test_ase_nvt_restart(dump_period, ckpt_period):
    """"""
    structures = read("./assets/Pd38_oct.xyz", ":")

    with open("./assets/emtnvt.yaml", "r") as fopen:
        worker_params = yaml.safe_load(fopen)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # tmpdirname = f"_xxx_{dump_period}_{ckpt_period}"
        tmpdirname = pathlib.Path(tmpdirname)

        # run first 24 steps
        config = copy.deepcopy(worker_params)
        config["driver"]["init"]["dump_period"] = dump_period
        config["driver"]["init"]["ckpt_period"] = ckpt_period

        worker = convert_config_to_potter(config)[0]
        worker.directory = tmpdirname
        worker.run(structures)

        #
        os.remove(tmpdirname / "_local_jobs.json")
        config["driver"]["run"]["steps"] = 37
        worker = convert_config_to_potter(config)[0]
        worker.directory = tmpdirname
        worker.run(structures)

        results = worker.retrieve(include_retrieved=True)

    step = results[-1][-1].info["step"]
    energy = results[-1][-1].get_potential_energy()

    assert step == 37
    assert energy == pytest.approx(16.641920)

    return


if __name__ == "__main__":
    ...
