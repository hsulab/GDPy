#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import logging

import numpy as np

import pytest


@pytest.fixture(autouse=True)
def change_test_dir(request):
    os.chdir(request.fspath.dirname)
    yield
    os.chdir(request.config.invocation_params.dir)

    return


# NOTE: We assigan a random_seed and register all necessary modules here!!
from gdpx import config

config.logger.setLevel(logging.DEBUG)
config.GRNG = np.random.Generator(np.random.PCG64())

from gdpx.core.register import import_all_modules_for_register

import_all_modules_for_register()


if __name__ == "__main__":
    ...
