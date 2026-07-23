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


# Give tests a deterministic package-level generator. Individual tests must
# import the components they exercise; the suite must not globally bootstrap
# every plugin and workflow node as an import side effect.
from gdpx import config

config.logger.setLevel(logging.DEBUG)
config.GRNG = np.random.Generator(np.random.PCG64())


if __name__ == "__main__":
    ...
