#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib

from ..core.register import registers

from .correction import correct
registers.operation.register(correct)

from .convert import convert_dataset

from .dsformat.singlexyz import SingleXyzDataloader
registers.dataloader.register("single_xyz")(SingleXyzDataloader)



if __name__ == "__main__":
    ...
