#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from gdpx.core.register import BaseRegister

REGISTER = BaseRegister("dataloader")

from .singlexyz import SingleXyzDataloader

REGISTER.register("single_xyz")(SingleXyzDataloader)

from .dataset import XyzDataloader

REGISTER.register(XyzDataloader)


if __name__ == "__main__":
    ...
