#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from gdpx.core.register import BaseRegister

REGISTER = BaseRegister("dataloader")

from .singlexyz import SingleXyzDataloader

REGISTER.register("single_xyz")(SingleXyzDataloader)

from .dataset import XyzDataloader

REGISTER.register(XyzDataloader)

from .deepmd import DeepmdDataloader

REGISTER.register(DeepmdDataloader)

from .mace import MaceDataloader

REGISTER.register(MaceDataloader)

from .reann import ReannDataloader

REGISTER.register(ReannDataloader)


if __name__ == "__main__":
    ...
