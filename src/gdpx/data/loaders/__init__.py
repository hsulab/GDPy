#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from gdpx.core.register import BaseRegister

REGISTER = BaseRegister("dataloader")

from .singlexyz import SingleXyzDataloader

REGISTER.register("single_xyz")(SingleXyzDataloader)
REGISTER.register(SingleXyzDataloader)

from .dataset import XyzDataloader

REGISTER.register(XyzDataloader)

from gdpx.providers.deepmd.data import DeepmdDataloader

REGISTER.register(DeepmdDataloader)

from gdpx.providers.mace.data import MaceDataloader

REGISTER.register(MaceDataloader)

from gdpx.providers.reann.data import ReannDataloader

REGISTER.register(ReannDataloader)


__all__ = [
    "REGISTER", "SingleXyzDataloader", "XyzDataloader", "DeepmdDataloader",
    "MaceDataloader", "ReannDataloader", "create_dataloader",
]

from .factory import create_dataloader
