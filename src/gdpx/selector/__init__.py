#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from ..core.register import registers

from .basin import BasinSelector
registers.selector.register(BasinSelector)

from .compare import CompareSelector
registers.selector.register(CompareSelector)

from .random import RandomSelector
registers.selector.register(RandomSelector)


if __name__ == "__main__":
    ...