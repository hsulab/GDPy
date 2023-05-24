#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from GDPy.builder.dimer import DimerBuilder


class TestDimerBuilder:

    def test_dimer(self):
        """"""
        elements = ["H", "H"]
        distances = [0.6, 1.0, 0.05]

        builder = DimerBuilder(elements, distances)
        frames = builder.run()

        nframes = len(frames)

        assert nframes == 9


if __name__ == "__main__":
    ...