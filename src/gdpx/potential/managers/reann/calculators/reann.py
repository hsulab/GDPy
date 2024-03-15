#!/usr/bin/env python3
# -*- coding: utf-8 -*

import numpy as np
import os
import torch
import re

# from gpu_sel import *
from ase.data import chemical_symbols, atomic_numbers
from ase.units import Bohr
from ase.calculators.calculator import (
    Calculator,
    all_changes,
    PropertyNotImplementedError,
)

"""Reann Calculator.

This is the same as the official one. However, we use a lazy initialisation of 
model here, which avoids serialisation problem when copying this calculator.

"""


class REANN(Calculator):

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        atomtype,
        maxneigh,
        getneigh,
        nn="PES.pt",
        device="cpu",
        dtype=torch.float32,
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.atomtype = atomtype
        self.maxneigh = maxneigh
        self.getneigh = getneigh

        # - lazy init
        self._nn_path = nn
        self.pes = None
        self.cutoff = None

        self.table = 0
        # self.pes=torch.compile(pes)

    def _init_model(self) -> None:
        """"""
        pes = torch.jit.load(self._nn_path)
        pes.to(self.device).to(self.dtype)
        pes.eval()
        self.cutoff = pes.cutoff
        self.pes = torch.jit.optimize_for_inference(pes)

        return

    def calculate(
        self, atoms=None, properties=["energy", "force"], system_changes=all_changes
    ):
        Calculator.calculate(self, atoms, properties, system_changes)
        if self.pes is None:
            self._init_model()

        # TODO: To re-use the same calculator, we init neigh_list every time
        cell = np.array(self.atoms.cell)
        #if self.table > 0.5 and "cell" in system_changes:
        #    self.getneigh.deallocate_all()
        #if "cell" in system_changes:
        #    if cell.ndim == 1:
        #        cell = np.diag(cell)
        #    self.getneigh.init_neigh(self.cutoff, self.cutoff / 2.0, cell.T)
        #    self.table += 1
        if self.getneigh.initmod.rc > 0.:
            self.getneigh.deallocate_all()

        if cell.ndim == 1:
            cell = np.diag(cell)
        self.getneigh.init_neigh(self.cutoff, self.cutoff / 2.0, cell.T)
        self.table += 1

        icart = self.atoms.get_positions()
        cart, neighlist, shiftimage, scutnum = self.getneigh.get_neigh(
            icart.T, self.maxneigh
        )
        cart = torch.from_numpy(cart.T).contiguous().to(self.device).to(self.dtype)
        neighlist = (
            torch.from_numpy(neighlist[:, :scutnum])
            .contiguous()
            .to(self.device)
            .to(torch.long)
        )
        shifts = (
            torch.from_numpy(shiftimage.T[:scutnum, :])
            .contiguous()
            .to(self.device)
            .to(self.dtype)
        )
        symbols = list(self.atoms.symbols)
        species = [self.atomtype.index(i) for i in symbols]
        species = torch.tensor(species, device=self.device, dtype=torch.long)
        energy, force = self.pes(cart, neighlist, shifts, species)
        energy = float(energy.detach().numpy())
        self.results["energy"] = energy
        force = force.detach().numpy()
        self.results["forces"] = force.copy()


if __name__ == "__main__":
    ...
