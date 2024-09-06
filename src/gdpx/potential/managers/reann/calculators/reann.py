#!/usr/bin/env python3
# -*- coding: utf-8 -*


import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list


"""Reann Calculator.

This is the same as the official one. However, we use a lazy initialisation of 
model here, which avoids serialisation problem when copying this calculator.

"""


class REANN(Calculator):

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        atomtype,
        nn="PES.pt",
        device="cpu",
        dtype=torch.float32,
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.atomtype = atomtype

        # lazy init
        self._nn_path = nn
        self.pes = None
        self.cutoff = None

        self.table = 0

        return

    def _init_model(self) -> None:
        """Lazy import some attributes as they not picklable."""
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

        positions = (
            torch.from_numpy(atoms.positions)
            .contiguous()
            .to(self.device)
            .to(self.dtype)
        )

        i, j, S = neighbor_list("ijS", atoms, cutoff=self.cutoff)
        pairs = (
            torch.from_numpy(np.vstack([i, j]))
            .contiguous()
            .to(self.device)
            .to(torch.long)
        )
        shifts = (
            torch.from_numpy(np.dot(S, atoms.cell))
            .contiguous()
            .to(self.device)
            .to(self.dtype)
        )

        symbols = list(self.atoms.symbols)
        species = [self.atomtype.index(i) for i in symbols]
        species = torch.tensor(species, device=self.device, dtype=torch.long)
        energy, force = self.pes(positions, pairs, shifts, species)

        energy = float(energy.detach().cpu().numpy())
        force = force.detach().cpu().numpy()

        self.results["energy"] = energy
        self.results["forces"] = force.copy()

        return


if __name__ == "__main__":
    ...
