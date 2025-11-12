#!/usr/bin/env python3
# -*- coding: utf-8 -*


from typing import Optional

import numpy as np
import torch
from ase import Atoms
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
        atomtype: list[str],
        nn: str = "PES.pt",
        device: str = "cpu",
        dtype: str = "float32",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.device = torch.device(device)

        if dtype == "float32":
            self.dtype = torch.float32
        elif dtype == "float64":
            self.dtype = torch.float64
        else:
            raise Exception("dtype must be float32 or float64")

        self.atomtype = atomtype

        # lazy init
        self._nn_path = nn
        self.pes = None
        self.cutoff = None

        return

    def _init_model(self) -> None:
        """Lazy import some attributes as they not picklable."""
        pes = torch.jit.load(self._nn_path)
        pes.to(self.device).to(self.dtype)
        pes.eval()
        self.cutoff = pes.cutoff
        self.pes = torch.jit.optimize_for_inference(pes)

        return

    def calculate(self, atoms: Optional[Atoms] = None, properties=["energy", "force"], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if self.pes is None:
            self._init_model()

        if atoms is None:
            raise Exception("Atoms object should not be None when calling calculate.")

        i, j, S = neighbor_list("ijS", atoms, cutoff=self.cutoff)
        pairs = torch.from_numpy(np.vstack([i, j])).contiguous().to(self.device).to(torch.long)
        shifts = torch.from_numpy(np.dot(S, atoms.cell)).contiguous().to(self.device).to(self.dtype)

        positions = torch.from_numpy(atoms.positions).contiguous().to(self.device).to(self.dtype)

        symbols = list(self.atoms.symbols)
        species = [self.atomtype.index(i) for i in symbols]
        species = torch.tensor(species, device=self.device, dtype=torch.long)

        energy, force = self.pes(positions, pairs, shifts, species)

        energy = float(energy.detach().cpu().numpy())
        force = force.detach().cpu().numpy()

        self.results["energy"] = energy
        self.results["forces"] = force

        return


if __name__ == "__main__":
    ...
