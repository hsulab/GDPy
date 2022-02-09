#!/usr/bin/env python3
# -*- coding: utf-8 -*

from math import dist
import numpy as np
import torch
from torch import optim

class Morse(torch.nn.Module):

    """
    morse potential with decayed gaussian radial functions
    """

    def __init__(self, epsilon=1.0, alpha=6.0, r0=1.0, rcut1=1.9, rcut2=2.7):
        super(Morse, self).__init__()

        # self.register_buffer("alpha", torch.Tensor([alpha])) # prefactor
        # self.register_buffer("epsilon", torch.Tensor([epsilon])) # potential depth
        # self.register_buffer("r0", torch.Tensor([r0])) # equalibrium distance

        self.alpha = torch.nn.parameter.Parameter(torch.Tensor([alpha]))
        self.epsilon = torch.nn.parameter.Parameter(torch.Tensor([epsilon]))
        self.r0 = torch.nn.parameter.Parameter(torch.Tensor([r0]))

        # cutoffs
        self.register_buffer("rcut1", 1.0*torch.Tensor([rcut1])) 
        self.register_buffer("rcut2", 1.0*torch.Tensor([rcut2])) 

        return
    
    def cutoff(self, distances):
        s = 1.0 - (distances - self.rcut1) / (self.rcut2 - self.rcut1)

        return (s >= 1.0) + (((0.0 < s) & (s < 1.0)) * (6.0 * s**5 - 15.0 * s**4 + 10.0 * s**3))
        #return (6.0 * s**5 - 15.0 * s**4 + 10.0 * s**3)
    
    def forward(
        self, 
        positions, # cartesian coordinates
        neigh_list, # shape (nparis, 2), centre i neighbour j
        shifts # shape (npairs, 3), in cartesian
    ):
        """"""
        # calculate distances
        #positions = positions.detach().clone()
        #positions.requires_grad_(True)
        positions.requires_grad = True

        # natoms = positions.shape[0]
        selected_positions = positions.index_select(
            0, neigh_list.view(-1)
        ).view(2, -1, 3)
        dvectors = selected_positions[1] - selected_positions[0] + shifts
        distances = torch.linalg.norm(dvectors, dim=-1)

        expf = torch.exp(self.alpha * (1.0 - distances/self.r0))
        fc = self.cutoff(distances)
        energy = 0.5*torch.sum(self.epsilon * expf * (expf - 2) * fc)

        #forces = -torch.autograd.grad([energy, ], [positions, ])[0]
        #return energy, forces
        return energy,

if __name__ == "__main__":
    # shared parameters
    eq_r0 = 2.2

    # ===== test ASE Morse =====
    from ase import Atoms
    from ase.build import bulk
    from ase.calculators.morse import MorsePotential
    #atoms = Atoms(
    #    "O2", positions=[[5., 5., 5.],[5.,5.,6.]],
    #    cell = 10.*np.eye(3), pbc=True
    #)
    #atoms = bulk("Cu", "fcc", a=2.0, cubic=True)
    atoms = Atoms(
        "Cu4", positions=[
            [0.,0.,0.], [0.,2.0,1.8],
            [1.8,0.,1.8], [1.8,1.8,0.]
        ],
        cell = 3.6*np.eye(3), pbc=True
    )
    calc = MorsePotential(r0=eq_r0)
    atoms.calc = calc
    print(atoms.get_potential_energy())
    print(atoms.get_forces())

    # ===== test torch Morse =====
    from ase.neighborlist import neighbor_list
    i, j, d, D, S = neighbor_list("ijdDS", atoms, 2.7*eq_r0)

    # prepare inputs
    positions = torch.from_numpy(atoms.positions)
    neigh_list = torch.from_numpy(np.vstack([i, j]))
    shifts = torch.from_numpy(np.dot(S, atoms.cell))

    calc = Morse(r0=eq_r0)
    # calc.half() # convert to half precision
    energy, forces = calc(positions, neigh_list, shifts)
    print(energy.detach().numpy())
    print(forces.detach().numpy())
