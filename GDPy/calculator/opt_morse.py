#!/usr/bin/env python3
# -*- coding: utf-8 -*

import numpy as np

import torch

from ase.io import read, write
from ase.neighborlist import neighbor_list

class DataLoader():
    def __init__(
        self,
        energies,  # properties, energy
        positions,  # coordinates (ntrains,(max_natoms,3))
        # neighbours
        pairs,  # (2, nframes, max_natoms*nneighs)
        shifts,  # (nframes, max_natoms*nneighs, 3)
        batchsize,
    ):
        # print(image)
        self.energies = energies
        self.positions = positions
        self.pairs = pairs
        self.shifts = shifts

        # calculate the initial and the final index of the current process
        self.batchsize = batchsize
        dim = positions.shape[0]  # number of structures
        # neglect the last batch that less than the batchsize
        self.end = dim - self.batchsize + 1
        self.length = int(np.ceil(self.end/self.batchsize))

        return

    def __len__(self):
        return self.length

    def __iter__(self):
        # reset self.ipoint when starts iteration
        self.ipoint = 0
        return self

    def __next__(self):
        if self.ipoint < self.end:
            index_batch = range(self.ipoint, self.ipoint+self.batchsize)

            # use indexing
            cur_energies = self.energies[index_batch]
            cur_positions = self.positions[index_batch]
            cur_pairs = self.pairs[index_batch]
            cur_shifts = self.shifts[index_batch]

            self.ipoint += self.batchsize
            return cur_energies, cur_positions, cur_pairs, cur_shifts
        else:
            raise StopIteration

class Loss(torch.nn.Module):
   def __init__(self):
      super(Loss, self).__init__()

   def forward(
      self, 
      reference,
      prediction
   ):
      #return torch.cat([self.loss_fn(ivar,iab).view(-1) for ivar, iab in zip(var,ab)])
      return torch.nn.MSELoss()(reference, prediction)

def extract_data(atoms, cutoff=6.0):
    i, j, d, D, S = neighbor_list("ijdDS", atoms, cutoff)

    # prepare inputs
    positions = atoms.positions
    neigh_list = np.vstack([i, j])
    shifts = np.dot(S, atoms.cell)

    return atoms.get_potential_energy(), positions, neigh_list, shifts

frames = read("O2-dimer.xyz", "4:9")

energies, positions, neigh_list, shifts = [], [], [], []
for atoms in frames:
    data = extract_data(atoms)
    energies.append(data[0])
    positions.append(data[1])
    neigh_list.append(data[2])
    shifts.append(data[3])
energies = torch.from_numpy(np.array(energies))
positions = torch.from_numpy(np.array(positions))
neigh_list = torch.from_numpy(np.array(neigh_list))
shifts = torch.from_numpy(np.array(shifts))

data_train = DataLoader(energies, positions, neigh_list, shifts, batchsize=1)

from Morse import Morse
model = Morse()
model.train()

optim = torch.optim.AdamW(
    model.parameters(),
    lr = 1e-1, weight_decay=0.0
)
#for p in model.parameters():
#    print(p)

LossFn = Loss()

for data in data_train:
    en, pos, pair, shi = data
    ret = model(pos[0], pair[0], shi[0])
    loss = LossFn(en[0], ret[0])
    # print("=== LOSS === ", loss.detach().numpy())
    optim.zero_grad()
    #optim.zero_grad(set_to_none=True)
    loss.backward()
    #loss.backward(retain_graph=True)
    optim.step()

for p in model.named_parameters():
    print(p)

model.eval()
ret = model(pos[0], pair[0], shi[0])
print(ret)

if __name__ == "__main__":
    pass