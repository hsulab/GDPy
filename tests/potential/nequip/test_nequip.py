#!/usr/bin/env python3
# -*- coding: utf-8 -*

try:
    import torch
    from nequip.ase import NequIPCalculator
except:
    ...


from ase.io import read, write

from GDPy.computation.mixer import CommitteeCalculator

atypes = ["C", "H", "N", "O", "S"]
models = [
    "/mnt/scratch2/users/40247882/porous/nqtrain/r0/_ensemble/0004.train/m0/nequip.pth",
    "/mnt/scratch2/users/40247882/porous/nqtrain/r0/_ensemble/0004.train/m1/nequip.pth",
    "/mnt/scratch2/users/40247882/porous/nqtrain/r0/_ensemble/0004.train/m2/nequip.pth",
    "/mnt/scratch2/users/40247882/porous/nqtrain/r0/_ensemble/0004.train/m3/nequip.pth",
]

calcs = []
for m in models:
    curr_calc = NequIPCalculator.from_deployed_model(
        model_path=m,
        species_to_type_name={k: k for k in atypes},
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    calcs.append(curr_calc)

calc = CommitteeCalculator(calcs)

atoms = read("/mnt/scratch2/users/40247882/porous/init/structures/methanol.xyz")

atoms.calc = calc

# print(atoms.get_potential_energy())
print(atoms.get_forces())
print(atoms.calc.results)


if __name__ == "__main__":
    ...
