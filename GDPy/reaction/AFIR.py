#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Artificial force induced reaction (AFIR)
"""

from itertools import product, combinations

#import numpy as np
from jax import numpy as np
from jax import grad, jit

from ase.data import covalent_radii
from ase.neighborlist import neighbor_list, natural_cutoffs, NeighborList
from ase.optimize import BFGS
from ase.calculators.singlepoint import SinglePointCalculator


def grid_iterator(grid):
    """Yield all of the coordinates in a 3D grid as tuples

    Args:
        grid (tuple[int] or int): The grid dimension(s) to
                                  iterate over (x or (x, y, z))

    Yields:
        tuple: (x, y, z) coordinates
    """
    if isinstance(grid, int): # Expand to 3D grid
        grid = (grid, grid, grid)

    for x in range(-grid[0], grid[0]+1):
        for y in range(-grid[1], grid[1]+1):
            for z in range(-grid[2], grid[2]+1):
                yield (x, y, z)

def partition_fragments(creator, atoms):
    """ generate fragments from a single component
        use graph to find adsorbates and group them into fragments
    """
    creator.generate_graph(atoms)

    chem_envs = creator.extract_chem_envs()

    fragments = []
    for adsorbate in chem_envs:
        ads_indices = []
        for (u, d) in adsorbate.nodes.data():
            #print(u, d)
            if d["central_ads"] and d.get("ads"):
                ads_indices.append(d["index"])
        ads_indices.sort()
        fragments.append(ads_indices)

    #print("nadsorbates: ", len(chem_envs))
    assert len(chem_envs) == len(fragments), "fragment number inconsistent"

    return fragments

@jit
def force_function(
    positions, covalent_radii, pair_indices, 
    pair_shifts, # in cartesian, AA
    gamma=2.5
):
    """ AFIR function
    """
    bias = 0.0
    # collision coef
    r0 = 3.8164 # Ar-Ar LJ
    epsilon = 1.0061 / 96.485
    alpha = gamma/((2**(-1/6)-(1+(1+gamma/epsilon)**0.5)**(-1/6))*r0)

    # inverse distance weights
    pair_positions = np.take(positions, pair_indices, axis=0)
    #print(pair_positions.shape)
    dvecs = pair_positions[0] - pair_positions[1] + pair_shifts
    #print(dvecs)
    distances = np.linalg.norm(dvecs, axis=1)
    #print(distances)

    pair_radii = np.take(covalent_radii, pair_indices, axis=0)
    #print(pair_radii)
    #print((pair_radii[0]+pair_radii[1]))

    weights = ((pair_radii[0]+pair_radii[1])/distances)**6
    #print(weights)

    bias = alpha * np.sum(weights*distances) / np.sum(weights)
    #print(bias)

    return bias

class AFIR(BFGS):

    has_reaction = False

    def __init__(self, atoms, gamma, frag_indices, graph_creator=None, **kwargs):
        """"""
        super().__init__(atoms, **kwargs)

        self.graph_creator = graph_creator

        if self.graph_creator is not None:
            self.initial_fragments = partition_fragments(self.graph_creator, atoms)
        else:
            self.initial_fragments = frag_indices

        self.gamma = gamma # model collision parameter

        self.dfn = grad(force_function, argnums=0)

        # init atoms related info
        atomic_numbers = self.atoms.get_atomic_numbers()
        self.atomic_radii = np.array([covalent_radii[i] for i in atomic_numbers])

        # set fragments
        #self.pair_indices = np.array(pair_indices)
        #if pair_shifts is None:
        #    # indicates that no pbc condition is considered
        #    self.pair_shifts = np.zeros((self.pair_indices.shape[1],3))
        #else:
        #    self.pair_shifts = pair_shifts
        self.frag_indices = frag_indices
        assert len(self.frag_indices) == 2, "Number of fragments should be two."

        self.nl = NeighborList(
            cutoffs=[10.]*len(atoms), skin=0.3, self_interaction=False,
            bothways=True
        )
        
        self.grids = list(grid_iterator(1))

        # init biased forces
        # NOTE: set to inf, avoid convergence at first step
        self.biased_forces = np.zeros(self.atoms.positions.shape) + np.inf

        return
    
    def calc_pairs(self):
        # calc pair info
        self.nl.update(atoms)

        pair_info = {}
        for i in self.frag_indices[0]:
            nei_indices, offsets = self.nl.get_neighbors(i)
            for j, offset in zip(nei_indices, offsets):
                if j in self.frag_indices[1]:
                    distance = atoms.positions[i] - atoms.positions[j] + np.dot(offset, atoms.get_cell(complete=True))
                    key = f"{i}-{j}"
                    prev = pair_info.get(key, None)
                    if prev:
                        if distance < prev[0]:
                            pair_info[key] = [distance, offset]
                    else:
                        pair_info[key] = [distance, offset]
        print(pair_info)
        pair_indices, pair_shifts = [], []

        return
    
    def update_positions(self, atoms):
        """"""
        f1_indices, f2_indices = self.frag_indices
        
        # frag 1
        cell = atoms.get_cell(complete=True)[:]
        pos1 = atoms.positions[f1_indices[0]]
        for x in (f1_indices[1:]+f2_indices[:1]):
            mindis, minshift, posx = np.inf, None, None
            for shift in self.grids:
                shift = np.array(shift)
                cur_posx = atoms.positions[x] + np.dot(shift, cell)
                dis = np.linalg.norm(cur_posx - pos1)
                if dis < mindis:
                    mindis, minshift, posx = dis, shift, cur_posx
            atoms.positions[x] = posx
        pos1 = atoms.positions[f2_indices[0]]
        for x in (f2_indices[1:]):
            mindis, minshift, posx = np.inf, None, None
            for shift in self.grids:
                shift = np.array(shift)
                cur_posx = atoms.positions[x] + np.dot(shift, cell)
                dis = np.linalg.norm(cur_posx - pos1)
                if dis < mindis:
                    mindis, minshift, posx = dis, shift, cur_posx
            atoms.positions[x] = posx

        return False # no reaction happens
    
    def update_positions_by_graph(self, atoms):
        """"""
        if self.graph_creator is None:
            return self.update_positions(atoms), []

        has_reaction = False
        
        # use graph
        ini_fragments = self.initial_fragments.copy()
        new_fragments = partition_fragments(self.graph_creator, atoms)
        #print(new_fragments)

        # - check if reaction happens
        same_frags, products = [], []
        for x in new_fragments:
            for y in ini_fragments:
                if x == y:
                    same_frags.append(x)
                    break
            else:
                products.append(x)
        if len(products) > 0:
            has_reaction = True
        else:
            # - adjust positions
            _ = self.update_positions(atoms)

        return has_reaction, products
    
    def step(self, f=None):
        """"""
        atoms = self.atoms
        if f is None:
            rforces = atoms.get_forces(apply_constraint=True).copy()
        else:
            rforces = f
        #print("rfmax: ", np.max(np.fabs(rforces)))

        pair_indices = list(product(*self.frag_indices))
        pair_indices = np.array(pair_indices).transpose()

        # TODO: find mic
        # TODO: better use graph
        # NOTE: change positions based on reference
        pair_shifts = np.zeros((pair_indices.shape[1],3))

        self.has_reaction, products = self.update_positions_by_graph(atoms)
        if not self.has_reaction:
            # bias
            #bias = force_function(atoms.positions, atomic_radii, pair_indices, pair_shifts)
            #print("energy", bias)
            aforces = -self.dfn(atoms.positions, self.atomic_radii, pair_indices, pair_shifts, self.gamma)
            #print(aforces[:2,:])

            bforces = rforces + aforces # biased forces
            #print("bfmax: ", np.max(np.fabs(bforces)))
            self.biased_forces = bforces

            super().step(f=bforces)
        else:
            self.biased_forces = np.zeros((3,3))
            print(f"reaction from {self.initial_fragments} with products {products}")

        return
    
    def converged(self):
        """"""

        return (self.biased_forces ** 2).sum(axis=1).max() < self.fmax ** 2
    
    def log(self, forces=None):
        """"""
        super().log(forces=self.biased_forces)

        return


def run(atoms):
    """"""
    atomic_numbers = atoms.get_atomic_numbers()
    atomic_radii = np.array([covalent_radii[i] for i in atomic_numbers])

    # ....
    print("fragments")
    target_indices = [0, 1, 2]
    #pair_indices = np.array(list(combinations(target_indices, 2))).transpose()
    pair_indices = np.array([[0,2],[1,1]])
    print(pair_indices)
    pair_shifts = np.zeros((pair_indices.shape[1],3))
    print(pair_shifts)

    dfn = grad(force_function, argnums=0)

    dyn = BFGS(atoms, trajectory="afir.traj")

    frames = [atoms.copy()]

    nsteps = 100
    for i in range(nsteps):
        print(f"AFIR step {i}")
        # real
        rforces = atoms.get_forces(apply_constraint=True).copy()
        print("rfmax: ", np.max(np.fabs(rforces)))
        # bias
        #bias = force_function(atoms.positions, atomic_radii, pair_indices, pair_shifts)
        #print("energy", bias)
        aforces = -dfn(atoms.positions, atomic_radii, pair_indices, pair_shifts)
        #print(aforces[:2,:])

        bforces = rforces + aforces # biased forces
        print("bfmax: ", np.max(np.fabs(bforces)))

        dyn.step(f=bforces)

        frames.append(atoms.copy())
    write("afir.xyz", frames)

    return

class AFIRSearch():

    def __init__(self, gmax=2.5, gintv=0.1, graph_creator=None):
        """
        """
        self.gmax = gmax # eV
        self.gintv = gintv # percent
        
        self.dyn_params = dict(
            fmax = 0.1, steps=100
        )

        self.graph_creator = graph_creator

        return

    def run(self, atoms, calc, target_indices=None):
        """
        """
        # --- some params
        gmax = self.gmax
        gintv = self.gintv

        fragments = partition_fragments(self.graph_creator, atoms)
        fragment_combinations = list(combinations(fragments, 2))

        print(fragments)
        print(fragment_combinations)

        # generate fragments from target atoms
        fragment_combinations = [
            #[[0,2,56], [1,52,53,57]]
            [[40], [36,38]]
        ]

        # run over fragments
        path_frames = [] # AFIR pathway
        for frag in fragment_combinations:
            print(f"run frag {frag}")
            cur_gcoef = 0.1 # should be a random number
            gamma = gmax*cur_gcoef
            while gamma <= gmax:
                gamma = gmax*cur_gcoef
                # run opt
                print("gamma: ", gamma)
                cur_atoms = atoms.copy()
                calc.reset()
                cur_atoms.calc = calc
                dyn = AFIR(cur_atoms, gamma=gamma, frag_indices=frag, graph_creator=self.graph_creator, trajectory="afir.traj")
                dyn.run(**self.dyn_params)
                # store results
                results = dict(
                    energy=float(cur_atoms.get_potential_energy()), 
                    free_energy=float(cur_atoms.get_potential_energy()),
                    forces=cur_atoms.get_forces().copy()
                )
                new_calc = SinglePointCalculator(cur_atoms, **results)
                cur_atoms.calc = new_calc
                print("energy: ", cur_atoms.get_potential_energy())
                path_frames.append(cur_atoms)
                # update
                cur_gcoef += gintv

                # TODO: check if reaction happens
                if dyn.has_reaction:
                    print("found new products...")
                    break
            
            # TODO: find highest point approximates TS
            # save trajectories...

        write("path.xyz", path_frames)

        return

    def irun(self):
        """ run single fragment combination
        """

        return

if __name__ == "__main__":
    from itertools import combinations
    from ase.io import read, write
    from GDPy.potential.manager import create_pot_manager
    pm = create_pot_manager(
        "/mnt/scratch2/users/40247882/pbe-oxides/eann-main/m08/AFIR-Test/pot-ase.yaml"
        #"/mnt/scratch2/users/40247882/pbe-oxides/eann-main/m09/ensemble/model-1/validations/pot-ase.yaml"
    )

    # opt
    def run_opt():
        dyn_params = {
          "backend": "ase",
          "method": "opt",
          "fmax": 0.05,
          "steps": 400,
          "repeat": 3
        }

        worker = pm.create_worker(dyn_params)

        atoms = read("/mnt/scratch2/users/40247882/pbe-oxides/eann-main/m08/AFIR-Test/IS.xyz")
        new_atoms = worker.minimise(atoms)
        write("IS_opt.xyz", new_atoms, columns=["symbols", "positions", "move_mask"])
    
    atoms = read("/mnt/scratch2/users/40247882/pbe-oxides/eann-main/m08/AFIR-Test/IS_opt.xyz")
    atoms.calc = pm.calc

    #run(atoms)
    #dyn = AFIR(atoms, gamma=0.85, frag_indices=[[0,2],[1]], trajectory="afir.traj")
    #dyn.run(fmax=0.05, steps=100)

    atoms = read("/mnt/scratch2/users/40247882/pbe-oxides/eann-main/m09/ga/rs/uged-calc_candidates.xyz", "0")

    # --- check graph
    graph_params = dict(
        pbc_grid =  [1, 1, 0], # NOTE: large adsorbate cross many cells?
        graph_radius = 0,
        # - neigh
        covalent_ratio = 1.2,
        skin = 0.25,
        adsorbate_elements = ["C", "O"],
        #coordination_numbers = [3],
        #site_radius = 3
    )
    from GDPy.graph.creator import StruGraphCreator
    creator = StruGraphCreator(**graph_params)

    rs = AFIRSearch(gmax=2.5, gintv=0.1, graph_creator=creator)
    rs.run(atoms, pm.calc)

    pass