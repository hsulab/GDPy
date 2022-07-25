#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Artificial force induced reaction (AFIR)
"""

from ast import For
import time
from tkinter import Frame

import numpy as np

from pathlib import Path
import itertools
from itertools import product, combinations
from typing import NamedTuple, List
from dataclasses import dataclass, field

import numpy as np
from jax import numpy as jnp
from jax import grad, jit

from ase import Atoms
from ase.formula import Formula
from ase.io import read, write
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list, natural_cutoffs, NeighborList
from ase.optimize import BFGS
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms

from GDPy.graph.utils import unpack_node_name
from GDPy.graph.creator import StruGraphCreator
from GDPy.builder.constraints import parse_constraint_info


@dataclass(repr=False, eq=False)
class Adsorbate:
    
    # NOTE: dataclass is more suitable?

    symbols: List[str] = field(default_factory=list)
    indices: List[int] = field(default_factory=list)
    shifts: List[List] = field(default_factory=list) # in cartesian coordinates
    positions: List[List] = field(default_factory=list)

    name: str = field(default="adsorbate", init=False)
    cop: List[float] = field(default_factory=list, init=False) # centre of position

    def __eq__(self, other):
        """"""
        # TODO: change this to graph comparasion since they may have isomers
        #return tuple(self.indices.sort()) == tuple(other.sort())
        #print(sorted(self.indices))
        #print(sorted(other.indices))
        #return (sorted(self.indices) == sorted(other.indices))
        # NOTE: indices are sorted in post_init
        return (self.indices == other.indices)
    
    def __post_init__(self):
        """"""
        # - sort data by atomic index
        sorted_mapping = sorted(range(len(self.indices)), key=lambda x: self.indices[x])
        self.symbols = np.array([self.symbols[x] for x in sorted_mapping])
        self.indices = [self.indices[x] for x in sorted_mapping]
        self.shifts = np.array([self.shifts[x] for x in sorted_mapping])
        self.positions = np.array([self.positions[x] for x in sorted_mapping])

        self.cop = np.mean(self.positions+self.shifts, axis=0)
        #print("xxx: ", self.positions+self.shifts)
        name = ""
        for s, i in zip(self.symbols, self.indices):
            name += f"{s}({i})"
        self.name = name

        return

    def __repr__(self) -> str:
        content = f"\n--- {self.name} ---\n"
        for s, i, p, o in zip(self.symbols, self.indices, self.positions, self.shifts):
            content += ("{:<8s}  {:<8d}  "+"[{:<4.2f} {:<4.2f} {:<4.2f}] "+"[{:<4.2f} {:<4.2f} {:<4.2f}]\n").format(
                s, i, *p, *o
            )
        content += ("COP: "+"[{:<4.2f} {:<4.2f} {:<4.2f}]").format(*self.cop)

        return content


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

def partition_fragments(creator, atoms, target_species=None) -> List[Adsorbate]:
    """ generate fragments from a single component
        use graph to find adsorbates and group them into fragments
    """
    creator.generate_graph(atoms)

    chem_envs = creator.extract_chem_envs()
    
    cell = atoms.get_cell(complete=True)[:]

    fragments = []
    for i, adsorbate in enumerate(chem_envs):
        ads_symbols, ads_indices, ads_shifts = [], [], []
        ads_positions = []
        for (u, d) in adsorbate.nodes.data():
            #print(i, u, d)
            #if d["central_ads"] and d.get("ads"):
            if d.get("ads"):
                chem_sym, idx, offset = unpack_node_name(u)
                ads_symbols.append(chem_sym)
                ads_indices.append(idx)
                ads_shifts.append(np.zeros(3))
                cartshift = np.dot(offset, cell)
                atoms.positions[idx] += cartshift
                ads_positions.append(atoms.positions[idx])
        #ads_indices.sort()
        ads = Adsorbate(
            symbols=ads_symbols, indices=ads_indices, shifts=ads_shifts, positions=ads_positions
        )
        # - TODO check if valid species
        if target_species is None:
            fragments.append(ads)
        else:
            # check species
            # if ads.symbols == species.symbols ???
            pass
        #print(i, ads)

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
    pair_positions = jnp.take(positions, pair_indices, axis=0)
    #print(pair_positions.shape)
    dvecs = pair_positions[0] - pair_positions[1] + pair_shifts
    #print(dvecs)
    distances = jnp.linalg.norm(dvecs, axis=1)
    #print(distances)

    pair_radii = jnp.take(covalent_radii, pair_indices, axis=0)
    #print(pair_radii)
    #print((pair_radii[0]+pair_radii[1]))

    weights = ((pair_radii[0]+pair_radii[1])/distances)**6
    #print(weights)

    bias = alpha * jnp.sum(weights*distances) / jnp.sum(weights)
    #print(bias)

    return bias

class AFIR(BFGS):

    """ integrate with graph calculation
    """

    has_reaction = False

    def __init__(self, atoms, gamma, fragments, graph_creator=None, **kwargs):
        """"""
        super().__init__(atoms, **kwargs)

        self.graph_creator = graph_creator

        #if self.graph_creator is not None:
        #    self.initial_fragments = partition_fragments(self.graph_creator, atoms)
        #else:
        #    self.initial_fragments = fragments
        assert len(fragments) == 2, "Number of fragments should be two."
        self.initial_fragments = fragments
        self.frag_names = [x.name for x in self.initial_fragments]
        print("names: ", self.frag_names)
        
        self.reacted, self.products = None, None

        # NOTE: here should be a list of adsorbate names
        frag_indices = [x.indices for x in self.initial_fragments]
        self.pair_indices = list(product(*frag_indices))
        self.pair_indices = np.array(self.pair_indices).transpose()
        self.pair_shifts = np.zeros((self.pair_indices.shape[1],3))

        # - bias setting
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


        # - neigh calculation?
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

        # use graph
        ini_fragments = self.initial_fragments
        new_fragments = partition_fragments(self.graph_creator, atoms) # TODO: use selected indices?
        #new_frag_names = [x.name for x in new_fragments]

        # - check if reaction happens
        # NOTE: only focus on reaction by selected framents, not check all possible reactions
        unreacted = []
        products, reacted = [], []
        for x in ini_fragments:
            for y in new_fragments:
                if x == y:
                    unreacted.append(y) # add adsorbates with current positions
                    break
            else:
                reacted.append(x)

        if len(unreacted) < 2: # indicate someone has reacted
            # - find products
            reactant_indices = list(itertools.chain(*[x.indices for x in reacted]))
            for x in new_fragments:
                for i in x.indices:
                    if i in reactant_indices:
                        products.append(x)
                        break
                else:
                    # current frag is irresiponsible to reaction
                    pass
            # NOTE: move products
            # they have been moved when partition fragments
        else:
            #print("no reaction happens...")
            f1, f2 = unreacted
            # TODO: use connected atom nodes as an adsorbate
            # - adjust positions
            #print(f1)
            #print(f2)
            mindis, mincartshift = np.inf, np.zeros(3)
            cell = atoms.get_cell(complete=True)[:]
            for shift in self.grids:
                shift = np.array(shift)
                cartshift = np.dot(shift, cell) # shift in cartesian
                cur_f2_cop = f2.cop + cartshift
                dis = np.linalg.norm(cur_f2_cop - f1.cop)
                if dis < mindis:
                    mindis, mincartshift = dis, cartshift
            
            for i in f2.indices:
                atoms.positions[i] += mincartshift
            #print("mindis: ", mindis)
            #print("minshift: ", mincartshift)

        return reacted, products
    
    def step(self, f=None):
        """"""
        atoms = self.atoms
        if f is None:
            rforces = atoms.get_forces(apply_constraint=True).copy()
        else:
            rforces = f
        #print("rfmax: ", np.max(np.fabs(rforces)))

        # TODO: find mic
        # TODO: better use graph
        # NOTE: change positions based on reference

        # NOTE: early stop? stop opt when new molecules are formed even when forces are not converged
        reacted, products = self.update_positions_by_graph(atoms)

        # bias
        #bias = force_function(atoms.positions, atomic_radii, pair_indices, pair_shifts)
        #print("energy", bias)
        aforces = -self.dfn(atoms.positions, self.atomic_radii, self.pair_indices, self.pair_shifts, self.gamma)
        #print(aforces[:2,:])

        bforces = rforces + aforces # biased forces
        #print("bfmax: ", np.max(np.fabs(bforces)))
        self.biased_forces = bforces

        super().step(f=bforces)

        if len(reacted) > 0:
            #self.biased_forces = np.zeros((3,3))
            #print(f"reaction from {self.initial_fragments} with products {products}")
            #print(f"{[x.name for x in reacted]} have reacted into {[x.name for x in products]}...")
            self.reacted, self.products = reacted, products

        return
    
    def converged(self):
        """"""

        return (self.biased_forces ** 2).sum(axis=1).max() < self.fmax ** 2
    
    def log(self, forces=None):
        """"""
        super().log(forces=self.biased_forces)

        return


class AFIRSearch():

    """ TODO: transform this into a dynamics object?
        reaction event search 
            in: one single structure 
            out: a bunch of opt trajs + a pseudo pathway
    """

    default_parameters = dict(
        dynamics = dict(
            fmax = 0.1, steps = 100
        )
    )

    nfragments_to_reaction = 2

    run_NEB = False

    def __init__(
        self, 
        directory=Path.cwd()/"results",
        target_pairs = None,
        gmax=2.5, # eV, maximum gamma
        ginit=None, # eV, inital gamma
        gintv=0.1, # percentage
        dyn_params=None,
        graph_params=None,
        seed=None, 
    ):
        """
        """
        # - parse target pairs
        if target_pairs is not None:
            assert len(target_pairs) == self.nfragments_to_reaction, "incorrect number of input target pairs"
            target_pairs_ = []
            for s in target_pairs:
                target_pairs_.append(Formula(s))
        else:
            target_pairs_ = None
        self.target_pairs = target_pairs_ # target fragment pairs

        self.gmax = gmax # eV
        self.ginit = ginit # eV
        self.gintv = gintv # percent
        
        self.dyn_params = dyn_params
        assert self.dyn_params is not None, "AFIR should have dynamics."

        if graph_params is not None:
            self.graph_creator = StruGraphCreator(**graph_params)
        else:
            self.graph_creator = None

        # - check outputs
        self.directory = directory

        # - assign random seeds
        if seed is None:
            # TODO: need a random number to be logged
            self.rng = np.random.default_rng()
        elif isinstance(seed, int):
            self.rng = np.random.default_rng(seed)

        return

    def run(self, atoms, calc):
        """ run all fragment pairs in one single structure
            output summary is in main.out
        """
        # --- prepare fragments
        fragments = partition_fragments(self.graph_creator, atoms)
        fragment_combinations = list(combinations(fragments, self.nfragments_to_reaction))

        # --- find valid fragment pairs 
        fragment_pairs = []
        if self.target_pairs is not None:
            for p in fragment_combinations:
                # - TODO: move this to adsorbate class???
                s0, s1 = Formula("".join(p[0].symbols.tolist())), Formula("".join(p[1].symbols.tolist()))
                #print("target: ", self.target_pairs)
                #print(s0, s1)
                if (
                    (s0 == self.target_pairs[0] and s1 == self.target_pairs[1]) or
                    (s1 == self.target_pairs[0] and s0 == self.target_pairs[1])
                ):
                    fragment_pairs.append(p)
            fragment_combinations = fragment_pairs
        # TODO: further select fragments based on distance???

        #print("fragments: ", fragments)
        print("fragments: ", [f.name for f in fragments])
        print("reaction pairs:", [p[0].name+"-"+p[1].name for p in fragment_combinations])

        #print("combinations: ", fragment_combinations)
        #exit()

        # TEST: generate fragments from target atoms
        #test_names = ["C(36)O(38)", "O(40)"]
        #fragment_combinations = [[
        #    x for x in fragments if x.name in test_names
        #]]

        # - check output directory before running
        main_out = self.directory / "main.out"
        if main_out.exists() and self.directory.exists():
            print("results exists, may cause inconsistent results...")
        else:
            self.directory.mkdir()

        with open(main_out, "w") as fopen:
            fopen.write("# count frag\n")

        # - run over fragments
        for ifrag, frag in enumerate(fragment_combinations):
            # - info for cur combination
            frag_names = [x.name for x in frag]
            print(f"run frag {frag_names[0]} - {frag_names[1]}")
            with open(main_out, "a") as fopen:
                fopen.write("{:>8d}  {}\n".format(ifrag, frag_names))

            fres_path = self.directory / ("f"+str(ifrag))
            fres_path.mkdir(exist_ok=True) # fragment results path
            # - run actual AFIR search
            path_frames = self.irun(atoms, fres_path, calc, frag)

        return

    def irun(self, init_atoms, fres_path, calc, frag) -> List[Atoms]:
        """ run single fragment combination
            output summary is in data.out
        """
        # --- some params
        gmax = self.gmax
        gintv = self.gintv

        with open(fres_path / "data.out", "w") as fopen:
            fopen.write("# count time gamma energy bfmax fmax reaction\n")

        # - results
        found_reaction = False
        gcount = 0 # count for each gamma
        path_frames = [] # AFIR pathway

        # - random
        if self.ginit is None:
            cur_gcoef = self.rng.random() # should be a random number
            gamma = gmax*cur_gcoef
        else:
            gamma = self.ginit
        print("Initial Gamma: ", gamma)

        while gamma <= gmax:
            print("----- gamma {:<8.4f} -----".format(gamma))
            st = time.time()

            # - prepare dirs
            cur_gdir = fres_path / ("g"+str(gcount))
            cur_gdir.mkdir(exist_ok=True)
            cur_traj = str(cur_gdir/"afir.traj")
            cur_logfile = cur_gdir / "afir.log"
            #cur_logfile = "-"

            # - run opt
            cur_atoms = init_atoms.copy()
            calc.reset()
            cur_atoms.calc = calc

            # TODO: some calculators need reset directory
            calc.directory = str(cur_gdir / "calculation")
            dyn = AFIR(
                cur_atoms, gamma=gamma, fragments=frag, graph_creator=self.graph_creator, 
                trajectory=cur_traj, logfile=cur_logfile
            )
            dyn.run(**self.dyn_params)

            # - store results
            results = dict(
                energy=float(cur_atoms.get_potential_energy()), 
                free_energy=float(cur_atoms.get_potential_energy()),
                forces=cur_atoms.get_forces().copy()
            )
            new_calc = SinglePointCalculator(cur_atoms, **results)
            cur_atoms.calc = new_calc
            print("energy: ", cur_atoms.get_potential_energy())
            print("fmax: ", np.max(np.fabs(cur_atoms.get_forces(apply_constraint=True))))
            path_frames.append(cur_atoms)

            traj_frames = read(cur_traj, ":")
            write(cur_gdir/"traj.xyz", traj_frames)

            et = time.time()

            bfmax = (dyn.biased_forces ** 2).sum(axis=1).max()
            fmax = (cur_atoms.get_forces() ** 2).sum(axis=1).max()

            content = "{:>4d}  {:>8.4f}  {:>8.4f}  {:>12.4f}  {:>8.4f}  {:>8.4f}".format(
                gcount, et-st, gamma, cur_atoms.get_potential_energy(), bfmax, fmax
            )
            if dyn.reacted is not None:
                content += f":  {[x.name for x in dyn.reacted]} -> {[x.name for x in dyn.products]}"
            content += "\n"

            with open(fres_path / "data.out", "a") as fopen:
                fopen.write(content)

            # - update
            gamma += gmax*gintv
            gcount += 1

            # TODO: check if reaction happens
            #if dyn.has_reaction:
            #    print("found new products...")
            #    break

            if dyn.reacted is not None:
                print("reaction happens...")
                # NOTE: add constraint?
                # ensure at least three optimisations for each fragment pair
                if gcount <= 2:
                    # NOTE: too large init gamma, then search back
                    print("too large gamma, search back...")
                    gamma /= 2.
                else:
                    found_reaction = True
                    break
        
        # - optimised last frame if found reaction
        #   since FS should be determined
        if found_reaction:
            print("----- final state -----")
            # - prepare dirs
            cur_gdir = fres_path / ("g"+str(gcount))
            cur_gdir.mkdir(exist_ok=True)
            cur_traj = str(cur_gdir/"bfgs.traj")
            cur_logfile = cur_gdir / "bfgs.log"

            cur_atoms = path_frames[-1].copy()
            calc.reset()
            cur_atoms.calc = calc
            
            # - run calculation
            calc.directory = str(cur_gdir / "calculation")
            dyn = BFGS(cur_atoms, logfile=cur_logfile, trajectory=cur_traj)
            dyn.run(**self.dyn_params)

            # - store results
            results = dict(
                energy=float(cur_atoms.get_potential_energy()), 
                free_energy=float(cur_atoms.get_potential_energy()),
                forces=cur_atoms.get_forces().copy()
            )
            new_calc = SinglePointCalculator(cur_atoms, **results)
            cur_atoms.calc = new_calc
            print("energy: ", cur_atoms.get_potential_energy())
            print("fmax: ", np.max(np.fabs(cur_atoms.get_forces(apply_constraint=True))))
            path_frames.append(cur_atoms)

            traj_frames = read(cur_traj, ":")
            write(cur_gdir/"traj.xyz", traj_frames)

            et = time.time()

            content = "{:>4d}  {:>8.4f}  {:>8.4f}  {:>12.4f}  ".format(gcount, et-st, gamma, cur_atoms.get_potential_energy())
            content += ":  final state\n"

            with open(fres_path / "data.out", "a") as fopen:
                fopen.write(content)
        
        # - save trajectories...
        # TODO: sort frames by gamma value
        write(fres_path/"pseudo_path.xyz", path_frames)

        # - if run NEB calculation to get a better pathway
        if self.run_NEB:
            pass

        # TODO: find highest point approximates TS
        ## save trajectories...
        #write(fres_path/"pseudo_path.xyz", path_frames)

        #exit()

        return path_frames

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

    #atoms = read("/mnt/scratch2/users/40247882/pbe-oxides/eann-main/m09/ga/rs/uged-calc_candidates.xyz", "0")
    atoms = read("/mnt/scratch2/users/40247882/pbe-oxides/eann-main/m09/ga/rs/test-mod.xsd")

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

    dyn_params = dict(fmax = 0.1, steps=100)

    rs = AFIRSearch(gmax=2.5, gintv=0.2, dyn_params=dyn_params, graph_params=graph_params, seed=1112)
    rs.run(atoms, pm.calc)

    pass